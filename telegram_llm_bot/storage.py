import json
import os
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace

from .constants import CONFIG_VERSION, DEFAULT_PROVIDER_ID, DEFAULT_PROVIDER_NAME
from .session import ChatSessionKey
from .utils import (
    normalize_reasoning_effort,
    normalize_optional_config_text,
    normalize_required_text,
    slugify_provider_id,
)


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    name: str
    base_url: str | None
    api_key: str
    default_model: str
    current_model: str
    reasoning_effort: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "current_model": self.current_model,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class RuntimeConfigV2:
    version: int
    current_provider_id: str
    providers: tuple[ProviderConfig, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "current_provider_id": self.current_provider_id,
            "providers": [provider.as_json() for provider in self.providers],
        }


class SQLiteChatStore:
    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    user_id INTEGER NOT NULL,
                    thread_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(chat_messages)").fetchall()
            }
            if "chat_id" not in columns:
                self._conn.execute("ALTER TABLE chat_messages ADD COLUMN chat_id INTEGER")
            if "thread_id" not in columns:
                self._conn.execute("ALTER TABLE chat_messages ADD COLUMN thread_id INTEGER")
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id_id
                ON chat_messages (user_id, id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages (chat_id, user_id, thread_id, id)
                """
            )
            self._conn.commit()

    def append_message(self, session: ChatSessionKey, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chat_messages (chat_id, user_id, thread_id, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.chat_id,
                    session.user_id,
                    session.thread_id,
                    role,
                    content,
                ),
            )
            self._conn.commit()

    def get_recent_messages(
        self,
        session: ChatSessionKey,
        limit: int,
    ) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE (
                    chat_id = ?
                    AND user_id = ?
                    AND COALESCE(thread_id, -1) = ?
                )
                OR (
                    ? = 1
                    AND chat_id IS NULL
                    AND user_id = ?
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    session.chat_id,
                    session.user_id,
                    session.normalized_thread_id,
                    1 if session.supports_legacy_private_history else 0,
                    session.user_id,
                    limit,
                ),
            ).fetchall()
        rows.reverse()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def clear_session_history(self, session: ChatSessionKey) -> None:
        with self._lock:
            self._conn.execute(
                """
                DELETE FROM chat_messages
                WHERE (
                    chat_id = ?
                    AND user_id = ?
                    AND COALESCE(thread_id, -1) = ?
                )
                OR (
                    ? = 1
                    AND chat_id IS NULL
                    AND user_id = ?
                )
                """,
                (
                    session.chat_id,
                    session.user_id,
                    session.normalized_thread_id,
                    1 if session.supports_legacy_private_history else 0,
                    session.user_id,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SlidingWindowRateLimiter:
    _PURGE_INTERVAL = 500

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.events: dict[int, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()
        self._calls_since_purge = 0

    def allow(self, user_id: int) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self.lock:
            self._calls_since_purge += 1
            if self._calls_since_purge >= self._PURGE_INTERVAL:
                self._calls_since_purge = 0
                stale = [uid for uid, q in self.events.items() if not q or q[-1] <= cutoff]
                for uid in stale:
                    del self.events[uid]

            bucket = self.events[user_id]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return False, retry_after
            bucket.append(now)
            return True, 0


class RuntimeConfigStore:
    def __init__(self, config_path: str, bootstrap_provider: ProviderConfig) -> None:
        self.path = config_path
        self._bootstrap_provider = bootstrap_provider
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._config = self._load_initial_config()

    def _build_bootstrap_config(self) -> RuntimeConfigV2:
        if not self._bootstrap_provider.api_key:
            raise RuntimeError(
                "Missing provider config. Set OPENAI_API_KEY in .env or create "
                f"{self.path} with at least one provider."
            )
        return RuntimeConfigV2(
            version=CONFIG_VERSION,
            current_provider_id=self._bootstrap_provider.id,
            providers=(self._bootstrap_provider,),
        )

    def _provider_from_legacy_payload(self, payload: dict[str, object]) -> ProviderConfig:
        api_key = normalize_optional_config_text(payload.get("openai_api_key"))
        if api_key is None:
            api_key = self._bootstrap_provider.api_key
        if not api_key:
            raise RuntimeError(
                "Legacy config migration requires openai_api_key or OPENAI_API_KEY."
            )

        model = normalize_optional_config_text(payload.get("openai_model"))
        if model is None:
            model = self._bootstrap_provider.default_model
        base_url = normalize_optional_config_text(payload.get("openai_base_url"))
        if "openai_base_url" not in payload:
            base_url = self._bootstrap_provider.base_url

        return ProviderConfig(
            id=DEFAULT_PROVIDER_ID,
            name=DEFAULT_PROVIDER_NAME,
            base_url=base_url,
            api_key=api_key,
            default_model=model,
            current_model=model,
            reasoning_effort=self._bootstrap_provider.reasoning_effort,
        )

    def _provider_from_payload(self, payload: object) -> ProviderConfig:
        if not isinstance(payload, dict):
            raise RuntimeError("Each provider entry in config.json must be an object.")

        provider_id = normalize_required_text(payload.get("id"), "providers[].id")
        name = normalize_optional_config_text(payload.get("name")) or provider_id
        api_key = normalize_required_text(
            payload.get("api_key"),
            f"providers[{provider_id}].api_key",
        )
        default_model = normalize_required_text(
            payload.get("default_model"),
            f"providers[{provider_id}].default_model",
        )
        current_model = (
            normalize_optional_config_text(payload.get("current_model")) or default_model
        )
        base_url = normalize_optional_config_text(payload.get("base_url"))
        reasoning_effort = normalize_reasoning_effort(payload.get("reasoning_effort"))

        return ProviderConfig(
            id=provider_id,
            name=name,
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
            current_model=current_model,
            reasoning_effort=reasoning_effort,
        )

    def _parse_v2_config(self, raw_payload: dict[str, object]) -> RuntimeConfigV2:
        providers_payload = raw_payload.get("providers")
        if not isinstance(providers_payload, list):
            raise RuntimeError("config.json version 2 must contain a providers list.")

        providers = tuple(self._provider_from_payload(item) for item in providers_payload)
        current_provider_id = normalize_required_text(
            raw_payload.get("current_provider_id"),
            "current_provider_id",
        )
        config = RuntimeConfigV2(
            version=CONFIG_VERSION,
            current_provider_id=current_provider_id,
            providers=providers,
        )
        self._validate_config(config)
        return config

    def _validate_config(self, config: RuntimeConfigV2) -> None:
        if not config.providers:
            raise RuntimeError("Runtime config must contain at least one provider.")

        seen_ids: set[str] = set()
        for provider in config.providers:
            if provider.id in seen_ids:
                raise RuntimeError(f"Duplicate provider id in config.json: {provider.id}")
            seen_ids.add(provider.id)
            if not provider.api_key:
                raise RuntimeError(f"Provider {provider.id} is missing an api_key.")
            if not provider.default_model:
                raise RuntimeError(f"Provider {provider.id} is missing a default_model.")
            if not provider.current_model:
                raise RuntimeError(f"Provider {provider.id} is missing a current_model.")

        if config.current_provider_id not in seen_ids:
            raise RuntimeError(
                f"current_provider_id {config.current_provider_id!r} does not match any provider."
            )

    def _load_initial_config(self) -> RuntimeConfigV2:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw_payload = json.load(handle)
        except FileNotFoundError:
            config = self._build_bootstrap_config()
            with self._lock:
                self._persist_locked(config)
            return config
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {self.path}.") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read runtime config {self.path}.") from exc

        if not isinstance(raw_payload, dict):
            raise RuntimeError(f"{self.path} must contain a JSON object.")

        if "providers" in raw_payload or "current_provider_id" in raw_payload:
            return self._parse_v2_config(raw_payload)

        if (
            "openai_api_key" in raw_payload
            or "openai_model" in raw_payload
            or "openai_base_url" in raw_payload
        ):
            provider = self._provider_from_legacy_payload(raw_payload)
            config = RuntimeConfigV2(
                version=CONFIG_VERSION,
                current_provider_id=provider.id,
                providers=(provider,),
            )
            self._validate_config(config)
            with self._lock:
                self._persist_locked(config)
            return config

        raise RuntimeError(
            f"Unsupported runtime config format in {self.path}. "
            "Expected version 2 provider config or legacy flat OpenAI keys."
        )

    def _persist_locked(self, config: RuntimeConfigV2) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, temp_path = tempfile.mkstemp(
            dir=directory,
            prefix=".config.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(config.as_json(), handle, indent=2, ensure_ascii=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def current(self) -> RuntimeConfigV2:
        with self._lock:
            return self._config

    def get_provider(self, provider_id: str) -> ProviderConfig:
        with self._lock:
            for provider in self._config.providers:
                if provider.id == provider_id:
                    return provider
        raise KeyError(provider_id)

    def get_current_provider(self) -> ProviderConfig:
        with self._lock:
            provider_id = self._config.current_provider_id
            for provider in self._config.providers:
                if provider.id == provider_id:
                    return provider
        raise RuntimeError("Current provider is missing from runtime config.")

    def set_current_provider(self, provider_id: str) -> ProviderConfig:
        with self._lock:
            if provider_id == self._config.current_provider_id:
                for provider in self._config.providers:
                    if provider.id == provider_id:
                        return provider
            provider = next(
                (item for item in self._config.providers if item.id == provider_id),
                None,
            )
            if provider is None:
                raise KeyError(provider_id)
            updated_config = replace(self._config, current_provider_id=provider_id)
            self._persist_locked(updated_config)
            self._config = updated_config
            return provider

    def set_provider_current_model(self, provider_id: str, model_id: str) -> ProviderConfig:
        normalized_model = normalize_required_text(model_id, "model_id")
        with self._lock:
            updated_provider: ProviderConfig | None = None
            providers: list[ProviderConfig] = []
            for provider in self._config.providers:
                if provider.id == provider_id:
                    updated_provider = replace(provider, current_model=normalized_model)
                    providers.append(updated_provider)
                else:
                    providers.append(provider)
            if updated_provider is None:
                raise KeyError(provider_id)
            updated_config = replace(self._config, providers=tuple(providers))
            self._persist_locked(updated_config)
            self._config = updated_config
            return updated_provider

    def set_provider_reasoning_effort(
        self,
        provider_id: str,
        reasoning_effort: str | None,
    ) -> ProviderConfig:
        normalized_reasoning = normalize_reasoning_effort(reasoning_effort)
        with self._lock:
            updated_provider: ProviderConfig | None = None
            providers: list[ProviderConfig] = []
            for provider in self._config.providers:
                if provider.id == provider_id:
                    updated_provider = replace(
                        provider,
                        reasoning_effort=normalized_reasoning,
                    )
                    providers.append(updated_provider)
                else:
                    providers.append(provider)
            if updated_provider is None:
                raise KeyError(provider_id)
            updated_config = replace(self._config, providers=tuple(providers))
            self._persist_locked(updated_config)
            self._config = updated_config
            return updated_provider

    def add_provider(
        self,
        name: str,
        base_url: str | None,
        api_key: str,
        default_model: str,
        reasoning_effort: str | None = None,
    ) -> ProviderConfig:
        normalized_name = normalize_required_text(name, "provider name")
        normalized_api_key = normalize_required_text(api_key, "provider api_key")
        normalized_model = normalize_required_text(default_model, "provider default_model")
        normalized_base_url = normalize_optional_config_text(base_url)
        normalized_reasoning = normalize_reasoning_effort(reasoning_effort)

        with self._lock:
            existing_ids = {provider.id for provider in self._config.providers}
            provider_id = slugify_provider_id(normalized_name, existing_ids)
            provider = ProviderConfig(
                id=provider_id,
                name=normalized_name,
                base_url=normalized_base_url,
                api_key=normalized_api_key,
                default_model=normalized_model,
                current_model=normalized_model,
                reasoning_effort=normalized_reasoning,
            )
            updated_config = replace(
                self._config,
                providers=self._config.providers + (provider,),
            )
            self._persist_locked(updated_config)
            self._config = updated_config
            return provider

    def edit_provider(
        self,
        provider_id: str,
        name: str,
        base_url: str | None,
        api_key: str,
        default_model: str,
        reasoning_effort: str | None | object = None,
    ) -> ProviderConfig:
        normalized_name = normalize_required_text(name, "provider name")
        normalized_api_key = normalize_required_text(api_key, "provider api_key")
        normalized_model = normalize_required_text(default_model, "provider default_model")
        normalized_base_url = normalize_optional_config_text(base_url)
        preserve_reasoning = reasoning_effort is None
        normalized_reasoning = (
            None if preserve_reasoning else normalize_reasoning_effort(reasoning_effort)
        )

        with self._lock:
            updated_provider: ProviderConfig | None = None
            providers: list[ProviderConfig] = []
            for provider in self._config.providers:
                if provider.id == provider_id:
                    updated_provider = replace(
                        provider,
                        name=normalized_name,
                        base_url=normalized_base_url,
                        api_key=normalized_api_key,
                        default_model=normalized_model,
                        current_model=normalized_model,
                        reasoning_effort=(
                            provider.reasoning_effort
                            if preserve_reasoning
                            else normalized_reasoning
                        ),
                    )
                    providers.append(updated_provider)
                else:
                    providers.append(provider)
            if updated_provider is None:
                raise KeyError(provider_id)
            updated_config = replace(self._config, providers=tuple(providers))
            self._persist_locked(updated_config)
            self._config = updated_config
            return updated_provider

    def delete_provider(self, provider_id: str) -> RuntimeConfigV2:
        with self._lock:
            if len(self._config.providers) <= 1:
                raise ValueError("Cannot delete the last provider.")

            providers = tuple(
                provider
                for provider in self._config.providers
                if provider.id != provider_id
            )
            if len(providers) == len(self._config.providers):
                raise KeyError(provider_id)

            current_provider_id = self._config.current_provider_id
            if provider_id == current_provider_id:
                current_provider_id = providers[0].id

            updated_config = replace(
                self._config,
                current_provider_id=current_provider_id,
                providers=providers,
            )
            self._persist_locked(updated_config)
            self._config = updated_config
            return updated_config
