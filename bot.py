import asyncio
import base64
import html
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_TEXT_LIMIT = 4096
CONFIG_VERSION = 2
CONFIG_PATH = "./data/config.json"
DEFAULT_PROVIDER_ID = "default"
DEFAULT_PROVIDER_NAME = "Default"
MODELS_MENU_CACHE_KEY = "models_menu"
PROVIDER_WIZARD_KEY = "provider_wizard"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer env: {name}") from exc
    return max(minimum, value)


def get_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"Invalid float env: {name}") from exc
    return max(minimum, value)


def parse_user_id_set(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    user_ids: set[int] = set()
    for token in raw.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        try:
            user_ids.add(int(candidate))
        except ValueError as exc:
            raise RuntimeError(
                "WHITELIST_USER_IDS must be a comma-separated integer list"
            ) from exc
    return user_ids


def normalize_optional_config_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def normalize_required_text(value: object, field_name: str) -> str:
    normalized = normalize_optional_config_text(value)
    if normalized is None:
        raise RuntimeError(f"Missing required config field: {field_name}")
    return normalized


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    stripped = re.sub(
        r"(?is)<think>.*?(?:</think>|$)",
        "",
        text,
    )
    stripped = re.sub(r"(?is)</think>", "", stripped)
    return stripped


def _partial_tag_suffix_length(text: str, patterns: tuple[str, ...]) -> int:
    lower_text = text.lower()
    best = 0
    for pattern in patterns:
        limit = min(len(lower_text), len(pattern) - 1)
        for size in range(limit, 0, -1):
            if lower_text[-size:] == pattern[:size]:
                best = max(best, size)
                break
    return best


class ThinkTagFilter:
    def __init__(self) -> None:
        self._pending = ""
        self._inside_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._pending += chunk
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        output: list[str] = []
        while self._pending:
            lower_pending = self._pending.lower()
            if self._inside_think:
                close_index = lower_pending.find(THINK_CLOSE_TAG)
                if close_index == -1:
                    if final:
                        self._pending = ""
                    else:
                        keep = min(len(self._pending), len(THINK_CLOSE_TAG) - 1)
                        self._pending = self._pending[-keep:] if keep else ""
                    break
                self._pending = self._pending[close_index + len(THINK_CLOSE_TAG):]
                self._inside_think = False
                continue

            open_index = lower_pending.find(THINK_OPEN_TAG)
            close_index = lower_pending.find(THINK_CLOSE_TAG)

            if close_index != -1 and (open_index == -1 or close_index < open_index):
                if close_index > 0:
                    output.append(self._pending[:close_index])
                self._pending = self._pending[close_index + len(THINK_CLOSE_TAG):]
                continue

            if open_index != -1:
                if open_index > 0:
                    output.append(self._pending[:open_index])
                self._pending = self._pending[open_index + len(THINK_OPEN_TAG):]
                self._inside_think = True
                continue

            if final:
                output.append(self._pending)
                self._pending = ""
            else:
                keep = _partial_tag_suffix_length(
                    self._pending,
                    (THINK_OPEN_TAG, THINK_CLOSE_TAG),
                )
                emit_end = len(self._pending) - keep
                if emit_end > 0:
                    output.append(self._pending[:emit_end])
                    self._pending = self._pending[emit_end:]
                break

        return "".join(output)


def mask_secret(value: str, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return "(missing)"
    if len(value) <= prefix + suffix:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


def format_base_url(base_url: str | None) -> str:
    return base_url or "https://api.openai.com/v1 (default)"


def slugify_provider_id(name: str, existing_ids: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "provider"
    base = base[:24].rstrip("-") or "provider"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def normalize_base_url_input(text: str) -> str | None:
    candidate = text.strip()
    if not candidate or candidate.lower() in {"-", "none", "null", "official", "default"}:
        return None
    return candidate


def is_yes_text(text: str) -> bool:
    return text.strip().lower() in {"y", "yes", "save", "confirm"}


def is_no_text(text: str) -> bool:
    return text.strip().lower() in {"n", "no", "cancel"}


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    name: str
    base_url: str | None
    api_key: str
    default_model: str
    current_model: str

    def as_json(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "current_model": self.current_model,
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
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id_id
                ON chat_messages (user_id, id)
                """
            )
            self._conn.commit()

    def append_message(self, user_id: int, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_messages (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            self._conn.commit()

    def get_recent_messages(self, user_id: int, limit: int) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        rows.reverse()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def clear_user_history(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
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
        )

    def _provider_from_payload(self, payload: object) -> ProviderConfig:
        if not isinstance(payload, dict):
            raise RuntimeError("Each provider entry in config.json must be an object.")

        provider_id = normalize_required_text(payload.get("id"), "providers[].id")
        name = normalize_optional_config_text(payload.get("name")) or provider_id
        api_key = normalize_required_text(payload.get("api_key"), f"providers[{provider_id}].api_key")
        default_model = normalize_required_text(
            payload.get("default_model"),
            f"providers[{provider_id}].default_model",
        )
        current_model = normalize_optional_config_text(payload.get("current_model")) or default_model
        base_url = normalize_optional_config_text(payload.get("base_url"))

        return ProviderConfig(
            id=provider_id,
            name=name,
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
            current_model=current_model,
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
            provider = next((item for item in self._config.providers if item.id == provider_id), None)
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

    def add_provider(
        self,
        name: str,
        base_url: str | None,
        api_key: str,
        default_model: str,
    ) -> ProviderConfig:
        normalized_name = normalize_required_text(name, "provider name")
        normalized_api_key = normalize_required_text(api_key, "provider api_key")
        normalized_model = normalize_required_text(default_model, "provider default_model")
        normalized_base_url = normalize_optional_config_text(base_url)

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
    ) -> ProviderConfig:
        normalized_name = normalize_required_text(name, "provider name")
        normalized_api_key = normalize_required_text(api_key, "provider api_key")
        normalized_model = normalize_required_text(default_model, "provider default_model")
        normalized_base_url = normalize_optional_config_text(base_url)

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

            providers = tuple(provider for provider in self._config.providers if provider.id != provider_id)
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


TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
BOOTSTRAP_DEFAULT_MODEL = normalize_optional_config_text(os.getenv("OPENAI_MODEL")) or "gpt-4.1-mini"
BOOTSTRAP_PROVIDER = ProviderConfig(
    id=DEFAULT_PROVIDER_ID,
    name=DEFAULT_PROVIDER_NAME,
    base_url=normalize_optional_config_text(os.getenv("OPENAI_BASE_URL")),
    api_key=normalize_optional_config_text(os.getenv("OPENAI_API_KEY")) or "",
    default_model=BOOTSTRAP_DEFAULT_MODEL,
    current_model=BOOTSTRAP_DEFAULT_MODEL,
)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful Telegram assistant. Answer clearly and briefly.",
)
MAX_HISTORY_PAIRS = get_int_env("MAX_HISTORY_PAIRS", 10)
MAX_HISTORY_MESSAGES = MAX_HISTORY_PAIRS * 2
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/chat_history.db")
WHITELIST_USER_IDS = parse_user_id_set(os.getenv("WHITELIST_USER_IDS", ""))
RATE_LIMIT_COUNT = get_int_env("RATE_LIMIT_COUNT", 20)
RATE_LIMIT_WINDOW_SECONDS = get_int_env("RATE_LIMIT_WINDOW_SECONDS", 60)
STREAM_EDIT_INTERVAL_SECONDS = get_float_env("STREAM_EDIT_INTERVAL_SECONDS", 0.8, 0.1)
STREAM_MIN_CHARS_DELTA = get_int_env("STREAM_MIN_CHARS_DELTA", 24)
MODELS_MENU_PAGE_SIZE = get_int_env("MODELS_MENU_PAGE_SIZE", 8)

runtime_config_store = RuntimeConfigStore(CONFIG_PATH, BOOTSTRAP_PROVIDER)
chat_store = SQLiteChatStore(SQLITE_PATH)
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_SECONDS)


def authorized(user_id: int) -> bool:
    return not WHITELIST_USER_IDS or user_id in WHITELIST_USER_IDS


def get_runtime_config() -> RuntimeConfigV2:
    return runtime_config_store.current()


def get_current_provider() -> ProviderConfig:
    return runtime_config_store.get_current_provider()


def get_active_model() -> str:
    return get_current_provider().current_model


def build_openai_client(provider: ProviderConfig) -> AsyncOpenAI:
    client_kwargs = {"api_key": provider.api_key}
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    return AsyncOpenAI(**client_kwargs)


async def fetch_available_model_ids(provider: ProviderConfig) -> list[str]:
    client = build_openai_client(provider)
    models = await client.models.list()
    items = getattr(models, "data", None)
    if items is None:
        try:
            items = list(models)
        except TypeError:
            items = []

    ids: list[str] = []
    for item in items:
        model_id = getattr(item, "id", None)
        if model_id is None and isinstance(item, dict):
            model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            ids.append(model_id)

    return sorted(set(ids))


async def validate_provider_settings(
    base_url: str | None,
    api_key: str,
    default_model: str,
) -> tuple[bool, str, list[str]]:
    provider = ProviderConfig(
        id="validation",
        name="Validation",
        base_url=normalize_optional_config_text(base_url),
        api_key=api_key,
        default_model=default_model,
        current_model=default_model,
    )
    try:
        ids = await fetch_available_model_ids(provider)
    except Exception as exc:
        logging.exception("Provider validation failed while fetching model list")
        return False, str(exc), []

    if not ids:
        return False, "Provider returned no models.", []
    if default_model not in ids:
        return False, "Default model is not available from this provider.", ids
    return True, "", ids


def get_models_menu_cache(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object] | None:
    cached = context.user_data.get(MODELS_MENU_CACHE_KEY)
    if not isinstance(cached, dict):
        return None
    provider_id = cached.get("provider_id")
    ids = cached.get("ids")
    if isinstance(provider_id, str) and isinstance(ids, list) and all(isinstance(item, str) for item in ids):
        return {"provider_id": provider_id, "ids": ids}
    return None


def set_models_menu_cache(
    context: ContextTypes.DEFAULT_TYPE,
    provider_id: str,
    ids: list[str],
) -> None:
    context.user_data[MODELS_MENU_CACHE_KEY] = {
        "provider_id": provider_id,
        "ids": ids,
    }


def clear_models_menu_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop(MODELS_MENU_CACHE_KEY, None)


def get_provider_wizard(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object] | None:
    wizard = context.user_data.get(PROVIDER_WIZARD_KEY)
    if isinstance(wizard, dict):
        return wizard
    return None


def clear_provider_wizard(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop(PROVIDER_WIZARD_KEY, None)


def format_provider_line(provider: ProviderConfig, current_provider_id: str) -> str:
    marker = " [current]" if provider.id == current_provider_id else ""
    return (
        f"- <b>{html.escape(provider.name)}</b> "
        f"<code>{html.escape(provider.id)}</code>{marker}\n"
        f"  model: <code>{html.escape(provider.current_model)}</code>\n"
        f"  base_url: <code>{html.escape(format_base_url(provider.base_url))}</code>\n"
        f"  api_key: <code>{html.escape(mask_secret(provider.api_key))}</code>"
    )


def build_model_settings_text() -> str:
    runtime_config = get_runtime_config()
    current_provider = get_current_provider()
    providers_text = "\n".join(
        format_provider_line(provider, runtime_config.current_provider_id)
        for provider in runtime_config.providers
    )
    return (
        "<b>Current provider settings</b>\n"
        f"current_provider: <code>{html.escape(current_provider.name)}</code> "
        f"(<code>{html.escape(current_provider.id)}</code>)\n"
        f"current_model: <code>{html.escape(current_provider.current_model)}</code>\n"
        f"default_model_from_env: <code>{html.escape(BOOTSTRAP_PROVIDER.default_model)}</code>\n"
        f"base_url: <code>{html.escape(format_base_url(current_provider.base_url))}</code>\n"
        f"config_path: <code>{html.escape(CONFIG_PATH)}</code>\n"
        f"max_history_pairs: <code>{MAX_HISTORY_PAIRS}</code>\n"
        f"rate_limit: <code>{RATE_LIMIT_COUNT} requests / {RATE_LIMIT_WINDOW_SECONDS}s</code>\n"
        "streaming: <code>enabled</code>\n\n"
        "<b>Configured providers</b>\n"
        f"{providers_text}\n\n"
        "Use <code>/model &lt;model_id&gt;</code> for the current provider, "
        "or tap the buttons below."
    )


def build_model_settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Open providers", callback_data="providers:summary")],
            [InlineKeyboardButton("Open model picker", callback_data="providers:models_menu")],
        ]
    )


def build_provider_summary_text() -> str:
    runtime_config = get_runtime_config()
    current_provider = get_current_provider()
    providers_text = "\n".join(
        format_provider_line(provider, runtime_config.current_provider_id)
        for provider in runtime_config.providers
    )
    return (
        "<b>Providers</b>\n"
        f"current_provider: <code>{html.escape(current_provider.name)}</code> "
        f"(<code>{html.escape(current_provider.id)}</code>)\n"
        f"current_model: <code>{html.escape(current_provider.current_model)}</code>\n"
        f"config_path: <code>{html.escape(CONFIG_PATH)}</code>\n\n"
        f"{providers_text}\n\n"
        "<b>Management</b>\n"
        "<code>/provider_add</code>\n"
        "<code>/provider_edit &lt;provider_id&gt;</code>\n"
        "<code>/provider_delete &lt;provider_id&gt;</code>\n"
        "<code>/provider_cancel</code>"
    )


def build_provider_summary_keyboard() -> InlineKeyboardMarkup:
    runtime_config = get_runtime_config()
    rows: list[list[InlineKeyboardButton]] = []
    for provider in runtime_config.providers:
        label = (
            f"{provider.name} [current]"
            if provider.id == runtime_config.current_provider_id
            else provider.name
        )
        rows.append([InlineKeyboardButton(label, callback_data=f"providers:switch:{provider.id}")])
    rows.append([InlineKeyboardButton("Open model picker", callback_data="providers:models_menu")])
    return InlineKeyboardMarkup(rows)


def build_provider_picker_text() -> str:
    current_provider = get_current_provider()
    return (
        "<b>Select provider</b>\n"
        f"current_provider: <code>{html.escape(current_provider.name)}</code> "
        f"(<code>{html.escape(current_provider.id)}</code>)\n"
        "Tap a provider below to switch to it and view its models."
    )


def build_provider_picker_keyboard() -> InlineKeyboardMarkup:
    runtime_config = get_runtime_config()
    rows: list[list[InlineKeyboardButton]] = []
    for provider in runtime_config.providers:
        label = (
            f"{provider.name} [current]"
            if provider.id == runtime_config.current_provider_id
            else provider.name
        )
        rows.append([InlineKeyboardButton(label, callback_data=f"providers:models:{provider.id}")])
    rows.append([InlineKeyboardButton("<< Back", callback_data="providers:summary")])
    return InlineKeyboardMarkup(rows)


def build_models_menu_text(provider: ProviderConfig, ids: list[str], page: int) -> str:
    page_count = max(1, (len(ids) + MODELS_MENU_PAGE_SIZE - 1) // MODELS_MENU_PAGE_SIZE)
    page = max(0, min(page, page_count - 1))
    page_label = f"\nPage {page + 1}/{page_count}" if page_count > 1 else ""
    return (
        f"<b>Models</b> ({html.escape(provider.name)} | "
        f"<code>{html.escape(provider.id)}</code>) - {len(ids)} available\n"
        f"base_url: <code>{html.escape(format_base_url(provider.base_url))}</code>\n"
        f"api_key: <code>{html.escape(mask_secret(provider.api_key))}</code>\n"
        f"current_model: <code>{html.escape(provider.current_model)}</code>{page_label}\n"
        "Tap a model below to switch it for this provider."
    )


def build_models_keyboard(provider_id: str, ids: list[str], current_model: str, page: int) -> InlineKeyboardMarkup:
    page_count = max(1, (len(ids) + MODELS_MENU_PAGE_SIZE - 1) // MODELS_MENU_PAGE_SIZE)
    page = max(0, min(page, page_count - 1))
    start = page * MODELS_MENU_PAGE_SIZE
    end = min(start + MODELS_MENU_PAGE_SIZE, len(ids))

    rows: list[list[InlineKeyboardButton]] = []
    for index in range(start, end):
        model_id = ids[index]
        label = f"{model_id} [current]" if model_id == current_model else model_id
        rows.append(
            [InlineKeyboardButton(label, callback_data=f"models:set:{provider_id}:{index}")]
        )

    navigation: list[InlineKeyboardButton] = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton("< Prev", callback_data=f"models:page:{provider_id}:{page - 1}")
        )
    if page + 1 < page_count:
        navigation.append(
            InlineKeyboardButton("Next >", callback_data=f"models:page:{provider_id}:{page + 1}")
        )
    if navigation:
        rows.append(navigation)

    rows.append([InlineKeyboardButton("<< Providers", callback_data="models:backproviders")])
    return InlineKeyboardMarkup(rows)


async def edit_callback_text(
    update: Update,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    query = update.callback_query
    if not query:
        return
    try:
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except RetryAfter as exc:
        await asyncio.sleep(exc.retry_after)
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
    except BadRequest as exc:
        if "Message is not modified" not in str(exc):
            raise


def truncate_for_telegram(text: str) -> str:
    if len(text) <= TELEGRAM_TEXT_LIMIT:
        return text
    return text[: TELEGRAM_TEXT_LIMIT - 3] + "..."


def split_text_for_telegram(text: str) -> list[str]:
    if not text:
        return ["(empty)"]
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + TELEGRAM_TEXT_LIMIT, length)
        chunks.append(text[start:end])
        start = end
    return chunks


def markdown_to_telegram_html(text: str) -> str:
    if not text:
        return text

    slots: dict[str, str] = {}
    slot_index = 0

    def stash(value: str) -> str:
        nonlocal slot_index
        key = f"TGSLOT{slot_index}TOKEN"
        slot_index += 1
        slots[key] = value
        return key

    def replace_code_block(match: re.Match[str]) -> str:
        code_text = match.group(1).strip("\n")
        return stash(f"<pre><code>{html.escape(code_text)}</code></pre>")

    def replace_inline_code(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    def replace_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return stash(f'<a href="{url}">{label}</a>')

    text = re.sub(r"```(?:[^\n`]*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", replace_inline_code, text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", replace_link, text)

    escaped = html.escape(text)
    escaped = re.sub(r"(?m)^\s*#{1,6}\s+(.+)$", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "────────", escaped)
    escaped = re.sub(r"(?m)^(\s*)[-*]\s+(.+)$", r"\1• \2", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped, flags=re.DOTALL)
    escaped = re.sub(r"~~(.+?)~~", r"<s>\1</s>", escaped, flags=re.DOTALL)
    escaped = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", escaped, flags=re.DOTALL)
    escaped = re.sub(r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)", r"<i>\1</i>", escaped, flags=re.DOTALL)

    for key, value in slots.items():
        escaped = escaped.replace(key, value)

    return escaped


async def finalize_reply(message: Message, text: str) -> None:
    chunks = [markdown_to_telegram_html(chunk) for chunk in split_text_for_telegram(text)]

    async def _edit_with_markup_fallback(content: str) -> None:
        try:
            await message.edit_text(content, parse_mode=ParseMode.HTML)
            return
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            try:
                await message.edit_text(content, parse_mode=ParseMode.HTML)
                return
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

        try:
            await message.edit_text(content)
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            try:
                await message.edit_text(content)
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

    async def _reply_with_markup_fallback(content: str) -> None:
        try:
            await message.reply_text(content, parse_mode=ParseMode.HTML)
            return
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            try:
                await message.reply_text(content, parse_mode=ParseMode.HTML)
                return
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

        try:
            await message.reply_text(content)
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            try:
                await message.reply_text(content)
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

    await _edit_with_markup_fallback(chunks[0])
    for chunk in chunks[1:]:
        await _reply_with_markup_fallback(chunk)


async def try_delete_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int | None,
) -> None:
    if message_id is None:
        return
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        logging.info("Failed to delete message %s in chat %s", message_id, chat_id)


async def prompt_provider_wizard_step(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    wizard = get_provider_wizard(context)
    if wizard is None or not update.effective_chat:
        return

    step = wizard["step"]
    draft = wizard["draft"]
    mode = wizard["mode"]
    provider_id = wizard.get("provider_id")
    current_label = "existing value" if mode == "edit" else "draft value"

    if step == "name":
        text = (
            "Provider setup: send the provider display name.\n"
            f"Current {current_label}: <code>{html.escape(str(draft['name']))}</code>"
        )
    elif step == "base_url":
        text = (
            "Provider setup: send the base URL.\n"
            "Send <code>-</code>, <code>official</code>, or <code>none</code> for the official OpenAI endpoint.\n"
            f"Current {current_label}: <code>{html.escape(format_base_url(draft['base_url']))}</code>"
        )
    elif step == "api_key":
        text = (
            "Provider setup: send the API key for this provider.\n"
            f"Current {current_label}: <code>{html.escape(mask_secret(str(draft['api_key'])))}</code>"
        )
    elif step == "default_model":
        text = (
            "Provider setup: send the default model id for this provider.\n"
            f"Current {current_label}: <code>{html.escape(str(draft['default_model']))}</code>"
        )
    elif step == "confirm":
        header = "Confirm provider update" if mode == "edit" else "Confirm new provider"
        provider_id_line = (
            f"provider_id: <code>{html.escape(str(provider_id))}</code>\n"
            if provider_id
            else ""
        )
        text = (
            f"<b>{header}</b>\n"
            f"{provider_id_line}"
            f"name: <code>{html.escape(str(draft['name']))}</code>\n"
            f"base_url: <code>{html.escape(format_base_url(draft['base_url']))}</code>\n"
            f"api_key: <code>{html.escape(mask_secret(str(draft['api_key'])))}</code>\n"
            f"default_model: <code>{html.escape(str(draft['default_model']))}</code>\n\n"
            "Reply with <code>yes</code> to save or <code>no</code> to cancel."
        )
    else:
        return

    prompt = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode=ParseMode.HTML,
    )
    if step == "api_key":
        wizard["api_prompt_message_id"] = prompt.message_id
    else:
        wizard["api_prompt_message_id"] = None


def build_provider_draft_from_provider(provider: ProviderConfig) -> dict[str, str | None]:
    return {
        "name": provider.name,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "default_model": provider.default_model,
    }


async def start_provider_wizard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: str,
    provider: ProviderConfig | None = None,
) -> None:
    default_draft = build_provider_draft_from_provider(provider) if provider else {
        "name": "",
        "base_url": None,
        "api_key": "",
        "default_model": "",
    }
    context.user_data[PROVIDER_WIZARD_KEY] = {
        "mode": mode,
        "provider_id": provider.id if provider else None,
        "step": "name",
        "draft": default_draft,
        "api_prompt_message_id": None,
    }
    await prompt_provider_wizard_step(update, context)


async def save_provider_wizard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    wizard = get_provider_wizard(context)
    if wizard is None or not update.message:
        return

    draft = wizard["draft"]
    mode = str(wizard["mode"])
    provider_id = wizard.get("provider_id")
    base_url = normalize_optional_config_text(draft["base_url"])
    api_key = normalize_required_text(draft["api_key"], "provider api_key")
    default_model = normalize_required_text(draft["default_model"], "provider default_model")

    is_valid, error_message, ids = await validate_provider_settings(base_url, api_key, default_model)
    if not is_valid:
        if ids:
            sample_ids = "\n".join(f"- {model_id}" for model_id in ids[:15])
            more = "\n..." if len(ids) > 15 else ""
            wizard["step"] = "default_model"
            await update.message.reply_text(
                "Provider connection succeeded, but the default model was not found.\n"
                f"Available models include:\n{sample_ids}{more}\n\n"
                "Send another default model id."
            )
            await prompt_provider_wizard_step(update, context)
            return

        wizard["step"] = "base_url"
        await update.message.reply_text(
            "Failed to validate this provider before saving.\n"
            f"Error: {error_message or 'models.list() failed.'}\n\n"
            "Send the base URL again to continue, or use /provider_cancel."
        )
        await prompt_provider_wizard_step(update, context)
        return

    if mode == "add":
        provider = runtime_config_store.add_provider(
            name=normalize_required_text(draft["name"], "provider name"),
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
        )
        clear_provider_wizard(context)
        await update.message.reply_text(
            f"Provider added: {provider.name} ({provider.id}).\n"
            "Use /providers or /models to switch to it."
        )
        return

    if provider_id is None:
        clear_provider_wizard(context)
        await update.message.reply_text("Provider edit session lost its provider id. Please try again.")
        return

    provider = runtime_config_store.edit_provider(
        provider_id=provider_id,
        name=normalize_required_text(draft["name"], "provider name"),
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
    )
    clear_provider_wizard(context)
    await update.message.reply_text(
        f"Provider updated: {provider.name} ({provider.id}).\n"
        f"Current model for this provider reset to {provider.current_model}."
    )


async def handle_provider_wizard_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    wizard = get_provider_wizard(context)
    if wizard is None or not update.message or not update.message.text:
        return False

    text = update.message.text.strip()
    step = str(wizard["step"])
    draft = wizard["draft"]
    mode = str(wizard["mode"])

    if step == "name":
        draft["name"] = text
        wizard["step"] = "base_url"
        await prompt_provider_wizard_step(update, context)
        return True

    if step == "base_url":
        draft["base_url"] = normalize_base_url_input(text)
        wizard["step"] = "api_key"
        await prompt_provider_wizard_step(update, context)
        return True

    if step == "api_key":
        draft["api_key"] = text
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is not None:
            await try_delete_message(context, chat_id, update.message.message_id)
            await try_delete_message(
                context,
                chat_id,
                wizard.get("api_prompt_message_id"),
            )
        wizard["api_prompt_message_id"] = None
        wizard["step"] = "default_model"
        await prompt_provider_wizard_step(update, context)
        return True

    if step == "default_model":
        draft["default_model"] = text
        wizard["step"] = "confirm"
        await prompt_provider_wizard_step(update, context)
        return True

    if step == "confirm":
        if is_yes_text(text):
            await save_provider_wizard(update, context)
            return True
        if is_no_text(text):
            clear_provider_wizard(context)
            await update.message.reply_text("Provider setup cancelled.")
            return True
        await update.message.reply_text(
            "Reply with <code>yes</code> to save or <code>no</code> to cancel.",
            parse_mode=ParseMode.HTML,
        )
        return True

    if mode == "edit":
        await update.message.reply_text("Use /skip to keep the current value, or /provider_cancel to stop.")
        return True
    return True


async def render_provider_summary(update: Update) -> None:
    if update.message:
        await update.message.reply_text(
            build_provider_summary_text(),
            parse_mode=ParseMode.HTML,
            reply_markup=build_provider_summary_keyboard(),
        )
        return
    await edit_callback_text(
        update,
        build_provider_summary_text(),
        build_provider_summary_keyboard(),
    )


async def render_provider_picker(update: Update) -> None:
    if update.message:
        await update.message.reply_text(
            build_provider_picker_text(),
            parse_mode=ParseMode.HTML,
            reply_markup=build_provider_picker_keyboard(),
        )
        return
    await edit_callback_text(
        update,
        build_provider_picker_text(),
        build_provider_picker_keyboard(),
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await update.message.reply_text(
        "Hi, I am ready. Send me any message and I will stream an LLM reply.\n"
        "Commands:\n"
        "/new - start a new session\n"
        "/model - show current provider settings\n"
        "/model <model_id> - switch the current provider model\n"
        "/models - choose a provider and then a model\n"
        "/providers - show provider summary and switching buttons\n"
        "/provider_add - create a provider\n"
        "/provider_edit <provider_id> - edit a provider\n"
        "/provider_delete <provider_id> - delete a provider\n"
        "/provider_cancel - cancel provider setup\n"
        "/skip - keep the current value during provider edit\n"
        "/reset - clear your conversation history"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await asyncio.to_thread(chat_store.clear_user_history, update.effective_user.id)
    await update.message.reply_text("Your conversation history has been cleared.")


async def new_session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await asyncio.to_thread(chat_store.clear_user_history, update.effective_user.id)
    await update.message.reply_text("Started a new session. Previous context was cleared.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    requested_model = " ".join(context.args).strip()
    if not requested_model:
        await update.message.reply_text(
            build_model_settings_text(),
            parse_mode=ParseMode.HTML,
            reply_markup=build_model_settings_keyboard(),
        )
        return

    provider = get_current_provider()
    try:
        available_ids = await fetch_available_model_ids(provider)
    except Exception:
        logging.exception("Failed to fetch model list for /model")
        await update.message.reply_text(
            "Failed to validate model ID from the current provider. Check provider config and try again."
        )
        return

    available_set = set(available_ids)
    if requested_model in available_set:
        if requested_model == provider.current_model:
            await update.message.reply_text(
                f"Provider {provider.name} is already using {requested_model}."
            )
            return
        try:
            updated_provider = runtime_config_store.set_provider_current_model(
                provider.id,
                requested_model,
            )
        except Exception:
            logging.exception("Failed to persist runtime config for /model")
            await update.message.reply_text(
                f"Failed to save the selected model to {CONFIG_PATH}. Please try again."
            )
            return
        await update.message.reply_text(
            f"Provider {updated_provider.name} model set to {requested_model}.\n"
            f"Saved to {CONFIG_PATH}."
        )
        return

    prefix_matches = sorted(model_id for model_id in available_set if model_id.startswith(requested_model))
    if prefix_matches:
        hint = "\n".join(f"- {model_id}" for model_id in prefix_matches[:10])
        extra = "\n..." if len(prefix_matches) > 10 else ""
        await update.message.reply_text(
            "Model ID seems incomplete. Please provide a full model ID.\n"
            f"Matches:\n{hint}{extra}"
        )
        return

    await update.message.reply_text(
        "Invalid model ID for the current provider. Use /models to open the provider and model menu."
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    clear_models_menu_cache(context)
    await render_provider_picker(update)


async def providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await render_provider_summary(update)


async def provider_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await start_provider_wizard(update, context, mode="add")


async def provider_edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    provider_id = " ".join(context.args).strip()
    if not provider_id:
        await update.message.reply_text("Usage: /provider_edit <provider_id>")
        return
    try:
        provider = runtime_config_store.get_provider(provider_id)
    except KeyError:
        await update.message.reply_text(
            f"Unknown provider id: {provider_id}. Use /providers to list valid ids."
        )
        return
    await start_provider_wizard(update, context, mode="edit", provider=provider)


async def provider_delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    provider_id = " ".join(context.args).strip()
    if not provider_id:
        await update.message.reply_text("Usage: /provider_delete <provider_id>")
        return
    try:
        provider = runtime_config_store.get_provider(provider_id)
    except KeyError:
        await update.message.reply_text(
            f"Unknown provider id: {provider_id}. Use /providers to list valid ids."
        )
        return
    if len(get_runtime_config().providers) <= 1:
        await update.message.reply_text("Cannot delete the last provider.")
        return

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Confirm delete",
                    callback_data=f"providers:delete_confirm:{provider.id}",
                ),
                InlineKeyboardButton(
                    "Cancel",
                    callback_data=f"providers:delete_cancel:{provider.id}",
                ),
            ]
        ]
    )
    await update.message.reply_text(
        f"Delete provider {provider.name} ({provider.id})?\n"
        "This cannot be undone from the bot.",
        reply_markup=keyboard,
    )


async def provider_cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    if get_provider_wizard(context) is None:
        await update.message.reply_text("No provider setup is in progress.")
        return
    clear_provider_wizard(context)
    await update.message.reply_text("Provider setup cancelled.")


async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    wizard = get_provider_wizard(context)
    if wizard is None:
        await update.message.reply_text("No provider setup is in progress.")
        return
    if wizard["mode"] != "edit":
        await update.message.reply_text("/skip is only available while editing a provider.")
        return

    step = str(wizard["step"])
    if step == "name":
        wizard["step"] = "base_url"
        await prompt_provider_wizard_step(update, context)
        return
    if step == "base_url":
        wizard["step"] = "api_key"
        await prompt_provider_wizard_step(update, context)
        return
    if step == "api_key":
        wizard["step"] = "default_model"
        await prompt_provider_wizard_step(update, context)
        return
    if step == "default_model":
        wizard["step"] = "confirm"
        await prompt_provider_wizard_step(update, context)
        return

    await update.message.reply_text("/skip is not valid at this step.")


async def provider_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await query.answer("Access denied for this bot.", show_alert=True)
        return

    data = query.data or ""
    parts = data.split(":")
    scope = parts[0] if parts else ""
    action = parts[1] if len(parts) > 1 else ""

    try:
        if scope == "providers":
            if action == "summary":
                await query.answer()
                await render_provider_summary(update)
                return

            if action == "switch":
                provider_id = parts[2] if len(parts) > 2 else ""
                provider = runtime_config_store.set_current_provider(provider_id)
                await query.answer(f"Current provider: {provider.name}")
                await render_provider_summary(update)
                return

            if action == "models_menu":
                await query.answer()
                await render_provider_picker(update)
                return

            if action == "models":
                provider_id = parts[2] if len(parts) > 2 else ""
                provider = runtime_config_store.get_provider(provider_id)
                ids = await fetch_available_model_ids(provider)
                if not ids:
                    await query.answer("No models returned by this provider.", show_alert=True)
                    return
                provider = runtime_config_store.set_current_provider(provider_id)
                set_models_menu_cache(context, provider.id, ids)
                await query.answer(f"Current provider: {provider.name}")
                await edit_callback_text(
                    update,
                    build_models_menu_text(provider, ids, 0),
                    build_models_keyboard(provider.id, ids, provider.current_model, 0),
                )
                return

            if action == "delete_confirm":
                provider_id = parts[2] if len(parts) > 2 else ""
                runtime_config_store.delete_provider(provider_id)
                clear_models_menu_cache(context)
                await query.answer("Provider deleted.")
                await edit_callback_text(
                    update,
                    build_provider_summary_text(),
                    build_provider_summary_keyboard(),
                )
                return

            if action == "delete_cancel":
                provider_id = parts[2] if len(parts) > 2 else ""
                await query.answer("Delete cancelled.")
                await edit_callback_text(
                    update,
                    f"Deletion cancelled for <code>{html.escape(provider_id)}</code>.",
                    None,
                )
                return

        if scope == "models":
            if action == "backproviders":
                await query.answer()
                await render_provider_picker(update)
                return

            if action == "page":
                provider_id = parts[2] if len(parts) > 2 else ""
                page = int(parts[3]) if len(parts) > 3 else 0
                cache = get_models_menu_cache(context)
                if cache is None or cache["provider_id"] != provider_id:
                    provider = runtime_config_store.get_provider(provider_id)
                    ids = await fetch_available_model_ids(provider)
                    set_models_menu_cache(context, provider.id, ids)
                else:
                    provider = runtime_config_store.get_provider(provider_id)
                    ids = cache["ids"]  # type: ignore[assignment]
                await query.answer()
                await edit_callback_text(
                    update,
                    build_models_menu_text(provider, ids, page),
                    build_models_keyboard(provider.id, ids, provider.current_model, page),
                )
                return

            if action == "set":
                provider_id = parts[2] if len(parts) > 2 else ""
                index = int(parts[3]) if len(parts) > 3 else -1
                cache = get_models_menu_cache(context)
                if cache is None or cache["provider_id"] != provider_id:
                    provider = runtime_config_store.get_provider(provider_id)
                    ids = await fetch_available_model_ids(provider)
                    set_models_menu_cache(context, provider.id, ids)
                else:
                    provider = runtime_config_store.get_provider(provider_id)
                    ids = cache["ids"]  # type: ignore[assignment]

                if not 0 <= index < len(ids):
                    await query.answer("Model list expired. Please reopen /models.", show_alert=True)
                    return

                model_id = ids[index]
                current_model = provider.current_model
                if model_id != current_model:
                    provider = runtime_config_store.set_provider_current_model(provider.id, model_id)
                page = index // MODELS_MENU_PAGE_SIZE
                await query.answer(
                    "Already using this model." if model_id == current_model else "Provider model updated."
                )
                await edit_callback_text(
                    update,
                    build_models_menu_text(provider, ids, page),
                    build_models_keyboard(provider.id, ids, provider.current_model, page),
                )
                if model_id != current_model and query.message:
                    await query.message.reply_text(
                        f"Provider {provider.name} model set to {model_id}.\nSaved to {CONFIG_PATH}."
                    )
                return

        await query.answer()
    except ValueError as exc:
        logging.exception("Provider callback rejected")
        await query.answer(str(exc), show_alert=True)
    except KeyError:
        logging.exception("Provider callback could not find provider")
        await query.answer("Provider not found. Use /providers to refresh the list.", show_alert=True)
    except Exception:
        logging.exception("Failed to handle provider/model callback")
        try:
            await query.answer("Failed to update the menu. Please try again.", show_alert=True)
        except BadRequest:
            pass


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await update.message.reply_text(
        "Commands:\n"
        "/new - start a new session\n"
        "/model - show current provider settings\n"
        "/model <model_id> - switch the current provider model\n"
        "/models - choose a provider and then a model\n"
        "/providers - show provider summary and switch buttons\n"
        "/provider_add - create a provider\n"
        "/provider_edit <provider_id> - edit a provider\n"
        "/provider_delete <provider_id> - delete a provider\n"
        "/provider_cancel - cancel provider setup\n"
        "/skip - keep the current field while editing a provider\n"
        "/reset - clear your conversation history\n\n"
        "Set TELEGRAM_BOT_TOKEN and bootstrap OpenAI values in .env.\n"
        f"Runtime provider config is persisted in {CONFIG_PATH}."
    )


async def stream_llm_answer(
    user_id: int,
    user_content: str | list,
    out_message: Message,
) -> str:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history_messages = await asyncio.to_thread(
        chat_store.get_recent_messages,
        user_id,
        MAX_HISTORY_MESSAGES,
    )
    messages.extend(
        {
            "role": item["role"],
            "content": (
                strip_think_tags(item["content"])
                if item["role"] == "assistant"
                else item["content"]
            ),
        }
        for item in history_messages
    )
    messages.append({"role": "user", "content": user_content})

    provider = get_current_provider()
    client = build_openai_client(provider)
    active_model = provider.current_model
    raw_text = ""
    visible_text = ""
    last_sent_text = ""
    last_edit_at = 0.0
    think_filter = ThinkTagFilter()

    stream = await client.chat.completions.create(
        model=active_model,
        messages=messages,
        stream=True,
        timeout=120,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = delta.content or ""
        if not token:
            continue

        raw_text += token
        visible_text += think_filter.feed(token)
        now = time.monotonic()
        changed_chars = len(visible_text) - len(last_sent_text)
        should_edit = (
            changed_chars >= STREAM_MIN_CHARS_DELTA
            and now - last_edit_at >= STREAM_EDIT_INTERVAL_SECONDS
        )

        if should_edit:
            preview = truncate_for_telegram(visible_text)
            if preview != last_sent_text:
                try:
                    await out_message.edit_text(preview)
                    last_sent_text = preview
                except RetryAfter as exc:
                    await asyncio.sleep(exc.retry_after)
                except BadRequest:
                    pass
                last_edit_at = now

    visible_text += think_filter.finish()
    answer = strip_think_tags(raw_text).strip()
    if not answer:
        answer = visible_text.strip()
    answer = answer or "I could not generate a response. Please try again."
    await finalize_reply(out_message, answer)
    return answer


async def _respond(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_content: str | list,
    history_text: str,
) -> None:
    user_id = update.effective_user.id  # type: ignore[union-attr]
    if not update.effective_chat or not update.message:
        return
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
        out_message = await update.message.reply_text("Thinking...")
        answer = await stream_llm_answer(user_id, user_content, out_message)
        await asyncio.to_thread(chat_store.append_message, user_id, "user", history_text)
        await asyncio.to_thread(chat_store.append_message, user_id, "assistant", answer)
    except Exception:
        logging.exception("Failed to process message")
        try:
            await update.message.reply_text(
                "Request failed. Check your provider/model/config and try again."
            )
        except Exception:
            logging.exception("Failed to send error reply")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not update.effective_user:
        return
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Access denied for this bot.")
        return

    if get_provider_wizard(context) is not None:
        await handle_provider_wizard_text(update, context)
        return

    allowed, retry_after = rate_limiter.allow(user_id)
    if not allowed:
        await update.message.reply_text(
            f"Rate limit exceeded. Try again in about {retry_after} seconds."
        )
        return
    user_text = update.message.text.strip()
    if not user_text:
        return
    await _respond(update, context, user_text, user_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo or not update.effective_user:
        return
    user_id = update.effective_user.id
    if not authorized(user_id):
        await update.message.reply_text("Access denied for this bot.")
        return
    if get_provider_wizard(context) is not None:
        await update.message.reply_text(
            "Provider setup is waiting for text input. Use /provider_cancel to stop it first."
        )
        return

    allowed, retry_after = rate_limiter.allow(user_id)
    if not allowed:
        await update.message.reply_text(
            f"Rate limit exceeded. Try again in about {retry_after} seconds."
        )
        return

    caption = (update.message.caption or "").strip()
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()
    b64 = base64.b64encode(data).decode()

    user_content: list[dict] = []
    if caption:
        user_content.append({"type": "text", "text": caption})
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }
    )

    await _respond(update, context, user_content, caption or "[image]")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    if WHITELIST_USER_IDS:
        logging.info("Whitelist enabled for %d users", len(WHITELIST_USER_IDS))
    logging.info("SQLite chat storage path: %s", os.path.abspath(SQLITE_PATH))
    logging.info("Runtime config path: %s", os.path.abspath(CONFIG_PATH))

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler(["new", "newchat"], new_session_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("providers", providers_command))
    app.add_handler(CommandHandler("provider_add", provider_add_command))
    app.add_handler(CommandHandler("provider_edit", provider_edit_command))
    app.add_handler(CommandHandler("provider_delete", provider_delete_command))
    app.add_handler(CommandHandler("provider_cancel", provider_cancel_command))
    app.add_handler(CommandHandler("skip", skip_command))
    app.add_handler(CallbackQueryHandler(provider_callback_router, pattern=r"^(providers|models):"))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        chat_store.close()


if __name__ == "__main__":
    main()
