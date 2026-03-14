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
from urllib.parse import urlparse

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


def normalize_optional_config_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def normalize_config_text(value: object, default: str) -> str:
    normalized = normalize_optional_config_text(value)
    return normalized if normalized is not None else default


@dataclass(frozen=True)
class LLMRuntimeConfig:
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
            "openai_base_url": self.openai_base_url,
        }


class RuntimeConfigStore:
    def __init__(self, config_path: str, default_config: LLMRuntimeConfig) -> None:
        self.path = config_path
        self._default_config = default_config
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._config = self._load_initial_config()

    def _merge_payload(self, payload: dict[str, object] | None) -> LLMRuntimeConfig:
        payload = payload or {}
        if "openai_base_url" in payload:
            openai_base_url = normalize_optional_config_text(payload.get("openai_base_url"))
        else:
            openai_base_url = self._default_config.openai_base_url
        return LLMRuntimeConfig(
            openai_api_key=normalize_config_text(
                payload.get("openai_api_key"),
                self._default_config.openai_api_key,
            ),
            openai_model=normalize_config_text(
                payload.get("openai_model"),
                self._default_config.openai_model,
            ),
            openai_base_url=openai_base_url,
        )

    def _load_initial_config(self) -> LLMRuntimeConfig:
        payload: dict[str, object] | None = None
        should_create = False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw_payload = json.load(handle)
            if isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                logging.warning(
                    "Runtime config %s must contain a JSON object; falling back to .env defaults.",
                    self.path,
                )
        except FileNotFoundError:
            should_create = True
        except json.JSONDecodeError:
            logging.exception(
                "Failed to parse runtime config %s; falling back to .env defaults.",
                self.path,
            )
        except OSError:
            logging.exception(
                "Failed to read runtime config %s; falling back to .env defaults.",
                self.path,
            )

        config = self._merge_payload(payload)
        if should_create:
            with self._lock:
                self._persist_locked(config)
        return config

    def _persist_locked(self, config: LLMRuntimeConfig) -> None:
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

    def current(self) -> LLMRuntimeConfig:
        with self._lock:
            return self._config

    def update_model(self, model_id: str) -> LLMRuntimeConfig:
        normalized_model = normalize_optional_config_text(model_id)
        if normalized_model is None:
            raise ValueError("Model ID cannot be empty.")

        with self._lock:
            if normalized_model == self._config.openai_model:
                return self._config
            updated_config = replace(self._config, openai_model=normalized_model)
            self._persist_locked(updated_config)
            self._config = updated_config
            return updated_config


TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
CONFIG_PATH = "./data/config.json"
DEFAULT_LLM_CONFIG = LLMRuntimeConfig(
    openai_api_key=normalize_config_text(os.getenv("OPENAI_API_KEY"), ""),
    openai_model=normalize_config_text(os.getenv("OPENAI_MODEL"), "gpt-4.1-mini"),
    openai_base_url=normalize_optional_config_text(os.getenv("OPENAI_BASE_URL")),
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

runtime_config_store = RuntimeConfigStore(CONFIG_PATH, DEFAULT_LLM_CONFIG)
initial_runtime_config = runtime_config_store.current()
if not initial_runtime_config.openai_api_key:
    raise RuntimeError(
        f"Missing LLM API key. Set OPENAI_API_KEY in .env or openai_api_key in {CONFIG_PATH}."
    )

client_kwargs = {"api_key": initial_runtime_config.openai_api_key}
if initial_runtime_config.openai_base_url:
    client_kwargs["base_url"] = initial_runtime_config.openai_base_url
llm_client = AsyncOpenAI(**client_kwargs)
chat_store = SQLiteChatStore(SQLITE_PATH)
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_SECONDS)


def authorized(user_id: int) -> bool:
    return not WHITELIST_USER_IDS or user_id in WHITELIST_USER_IDS


def get_runtime_config() -> LLMRuntimeConfig:
    return runtime_config_store.current()


def get_active_model() -> str:
    return get_runtime_config().openai_model


def format_base_url(base_url: str | None) -> str:
    return base_url or "https://api.openai.com/v1 (default)"


def mask_secret(value: str, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return "(missing)"
    if len(value) <= prefix + suffix:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


def get_models_source_label() -> str:
    runtime_base_url = get_runtime_config().openai_base_url
    if not runtime_base_url:
        return "api.openai.com/v1"
    parsed = urlparse(runtime_base_url)
    host = parsed.netloc or runtime_base_url
    path = parsed.path.rstrip("/")
    if path and path != "/":
        return f"{host}{path}"
    return host


async def fetch_available_model_ids() -> list[str]:
    models = await llm_client.models.list()
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


def build_model_settings_text() -> str:
    runtime_config = get_runtime_config()
    current_model = html.escape(runtime_config.openai_model)
    default_model = html.escape(DEFAULT_LLM_CONFIG.openai_model)
    base_url = html.escape(format_base_url(runtime_config.openai_base_url))
    config_path = html.escape(CONFIG_PATH)
    return (
        "<b>Current global model settings</b>\n"
        f"current_model: <code>{current_model}</code>\n"
        f"default_model_from_env: <code>{default_model}</code>\n"
        f"base_url: <code>{base_url}</code>\n"
        f"config_path: <code>{config_path}</code>\n"
        f"max_history_pairs: <code>{MAX_HISTORY_PAIRS}</code>\n"
        f"rate_limit: <code>{RATE_LIMIT_COUNT} requests / {RATE_LIMIT_WINDOW_SECONDS}s</code>\n"
        "streaming: <code>enabled</code>\n\n"
        "Use <code>/model &lt;model_id&gt;</code> to switch the global model, or tap the button below."
    )


def build_model_settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Open model list", callback_data="models:open:0")]
    ])


def build_models_menu_text(ids: list[str], page: int) -> str:
    runtime_config = get_runtime_config()
    current_model = html.escape(runtime_config.openai_model)
    source_label = html.escape(get_models_source_label())
    key_label = html.escape(mask_secret(runtime_config.openai_api_key))
    page_count = max(1, (len(ids) + MODELS_MENU_PAGE_SIZE - 1) // MODELS_MENU_PAGE_SIZE)
    page = max(0, min(page, page_count - 1))
    page_label = f"\nPage {page + 1}/{page_count}" if page_count > 1 else ""
    return (
        f"<b>Models</b> ({source_label} | key {key_label}) - {len(ids)} available\n"
        f"Current global model: <code>{current_model}</code>{page_label}\n"
        "Tap a model below to switch and save it to config.json."
    )


def build_models_keyboard(ids: list[str], current_model: str, page: int) -> InlineKeyboardMarkup:
    page_count = max(1, (len(ids) + MODELS_MENU_PAGE_SIZE - 1) // MODELS_MENU_PAGE_SIZE)
    page = max(0, min(page, page_count - 1))
    start = page * MODELS_MENU_PAGE_SIZE
    end = min(start + MODELS_MENU_PAGE_SIZE, len(ids))

    rows: list[list[InlineKeyboardButton]] = []
    for index in range(start, end):
        model_id = ids[index]
        label = f"{model_id} [current]" if model_id == current_model else model_id
        rows.append([InlineKeyboardButton(label, callback_data=f"models:set:{index}")])

    navigation: list[InlineKeyboardButton] = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton("< Prev", callback_data=f"models:page:{page - 1}")
        )
    if page + 1 < page_count:
        navigation.append(
            InlineKeyboardButton("Next >", callback_data=f"models:page:{page + 1}")
        )
    if navigation:
        rows.append(navigation)

    rows.append([InlineKeyboardButton("<< Back", callback_data="models:back")])
    return InlineKeyboardMarkup(rows)


def get_cached_model_ids(context: ContextTypes.DEFAULT_TYPE) -> list[str] | None:
    cached_ids = context.user_data.get("models_menu_ids")
    if isinstance(cached_ids, list) and all(isinstance(item, str) for item in cached_ids):
        return cached_ids
    return None


async def edit_callback_text(
    update: Update,
    text: str,
    reply_markup: InlineKeyboardMarkup,
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
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after)
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
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
            try:
                await message.edit_text(content, parse_mode=ParseMode.HTML)
                return
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

        try:
            await message.edit_text(content)
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
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
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
            try:
                await message.reply_text(content, parse_mode=ParseMode.HTML)
                return
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

        try:
            await message.reply_text(content)
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
            try:
                await message.reply_text(content)
            except (BadRequest, RetryAfter):
                pass
        except BadRequest:
            pass

    await _edit_with_markup_fallback(chunks[0])
    for chunk in chunks[1:]:
        await _reply_with_markup_fallback(chunk)


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
        "/model - show current global model settings\n"
        "/model <model_id> - switch the global model\n"
        "/models - open the model button menu\n"
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

    try:
        available_ids = await fetch_available_model_ids()
    except Exception:
        logging.exception("Failed to fetch model list for /model")
        await update.message.reply_text(
            "Failed to validate model ID from API. Check API key/base URL and try again."
        )
        return

    available_set = set(available_ids)
    if requested_model in available_set:
        current_model = get_active_model()
        if requested_model == current_model:
            await update.message.reply_text(
                f"Global model is already set to {requested_model}."
            )
            return
        try:
            runtime_config_store.update_model(requested_model)
        except Exception:
            logging.exception("Failed to persist runtime config for /model")
            await update.message.reply_text(
                f"Failed to save the selected model to {CONFIG_PATH}. Please try again."
            )
            return
        await update.message.reply_text(
            f"Global model set to {requested_model}.\n"
            f"Saved to {CONFIG_PATH}.\n"
            f"(default model in .env is: {DEFAULT_LLM_CONFIG.openai_model})"
        )
        return

    prefix_matches = sorted([m for m in available_set if m.startswith(requested_model)])
    if prefix_matches:
        hint = "\n".join(f"- {m}" for m in prefix_matches[:10])
        extra = "\n..." if len(prefix_matches) > 10 else ""
        await update.message.reply_text(
            "Model ID seems incomplete. Please provide a full model ID.\n"
            f"Matches:\n{hint}{extra}"
        )
        return

    await update.message.reply_text(
        "Invalid model ID. Use /models to open the model menu, or run /model <model_id> with a valid full ID."
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    try:
        ids = await fetch_available_model_ids()
        if not ids:
            await update.message.reply_text("No models returned by API.")
            return

        context.user_data["models_menu_ids"] = ids
        await update.message.reply_text(
            build_models_menu_text(ids, 0),
            parse_mode=ParseMode.HTML,
            reply_markup=build_models_keyboard(ids, get_active_model(), 0),
        )
    except Exception:
        logging.exception("Failed to list models")
        await update.message.reply_text(
            "Failed to fetch model list. Check API key/base URL and try again."
        )


async def models_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await query.answer("Access denied for this bot.", show_alert=True)
        return

    data = query.data or ""
    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""

    try:
        if action == "back":
            await query.answer()
            await edit_callback_text(
                update,
                build_model_settings_text(),
                build_model_settings_keyboard(),
            )
            return

        if action == "open":
            page = int(parts[2]) if len(parts) > 2 else 0
            ids = await fetch_available_model_ids()
            if not ids:
                await query.answer("No models returned by API.", show_alert=True)
                return
            context.user_data["models_menu_ids"] = ids
            await query.answer()
            await edit_callback_text(
                update,
                build_models_menu_text(ids, page),
                build_models_keyboard(ids, get_active_model(), page),
            )
            return

        if action == "page":
            page = int(parts[2]) if len(parts) > 2 else 0
            ids = get_cached_model_ids(context)
            if ids is None:
                ids = await fetch_available_model_ids()
                context.user_data["models_menu_ids"] = ids
            if not ids:
                await query.answer("No models returned by API.", show_alert=True)
                return
            await query.answer()
            await edit_callback_text(
                update,
                build_models_menu_text(ids, page),
                build_models_keyboard(ids, get_active_model(), page),
            )
            return

        if action == "set":
            ids = get_cached_model_ids(context)
            if ids is None:
                ids = await fetch_available_model_ids()
                context.user_data["models_menu_ids"] = ids
            if not ids:
                await query.answer("No models returned by API.", show_alert=True)
                return

            index = int(parts[2]) if len(parts) > 2 else -1
            if not 0 <= index < len(ids):
                await query.answer("Model list expired. Please reopen /models.", show_alert=True)
                return

            model_id = ids[index]
            current_model = get_active_model()
            page = index // MODELS_MENU_PAGE_SIZE

            if model_id != current_model:
                runtime_config_store.update_model(model_id)

            await edit_callback_text(
                update,
                build_models_menu_text(ids, page),
                build_models_keyboard(ids, model_id, page),
            )

            if model_id == current_model:
                await query.answer("Already using this global model.")
            else:
                await query.answer("Global model updated.")
                if query.message:
                    await query.message.reply_text(
                        f"Global model set to {model_id}.\nSaved to {CONFIG_PATH}."
                    )
            return

        await query.answer()
    except Exception:
        logging.exception("Failed to handle models menu callback")
        try:
            await query.answer(
                "Failed to update model menu. Please try /models again.",
                show_alert=True,
            )
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
        "/model - show current global model settings\n"
        "/model <model_id> - switch the global model\n"
        "/models - open the model button menu\n"
        "/reset - clear your conversation history\n\n"
        "Set TELEGRAM_BOT_TOKEN and the default OpenAI values in .env.\n"
        f"Runtime LLM config is persisted in {CONFIG_PATH}.\n"
        "Optional envs: OPENAI_MODEL, OPENAI_BASE_URL, MAX_HISTORY_PAIRS, "
        "SYSTEM_PROMPT, SQLITE_PATH, WHITELIST_USER_IDS, RATE_LIMIT_COUNT, "
        "RATE_LIMIT_WINDOW_SECONDS, MODELS_MENU_PAGE_SIZE"
    )


async def stream_llm_answer(
    user_id: int, user_content: str | list, out_message: Message,
) -> str:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(await asyncio.to_thread(chat_store.get_recent_messages, user_id, MAX_HISTORY_MESSAGES))
    messages.append({"role": "user", "content": user_content})

    full_text = ""
    last_sent_text = ""
    last_edit_at = 0.0
    active_model = get_active_model()

    stream = await llm_client.chat.completions.create(
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

        full_text += token
        now = time.monotonic()
        changed_chars = len(full_text) - len(last_sent_text)
        should_edit = (
            changed_chars >= STREAM_MIN_CHARS_DELTA
            and now - last_edit_at >= STREAM_EDIT_INTERVAL_SECONDS
        )

        if should_edit:
            preview = truncate_for_telegram(full_text)
            if preview != last_sent_text:
                try:
                    await out_message.edit_text(preview)
                    last_sent_text = preview
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after)
                except BadRequest:
                    pass
                last_edit_at = now

    answer = full_text.strip() or "I could not generate a response. Please try again."
    await finalize_reply(out_message, answer)
    return answer


async def _respond(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_content: str | list,
    history_text: str,
) -> None:
    """Shared logic: send typing indicator, stream LLM reply, persist history."""
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
                "Request failed. Check your API key/model/config and try again."
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
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    })

    await _respond(update, context, user_content, caption or "[image]")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    if WHITELIST_USER_IDS:
        logging.info("Whitelist enabled for %d users", len(WHITELIST_USER_IDS))
    logging.info("SQLite chat storage path: %s", os.path.abspath(SQLITE_PATH))

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler(["new", "newchat"], new_session_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CallbackQueryHandler(models_menu_callback, pattern=r"^models:"))
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
