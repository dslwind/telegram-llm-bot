import asyncio
import base64
import html
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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


TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
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

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL
llm_client = AsyncOpenAI(**client_kwargs)
chat_store = SQLiteChatStore(SQLITE_PATH)
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_SECONDS)


def authorized(user_id: int) -> bool:
    return not WHITELIST_USER_IDS or user_id in WHITELIST_USER_IDS


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
        "/model - show model information\n"
        "/models - list available models from API\n"
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
    base_url = OPENAI_BASE_URL or "https://api.openai.com/v1 (default)"
    await update.message.reply_text(
        "Current model settings:\n"
        f"- model: {OPENAI_MODEL}\n"
        f"- base_url: {base_url}\n"
        f"- max_history_pairs: {MAX_HISTORY_PAIRS}\n"
        f"- rate_limit: {RATE_LIMIT_COUNT} requests / {RATE_LIMIT_WINDOW_SECONDS}s\n"
        "- streaming: enabled"
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    try:
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

        ids = sorted(set(ids))
        if not ids:
            await update.message.reply_text("No models returned by API.")
            return

        text = "Available models:\n" + "\n".join(f"- {model_id}" for model_id in ids)
        for chunk in split_text_for_telegram(text):
            await update.message.reply_text(chunk)
    except Exception:
        logging.exception("Failed to list models")
        await update.message.reply_text(
            "Failed to fetch model list. Check API key/base URL and try again."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await update.message.reply_text(
        "Commands:\n"
        "/new - start a new session\n"
        "/model - show model information\n"
        "/models - list available models from API\n"
        "/reset - clear your conversation history\n\n"
        "Set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY in .env, then chat directly.\n"
        "Optional envs: OPENAI_MODEL, OPENAI_BASE_URL, MAX_HISTORY_PAIRS, "
        "SYSTEM_PROMPT, SQLITE_PATH, WHITELIST_USER_IDS, RATE_LIMIT_COUNT, "
        "RATE_LIMIT_WINDOW_SECONDS"
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

    stream = await llm_client.chat.completions.create(
        model=OPENAI_MODEL,
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
