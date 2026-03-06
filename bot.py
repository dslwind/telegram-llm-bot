import logging
import os
from collections import defaultdict, deque
from typing import Deque, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
MAX_HISTORY_PAIRS = max(1, int(os.getenv("MAX_HISTORY_PAIRS", "10")))
MAX_HISTORY_MESSAGES = MAX_HISTORY_PAIRS * 2
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful Telegram assistant. Answer clearly and briefly.",
)

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL
llm_client = AsyncOpenAI(**client_kwargs)

chat_histories: Dict[int, Deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Hi, I am ready. Send me any message and I will reply with the LLM.\n"
        "Commands:\n"
        "/reset - clear your conversation history"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    chat_histories[update.effective_user.id].clear()
    await update.message.reply_text("Your conversation history has been cleared.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY in .env, then chat directly.\n"
        "Optional envs: OPENAI_MODEL, OPENAI_BASE_URL, MAX_HISTORY_PAIRS, SYSTEM_PROMPT"
    )


async def ask_llm(user_id: int, user_text: str) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_histories[user_id])
    messages.append({"role": "user", "content": user_text})

    response = await llm_client.responses.create(model=OPENAI_MODEL, input=messages)
    answer = (response.output_text or "").strip()
    if not answer:
        answer = "I could not generate a response. Please try again."

    chat_histories[user_id].append({"role": "user", "content": user_text})
    chat_histories[user_id].append({"role": "assistant", "content": answer})
    return answer


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not update.effective_user:
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
        answer = await ask_llm(update.effective_user.id, user_text)
        await update.message.reply_text(answer)
    except Exception:
        logging.exception("Failed to process message")
        await update.message.reply_text(
            "Request failed. Check your API key/model/config and try again."
        )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
