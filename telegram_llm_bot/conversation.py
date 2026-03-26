import asyncio
import logging
import time

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import ContextTypes

from .runtime import (
    MAX_HISTORY_MESSAGES,
    STREAM_EDIT_INTERVAL_SECONDS,
    STREAM_MIN_CHARS_DELTA,
    SYSTEM_PROMPT,
    build_openai_client,
    chat_store,
    get_current_provider,
)
from .ui import finalize_reply
from .utils import ThinkTagFilter, strip_think_tags, truncate_for_telegram


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


async def respond(
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
