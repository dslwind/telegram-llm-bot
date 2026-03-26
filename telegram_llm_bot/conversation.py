import asyncio
import logging
import time
from urllib.parse import urlparse

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
from .storage import ProviderConfig
from .ui import finalize_reply
from .utils import ThinkTagFilter, strip_think_tags, truncate_for_telegram


def _is_official_openai_base_url(base_url: str | None) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    return parsed.netloc == "api.openai.com"


def _chat_content_to_responses_content(user_content: str | list) -> str | list[dict[str, str]]:
    if isinstance(user_content, str):
        return user_content

    content: list[dict[str, str]] = []
    for item in user_content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                content.append({"type": "input_text", "text": text})
            continue

        if item_type == "image_url":
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        "detail": "auto",
                    }
                )

    return content or ""


def _build_responses_input(
    history_messages: list[dict[str, str]],
    user_content: str | list,
) -> list[dict[str, object]]:
    input_items: list[dict[str, object]] = []
    for item in history_messages:
        role = item["role"]
        content = (
            strip_think_tags(item["content"])
            if role == "assistant"
            else item["content"]
        )
        input_items.append(
            {
                "type": "message",
                "role": role,
                "content": content,
            }
        )

    input_items.append(
        {
            "type": "message",
            "role": "user",
            "content": _chat_content_to_responses_content(user_content),
        }
    )
    return input_items


async def _stream_response_events(
    out_message: Message,
    response_stream,
) -> tuple[str, str]:
    raw_text = ""
    visible_text = ""
    last_sent_text = ""
    last_edit_at = 0.0
    think_filter = ThinkTagFilter()

    async for event in response_stream:
        if event.type != "response.output_text.delta":
            continue
        token = event.delta or ""
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
    return raw_text, visible_text


async def _stream_llm_answer_via_responses(
    provider: ProviderConfig,
    history_messages: list[dict[str, str]],
    user_content: str | list,
    out_message: Message,
) -> str:
    client = build_openai_client(provider)

    async with client.responses.stream(
        model=provider.current_model,
        instructions=SYSTEM_PROMPT,
        input=_build_responses_input(history_messages, user_content),
        timeout=120,
    ) as response_stream:
        raw_text, visible_text = await _stream_response_events(out_message, response_stream)
        final_response = await response_stream.get_final_response()

    answer = strip_think_tags(final_response.output_text or raw_text).strip()
    if not answer:
        answer = visible_text.strip()
    answer = answer or "I could not generate a response. Please try again."
    await finalize_reply(out_message, answer)
    return answer


async def _stream_llm_answer_via_chat_completions(
    provider: ProviderConfig,
    history_messages: list[dict[str, str]],
    user_content: str | list,
    out_message: Message,
) -> str:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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

    client = build_openai_client(provider)

    raw_text = ""
    visible_text = ""
    last_sent_text = ""
    last_edit_at = 0.0
    think_filter = ThinkTagFilter()

    stream = await client.chat.completions.create(
        model=provider.current_model,
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


async def stream_llm_answer(
    user_id: int,
    user_content: str | list,
    out_message: Message,
) -> str:
    history_messages = await asyncio.to_thread(
        chat_store.get_recent_messages,
        user_id,
        MAX_HISTORY_MESSAGES,
    )
    provider = get_current_provider()

    if _is_official_openai_base_url(provider.base_url):
        return await _stream_llm_answer_via_responses(
            provider,
            history_messages,
            user_content,
            out_message,
        )

    try:
        return await _stream_llm_answer_via_responses(
            provider,
            history_messages,
            user_content,
            out_message,
        )
    except Exception:
        logging.exception(
            "Responses API failed for provider %s; falling back to chat completions",
            provider.id,
        )
        return await _stream_llm_answer_via_chat_completions(
            provider,
            history_messages,
            user_content,
            out_message,
        )


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
