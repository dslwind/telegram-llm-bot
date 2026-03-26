import asyncio
import html
import logging

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, ContextTypes

from .constants import CONFIG_PATH, TELEGRAM_TEXT_LIMIT
from .runtime import (
    BOOTSTRAP_PROVIDER,
    MAX_HISTORY_PAIRS,
    MODELS_MENU_PAGE_SIZE,
    RATE_LIMIT_COUNT,
    RATE_LIMIT_WINDOW_SECONDS,
    get_current_provider,
    get_runtime_config,
)
from .storage import ProviderConfig
from .utils import (
    extract_think_sections,
    format_base_url,
    markdown_to_telegram_html,
    mask_secret,
    split_text_for_telegram,
)


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


def build_models_keyboard(
    provider_id: str,
    ids: list[str],
    current_model: str,
    page: int,
) -> InlineKeyboardMarkup:
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


def get_bot_commands() -> list[BotCommand]:
    return [
        BotCommand("start", "Show bot status and command overview"),
        BotCommand("new", "Start a new chat session"),
        BotCommand("model", "Show or switch the current provider model"),
        BotCommand("models", "Choose a provider and then a model"),
        BotCommand("providers", "Show and switch configured providers"),
        BotCommand("provider_add", "Create a new provider"),
        BotCommand("provider_edit", "Edit a provider by id"),
        BotCommand("provider_delete", "Delete a provider by id"),
        BotCommand("provider_cancel", "Cancel provider setup"),
        BotCommand("reset", "Clear your conversation history"),
        BotCommand("help", "Show command help"),
    ]


async def sync_bot_commands(application: Application) -> None:
    try:
        await application.bot.set_my_commands(get_bot_commands())
        logging.info("Telegram bot commands synced successfully")
    except Exception:
        logging.exception("Failed to sync Telegram bot commands on startup")


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


def _build_reasoning_chunk(text: str) -> str:
    escaped = html.escape(text)
    return f"<blockquote expandable><b>Reasoning</b>\n{escaped}</blockquote>"


def build_reply_html_chunks(text: str, raw_text: str | None = None) -> list[str]:
    answer_text = text
    reasoning_text = ""

    if raw_text:
        reasoning_text, extracted_answer = extract_think_sections(raw_text)
        if extracted_answer:
            answer_text = extracted_answer

    answer_chunks = [markdown_to_telegram_html(chunk) for chunk in split_text_for_telegram(answer_text)]
    reasoning_chunks = (
        [_build_reasoning_chunk(chunk) for chunk in split_text_for_telegram(reasoning_text)]
        if reasoning_text
        else []
    )

    if not reasoning_chunks:
        return answer_chunks

    if not answer_chunks:
        return reasoning_chunks

    combined_first = f"{reasoning_chunks[0]}\n\n{answer_chunks[0]}"
    if len(combined_first) <= TELEGRAM_TEXT_LIMIT:
        return [combined_first, *answer_chunks[1:], *reasoning_chunks[1:]]

    return [answer_chunks[0], *answer_chunks[1:], *reasoning_chunks]


async def finalize_reply(message: Message, text: str, raw_text: str | None = None) -> None:
    chunks = build_reply_html_chunks(text, raw_text=raw_text)

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
