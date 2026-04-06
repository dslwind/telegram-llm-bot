import asyncio
import base64
import html
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from .constants import CONFIG_PATH
from .conversation import respond
from .provider_wizard import (
    handle_provider_wizard_text,
    prompt_provider_wizard_step,
    start_provider_wizard,
)
from .runtime import (
    MODELS_MENU_PAGE_SIZE,
    authorized,
    chat_store,
    fetch_available_model_ids,
    get_current_provider,
    get_provider_for_user,
    get_runtime_config,
    rate_limiter,
    runtime_config_store,
    session_request_gate,
)
from .session import build_chat_session_key
from .state import (
    clear_models_menu_cache,
    clear_provider_wizard,
    get_models_menu_cache,
    get_provider_wizard,
    get_user_provider_id,
    set_models_menu_cache,
    set_user_provider_id,
)
from .ui import (
    build_model_settings_keyboard,
    build_model_settings_text,
    build_models_keyboard,
    build_models_menu_text,
    build_provider_picker_keyboard,
    build_provider_picker_text,
    build_provider_summary_keyboard,
    build_provider_summary_text,
    edit_callback_text,
    try_delete_message,
)
from .utils import REASONING_EFFORT_VALUES, format_reasoning_effort, normalize_reasoning_effort


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


async def render_provider_summary_for_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_pid = get_user_provider_id(context)
    if update.message:
        await update.message.reply_text(
            build_provider_summary_text(user_provider_id=user_pid),
            parse_mode=ParseMode.HTML,
            reply_markup=build_provider_summary_keyboard(user_provider_id=user_pid),
        )
        return
    await edit_callback_text(
        update,
        build_provider_summary_text(user_provider_id=user_pid),
        build_provider_summary_keyboard(user_provider_id=user_pid),
    )


async def render_provider_picker(update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    user_pid = get_user_provider_id(context) if context else None
    if update.message:
        await update.message.reply_text(
            build_provider_picker_text(user_provider_id=user_pid),
            parse_mode=ParseMode.HTML,
            reply_markup=build_provider_picker_keyboard(user_provider_id=user_pid),
        )
        return
    await edit_callback_text(
        update,
        build_provider_picker_text(user_provider_id=user_pid),
        build_provider_picker_keyboard(user_provider_id=user_pid),
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
        "/reasoning - show current provider reasoning effort\n"
        "/reasoning <effort> - switch current provider reasoning effort\n"
        "/stop - stop the current response in this chat session\n"
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
    session = build_chat_session_key(update)
    if session is None:
        return
    await asyncio.to_thread(chat_store.clear_session_history, session)
    await update.message.reply_text("This chat session history has been cleared.")


async def new_session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    await asyncio.to_thread(chat_store.clear_session_history, session)
    clear_provider_wizard(context)
    await update.message.reply_text("Started a new session. Current chat context was cleared.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    requested_model = " ".join(context.args).strip()
    if not requested_model:
        user_pid = get_user_provider_id(context)
        await update.message.reply_text(
            build_model_settings_text(user_provider_id=user_pid),
            parse_mode=ParseMode.HTML,
            reply_markup=build_model_settings_keyboard(),
        )
        return

    provider = get_provider_for_user(get_user_provider_id(context))
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

    prefix_matches = sorted(
        model_id for model_id in available_set if model_id.startswith(requested_model)
    )
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


async def reasoning_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    provider = get_provider_for_user(get_user_provider_id(context))
    requested_effort = " ".join(context.args).strip()
    if not requested_effort:
        await update.message.reply_text(
            f"Provider {provider.name} reasoning effort: "
            f"<code>{html.escape(format_reasoning_effort(provider.reasoning_effort))}</code>\n"
            "Use <code>/reasoning default</code> to clear it or one of: "
            f"<code>{'</code>, <code>'.join(REASONING_EFFORT_VALUES)}</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        normalized_effort = normalize_reasoning_effort(requested_effort)
    except RuntimeError:
        await update.message.reply_text(
            "Invalid reasoning effort.\n"
            "Use <code>default</code> or one of: "
            f"<code>{'</code>, <code>'.join(REASONING_EFFORT_VALUES)}</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    if normalized_effort == provider.reasoning_effort:
        await update.message.reply_text(
            f"Provider {provider.name} is already using "
            f"{format_reasoning_effort(provider.reasoning_effort)} reasoning."
        )
        return

    try:
        updated_provider = runtime_config_store.set_provider_reasoning_effort(
            provider.id,
            normalized_effort,
        )
    except Exception:
        logging.exception("Failed to persist runtime config for /reasoning")
        await update.message.reply_text(
            f"Failed to save the selected reasoning effort to {CONFIG_PATH}. Please try again."
        )
        return

    await update.message.reply_text(
        f"Provider {updated_provider.name} reasoning effort set to "
        f"{format_reasoning_effort(updated_provider.reasoning_effort)}.\n"
        f"Saved to {CONFIG_PATH}."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return

    session = build_chat_session_key(update)
    if session is None:
        return

    cancel_result = session_request_gate.cancel(session)
    active_request = cancel_result.active_request
    if active_request is None:
        await update.message.reply_text("No active request is running in this chat session.")
        return

    if not cancel_result.cancelled:
        await update.message.reply_text("No cancellable request is running in this chat session.")
        return

    if active_request.chat_id is not None and active_request.message_id is not None:
        try:
            await context.bot.edit_message_text(
                "Stopped.",
                chat_id=active_request.chat_id,
                message_id=active_request.message_id,
            )
        except Exception:
            logging.info(
                "Failed to edit active response placeholder while stopping session %s",
                session,
            )

    await update.message.reply_text("Stopping the current request.")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    clear_models_menu_cache(context)
    await render_provider_picker(update, context)


async def providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    await render_provider_summary_for_user(update, context)


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
        wizard["step"] = "reasoning_effort"
        await prompt_provider_wizard_step(update, context)
        return
    if step == "reasoning_effort":
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
                await render_provider_summary_for_user(update, context)
                return

            if action == "switch":
                provider_id = parts[2] if len(parts) > 2 else ""
                provider = runtime_config_store.get_provider(provider_id)
                set_user_provider_id(context, provider_id)
                await query.answer(f"Current provider: {provider.name}")
                await render_provider_summary_for_user(update, context)
                return

            if action == "models_menu":
                await query.answer()
                await render_provider_picker(update, context)
                return

            if action == "models":
                provider_id = parts[2] if len(parts) > 2 else ""
                provider = runtime_config_store.get_provider(provider_id)
                ids = await fetch_available_model_ids(provider)
                if not ids:
                    await query.answer("No models returned by this provider.", show_alert=True)
                    return
                set_user_provider_id(context, provider_id)
                set_models_menu_cache(context, provider.id, ids)
                await query.answer(f"Current provider: {provider.name}")
                await edit_callback_text(
                    update,
                    build_models_menu_text(provider, ids, 0),
                    build_models_keyboard(provider.id, ids, provider.current_model, 0),
                )
                return

            if action == "edit":
                provider_id = ":".join(parts[2:]) if len(parts) > 2 else ""
                try:
                    provider = runtime_config_store.get_provider(provider_id)
                except KeyError:
                    await query.answer("Provider not found.", show_alert=True)
                    return
                await query.answer()
                await start_provider_wizard(update, context, mode="edit", provider=provider)
                return

            if action == "delete":
                provider_id = ":".join(parts[2:]) if len(parts) > 2 else ""
                try:
                    provider = runtime_config_store.get_provider(provider_id)
                except KeyError:
                    await query.answer("Provider not found.", show_alert=True)
                    return
                if len(get_runtime_config().providers) <= 1:
                    await query.answer("Cannot delete the last provider.", show_alert=True)
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
                await query.answer()
                await edit_callback_text(
                    update,
                    f"Delete provider <b>{html.escape(provider.name)}</b> "
                    f"(<code>{html.escape(provider.id)}</code>)?\n"
                    "This cannot be undone from the bot.",
                    keyboard,
                )
                return

            if action == "delete_confirm":
                provider_id = ":".join(parts[2:]) if len(parts) > 2 else ""
                runtime_config_store.delete_provider(provider_id)
                clear_models_menu_cache(context)
                await query.answer("Provider deleted.")
                user_pid = get_user_provider_id(context)
                await edit_callback_text(
                    update,
                    build_provider_summary_text(user_provider_id=user_pid),
                    build_provider_summary_keyboard(user_provider_id=user_pid),
                )
                return

            if action == "delete_cancel":
                provider_id = ":".join(parts[2:]) if len(parts) > 2 else ""
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
                await render_provider_picker(update, context)
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
                    provider = runtime_config_store.set_provider_current_model(
                        provider.id,
                        model_id,
                    )
                await query.answer(
                    "Already using this model."
                    if model_id == current_model
                    else "Provider model updated."
                )
                if query.message:
                    await try_delete_message(context, query.message.chat_id, query.message.message_id)
                    if model_id != current_model:
                        await query.message.reply_text(
                            f"Provider {provider.name} model set to <code>{html.escape(model_id)}</code>.\n"
                            f"Saved to {CONFIG_PATH}.",
                            parse_mode=ParseMode.HTML,
                        )
                return

        await query.answer()
    except ValueError as exc:
        logging.exception("Provider callback rejected")
        await query.answer(str(exc), show_alert=True)
    except KeyError:
        logging.exception("Provider callback could not find provider")
        await query.answer(
            "Provider not found. Use /providers to refresh the list.",
            show_alert=True,
        )
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
        "/reasoning - show current provider reasoning effort\n"
        "/reasoning <effort> - switch current provider reasoning effort\n"
        "/stop - stop the current response in this chat session\n"
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
    user_pid = get_user_provider_id(context)
    await respond(update, context, user_text, user_text, user_provider_id=user_pid)


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

    user_pid = get_user_provider_id(context)
    await respond(update, context, user_content, caption or "[image]", user_provider_id=user_pid)
