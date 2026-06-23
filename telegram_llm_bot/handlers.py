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
from .session_manager import (
    build_delete_confirm_keyboard,
    build_sessions_keyboard,
    build_sessions_text,
)
from .state import (
    clear_models_menu_cache,
    clear_provider_wizard,
    clear_session_rename,
    get_models_menu_cache,
    get_provider_wizard,
    get_session_rename,
    get_user_provider_id,
    set_models_menu_cache,
    set_session_rename,
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


def _parse_session_id_arg(value: str) -> int:
    return int(value.strip().lstrip("#"))


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
        "/sessions - manage saved sessions\n"
        "/switch <session_id> - switch active session\n"
        "/stash - archive current session and start/resume another\n"
        "/session_delete <session_id> - permanently delete a session\n"
        "/session_rename <session_id> <title> - rename a session\n"
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
    session_id = await asyncio.to_thread(
        chat_store.create_managed_session,
        session,
        "New session",
    )
    clear_provider_wizard(context)
    await update.message.reply_text(f"Started a new session: #{session_id}.")


async def sessions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    await update.message.reply_text(
        await asyncio.to_thread(build_sessions_text, chat_store, session),
        parse_mode=ParseMode.HTML,
        reply_markup=await asyncio.to_thread(build_sessions_keyboard, chat_store, session),
    )


async def stash_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    active = await asyncio.to_thread(chat_store.get_active_managed_session, session)
    await asyncio.to_thread(chat_store.archive_managed_session, session, int(active["id"]))
    new_active = await asyncio.to_thread(chat_store.get_active_managed_session, session)
    await update.message.reply_text(
        f"Archived session #{active['id']}. Current session is now #{new_active['id']}."
    )


async def switch_session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /switch <session_id>")
        return
    try:
        switched = await asyncio.to_thread(
            chat_store.switch_managed_session,
            session,
            _parse_session_id_arg(context.args[0]),
        )
    except (ValueError, KeyError):
        await update.message.reply_text("Session not found. Use /sessions to list sessions.")
        return
    switch_msg = await update.message.reply_text(f"已切换到会话 #{switched['id']}: {switched['title']}")
    # Resend last message of the switched session as a blockquote
    recent = await asyncio.to_thread(chat_store.get_recent_messages, session, 1)
    if recent:
        await update.message.reply_text(
            f"以下是该会话的最后一条消息：\n\n<blockquote>{html.escape(recent[-1]['content'])}</blockquote>",
            parse_mode=ParseMode.HTML,
        )


async def delete_session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    if not context.args:
        await update.message.reply_text("Usage: /session_delete <session_id>")
        return
    try:
        session_id = _parse_session_id_arg(context.args[0])
        await asyncio.to_thread(chat_store.delete_managed_session, session, session_id)
    except (ValueError, KeyError):
        await update.message.reply_text("Session not found. Use /sessions to list sessions.")
        return
    await update.message.reply_text(f"Deleted session #{session_id}.")


async def rename_session_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await update.message.reply_text("Access denied for this bot.")
        return
    session = build_chat_session_key(update)
    if session is None:
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /session_rename <session_id> <title>")
        return
    try:
        session_id = _parse_session_id_arg(context.args[0])
        title = " ".join(context.args[1:]).strip()
        renamed = await asyncio.to_thread(
            chat_store.rename_managed_session,
            session,
            session_id,
            title,
        )
    except (ValueError, KeyError):
        await update.message.reply_text("Session not found. Use /sessions to list sessions.")
        return
    await update.message.reply_text(f"Renamed session #{renamed['id']}: {renamed['title']}")


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
            "获取模型列表失败，请检查供应商配置后重试。"
        )
        return

    available_set = set(available_ids)
    if requested_model in available_set:
        if requested_model == provider.current_model:
            await update.message.reply_text(
                f"{provider.name} 已在使用 {requested_model}。"
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
                f"保存模型配置失败，请重试。"
            )
            return
        await update.message.reply_text(
            f"已将 {updated_provider.name} 的模型切换为 {requested_model}。"
        )
        return

    prefix_matches = sorted(
        model_id for model_id in available_set if model_id.startswith(requested_model)
    )
    if prefix_matches:
        hint = "\n".join(f"- {model_id}" for model_id in prefix_matches[:10])
        extra = "\n..." if len(prefix_matches) > 10 else ""
        await update.message.reply_text(
            "模型 ID 不完整，请输入完整的模型 ID。\n"
            f"匹配项:\n{hint}{extra}"
        )
        return

    await update.message.reply_text(
        "无效的模型 ID，请使用 /models 菜单选择模型。"
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
            "无效的推理强度。\n"
            "可选值: <code>default</code> 或 "
            f"<code>{'</code>, <code>'.join(REASONING_EFFORT_VALUES)}</code>。",
            parse_mode=ParseMode.HTML,
        )
        return

    if normalized_effort == provider.reasoning_effort:
        await update.message.reply_text(
            f"{provider.name} 已在使用 "
            f"{format_reasoning_effort(provider.reasoning_effort)} 推理强度。"
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
            f"保存推理强度配置失败，请重试。"
        )
        return

    await update.message.reply_text(
        f"已将 {updated_provider.name} 的推理强度设为 "
        f"{format_reasoning_effort(updated_provider.reasoning_effort)}。"
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


async def session_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not update.effective_user:
        return
    if not authorized(update.effective_user.id):
        await query.answer("Access denied for this bot.", show_alert=True)
        return
    session = build_chat_session_key(update)
    if session is None:
        await query.answer("No chat session found.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    action = parts[1] if len(parts) > 1 else ""
    try:
        if action == "new":
            sid = await asyncio.to_thread(chat_store.create_managed_session, session, "New session")
            await query.answer(f"Created session #{sid}")
        elif action == "switch":
            if len(parts) < 3:
                raise ValueError("missing session id")
            item = await asyncio.to_thread(chat_store.switch_managed_session, session, int(parts[2]))
            await query.answer(f"已切换到 #{item['id']}")
            if query.message:
                await try_delete_message(context, query.message.chat_id, query.message.message_id)
                # Get last message of the switched session
                recent = await asyncio.to_thread(chat_store.get_recent_messages, session, 1)
                last_msg = recent[-1]["content"] if recent else None
                switch_msg = await query.message.reply_text(
                    f"已切换到会话 #{item['id']}: {item['title']}"
                )
                if last_msg:
                    await query.message.reply_text(
                        f"以下是该会话的最后一条消息：\n\n<blockquote>{html.escape(last_msg)}</blockquote>",
                        parse_mode=ParseMode.HTML,
                    )
            return
        elif action == "rename":
            if len(parts) < 3:
                raise ValueError("missing session id")
            session_id = int(parts[2])
            set_session_rename(context, session_id)
            await query.answer()
            await edit_callback_text(
                update,
                f"Send the new title for session <code>#{session_id}</code>.\n"
                "Use /provider_cancel to cancel this pending rename.",
                None,
            )
            return
        elif action == "archive":
            if len(parts) < 3:
                raise ValueError("missing session id")
            await asyncio.to_thread(chat_store.archive_managed_session, session, int(parts[2]))
            await query.answer("Session archived")
        elif action == "confirm_delete":
            if len(parts) < 3:
                raise ValueError("missing session id")
            session_id = int(parts[2])
            await query.answer()
            await edit_callback_text(
                update,
                f"Delete session <code>#{session_id}</code> permanently? This cannot be undone.",
                build_delete_confirm_keyboard(session_id),
            )
            return
        elif action == "delete":
            if len(parts) < 3:
                raise ValueError("missing session id")
            await asyncio.to_thread(chat_store.delete_managed_session, session, int(parts[2]))
            await query.answer("Session deleted")
        elif action == "refresh":
            await query.answer()
        else:
            await query.answer()
        await edit_callback_text(
            update,
            await asyncio.to_thread(build_sessions_text, chat_store, session),
            await asyncio.to_thread(build_sessions_keyboard, chat_store, session),
        )
    except (ValueError, KeyError):
        await query.answer("Session not found. Refresh /sessions.", show_alert=True)
    except Exception:
        logging.exception("Failed to handle session callback")
        await query.answer("Failed to update sessions.", show_alert=True)


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
                provider = runtime_config_store.set_provider_current_model(
                    provider.id,
                    model_id,
                )
                await query.answer(f"✅ 已将模型设置成 {provider.name}/{model_id}")
                # Delete the menu and send a confirmation message
                if query.message:
                    await try_delete_message(context, query.message.chat_id, query.message.message_id)
                    await query.message.reply_text(
                        f"✅ 已将模型设置成 {html.escape(provider.name)}/{html.escape(model_id)}。",
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
        "/sessions - manage saved sessions\n"
        "/switch <session_id> - switch active session\n"
        "/stash - archive current session and start/resume another\n"
        "/session_delete <session_id> - permanently delete a session\n"
        "/session_rename <session_id> <title> - rename a session\n"
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

    pending_rename = get_session_rename(context)
    if pending_rename is not None:
        session = build_chat_session_key(update)
        if session is None:
            return
        title = update.message.text.strip()
        try:
            renamed = await asyncio.to_thread(
                chat_store.rename_managed_session,
                session,
                int(pending_rename["session_id"]),
                title,
            )
        except KeyError:
            clear_session_rename(context)
            await update.message.reply_text("Session not found. Use /sessions to refresh the list.")
            return
        clear_session_rename(context)
        await update.message.reply_text(f"Renamed session #{renamed['id']}: {renamed['title']}")
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
