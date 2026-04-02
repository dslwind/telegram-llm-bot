import html

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from .constants import PROVIDER_WIZARD_KEY
from .runtime import runtime_config_store, validate_provider_settings
from .state import clear_provider_wizard, get_provider_wizard
from .storage import ProviderConfig
from .ui import try_delete_message
from .utils import (
    format_base_url,
    format_reasoning_effort,
    is_no_text,
    is_yes_text,
    mask_secret,
    normalize_base_url_input,
    normalize_optional_config_text,
    normalize_reasoning_effort,
    normalize_required_text,
)


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
            "Provider setup: send the provider display name.\n\n<i>Send /provider_cancel to abort.</i>"
            f"Current {current_label}: <code>{html.escape(str(draft['name']))}</code>"
        )
    elif step == "base_url":
        text = (
            "Provider setup: send the base URL.\n\n<i>Send /provider_cancel to abort.</i>"
            "Send <code>-</code>, <code>official</code>, or <code>none</code> for the official OpenAI endpoint.\n"
            f"Current {current_label}: <code>{html.escape(format_base_url(draft['base_url']))}</code>"
        )
    elif step == "api_key":
        text = (
            "Provider setup: send the API key for this provider.\n\n<i>Send /provider_cancel to abort.</i>"
            f"Current {current_label}: <code>{html.escape(mask_secret(str(draft['api_key'])))}</code>"
        )
    elif step == "default_model":
        text = (
            "Provider setup: send the default model id for this provider.\n\n<i>Send /provider_cancel to abort.</i>"
            f"Current {current_label}: <code>{html.escape(str(draft['default_model']))}</code>"
        )
    elif step == "reasoning_effort":
        text = (
            "Provider setup: send the reasoning effort for this provider.\n\n<i>Send /provider_cancel to abort.</i>"
            "Send <code>default</code> to use the model default, or one of "
            "<code>none</code>, <code>minimal</code>, <code>low</code>, "
            "<code>medium</code>, <code>high</code>, <code>xhigh</code>.\n"
            f"Current {current_label}: <code>{html.escape(format_reasoning_effort(draft['reasoning_effort']))}</code>"
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
            f"default_model: <code>{html.escape(str(draft['default_model']))}</code>\n"
            f"reasoning_effort: <code>{html.escape(format_reasoning_effort(draft['reasoning_effort']))}</code>\n\n"
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
        "reasoning_effort": provider.reasoning_effort,
    }


async def start_provider_wizard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: str,
    provider: ProviderConfig | None = None,
) -> None:
    default_draft = (
        build_provider_draft_from_provider(provider)
        if provider
        else {
            "name": "",
            "base_url": None,
            "api_key": "",
            "default_model": "",
            "reasoning_effort": None,
        }
    )
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
    reasoning_effort = normalize_reasoning_effort(draft["reasoning_effort"])

    is_valid, error_message, ids = await validate_provider_settings(
        base_url,
        api_key,
        default_model,
    )
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
            reasoning_effort=reasoning_effort,
        )
        clear_provider_wizard(context)
        await update.message.reply_text(
            f"Provider added: {provider.name} ({provider.id}).\n"
            "Use /providers or /models to switch to it."
        )
        return

    if provider_id is None:
        clear_provider_wizard(context)
        await update.message.reply_text(
            "Provider edit session lost its provider id. Please try again."
        )
        return

    provider = runtime_config_store.edit_provider(
        provider_id=provider_id,
        name=normalize_required_text(draft["name"], "provider name"),
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        reasoning_effort=reasoning_effort,
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
        wizard["step"] = "reasoning_effort"
        await prompt_provider_wizard_step(update, context)
        return True

    if step == "reasoning_effort":
        try:
            draft["reasoning_effort"] = normalize_reasoning_effort(text)
        except RuntimeError:
            await update.message.reply_text(
                "Invalid reasoning effort.\n"
                "Use <code>default</code> or one of: <code>none</code>, "
                "<code>minimal</code>, <code>low</code>, <code>medium</code>, "
                "<code>high</code>, <code>xhigh</code>.",
                parse_mode=ParseMode.HTML,
            )
            return True
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
        await update.message.reply_text(
            "Use /skip to keep the current value, or /provider_cancel to stop."
        )
        return True
    return True
