from telegram.ext import ContextTypes

from .constants import MODELS_MENU_CACHE_KEY, PROVIDER_WIZARD_KEY, SESSION_RENAME_KEY, USER_PROVIDER_KEY


def get_models_menu_cache(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object] | None:
    cached = context.user_data.get(MODELS_MENU_CACHE_KEY)
    if not isinstance(cached, dict):
        return None
    provider_id = cached.get("provider_id")
    ids = cached.get("ids")
    if isinstance(provider_id, str) and isinstance(ids, list) and all(
        isinstance(item, str) for item in ids
    ):
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


def get_user_provider_id(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    provider_id = context.user_data.get(USER_PROVIDER_KEY)
    return provider_id if isinstance(provider_id, str) else None


def set_user_provider_id(context: ContextTypes.DEFAULT_TYPE, provider_id: str) -> None:
    context.user_data[USER_PROVIDER_KEY] = provider_id


def get_session_rename(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object] | None:
    pending = context.user_data.get(SESSION_RENAME_KEY)
    if isinstance(pending, dict) and isinstance(pending.get("session_id"), int):
        return pending
    return None


def set_session_rename(context: ContextTypes.DEFAULT_TYPE, session_id: int) -> None:
    context.user_data[SESSION_RENAME_KEY] = {"session_id": session_id}


def clear_session_rename(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop(SESSION_RENAME_KEY, None)
