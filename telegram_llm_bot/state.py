from telegram.ext import ContextTypes

from .constants import MODELS_MENU_CACHE_KEY, PROVIDER_WIZARD_KEY


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
