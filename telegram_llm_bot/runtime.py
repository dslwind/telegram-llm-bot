import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .constants import CONFIG_PATH, DEFAULT_PROVIDER_ID, DEFAULT_PROVIDER_NAME
from .model_cache import ProviderModelListCache
from .provider_capabilities import ProviderCapabilityCache
from .request_gate import SessionRequestGate
from .storage import (
    ProviderConfig,
    RuntimeConfigStore,
    RuntimeConfigV2,
    SQLiteChatStore,
    SlidingWindowRateLimiter,
)
from .utils import (
    normalize_reasoning_effort,
    get_float_env,
    get_int_env,
    normalize_optional_config_text,
    parse_user_id_set,
    require_env,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
BOOTSTRAP_DEFAULT_MODEL = (
    normalize_optional_config_text(os.getenv("OPENAI_MODEL")) or "gpt-4.1-mini"
)
BOOTSTRAP_PROVIDER = ProviderConfig(
    id=DEFAULT_PROVIDER_ID,
    name=DEFAULT_PROVIDER_NAME,
    base_url=normalize_optional_config_text(os.getenv("OPENAI_BASE_URL")),
    api_key=normalize_optional_config_text(os.getenv("OPENAI_API_KEY")) or "",
    default_model=BOOTSTRAP_DEFAULT_MODEL,
    current_model=BOOTSTRAP_DEFAULT_MODEL,
    reasoning_effort=normalize_reasoning_effort(os.getenv("OPENAI_REASONING_EFFORT")),
)
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
MODELS_MENU_PAGE_SIZE = get_int_env("MODELS_MENU_PAGE_SIZE", 8)
MODELS_CACHE_TTL_SECONDS = get_int_env("MODELS_CACHE_TTL_SECONDS", 300, 0)

runtime_config_store = RuntimeConfigStore(CONFIG_PATH, BOOTSTRAP_PROVIDER)
chat_store = SQLiteChatStore(SQLITE_PATH)
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_SECONDS)
provider_capability_cache = ProviderCapabilityCache()
provider_model_list_cache = ProviderModelListCache(MODELS_CACHE_TTL_SECONDS)
session_request_gate = SessionRequestGate()


def authorized(user_id: int) -> bool:
    return not WHITELIST_USER_IDS or user_id in WHITELIST_USER_IDS


def get_runtime_config() -> RuntimeConfigV2:
    return runtime_config_store.current()


def get_current_provider() -> ProviderConfig:
    return runtime_config_store.get_current_provider()


def get_provider_for_user(user_provider_id: str | None) -> ProviderConfig:
    """Return the provider for a specific user.

    If the user has a per-user provider_id that still exists, use it.
    Otherwise fall back to the global default.
    """
    if user_provider_id:
        try:
            return runtime_config_store.get_provider(user_provider_id)
        except KeyError:
            pass
    return runtime_config_store.get_current_provider()


def get_active_model() -> str:
    return get_current_provider().current_model


def build_openai_client(provider: ProviderConfig) -> AsyncOpenAI:
    client_kwargs = {"api_key": provider.api_key}
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    return AsyncOpenAI(**client_kwargs)


async def fetch_available_model_ids(
    provider: ProviderConfig,
    *,
    use_cache: bool = True,
) -> list[str]:
    if use_cache:
        cached = provider_model_list_cache.get(provider)
        if cached is not None:
            return cached

    client = build_openai_client(provider)
    models = await client.models.list()
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

    resolved_ids = sorted(set(ids))
    if use_cache and resolved_ids:
        provider_model_list_cache.set(provider, resolved_ids)
    return resolved_ids


async def validate_provider_settings(
    base_url: str | None,
    api_key: str,
    default_model: str,
) -> tuple[bool, str, list[str]]:
    provider = ProviderConfig(
        id="validation",
        name="Validation",
        base_url=normalize_optional_config_text(base_url),
        api_key=api_key,
        default_model=default_model,
        current_model=default_model,
        reasoning_effort=None,
    )
    try:
        ids = await fetch_available_model_ids(provider, use_cache=False)
    except Exception as exc:
        logging.exception("Provider validation failed while fetching model list")
        return False, str(exc), []

    if not ids:
        return False, "Provider returned no models.", []
    if default_model not in ids:
        return False, "Default model is not available from this provider.", ids
    return True, "", ids
