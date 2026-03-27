import threading
from dataclasses import dataclass

from .storage import ProviderConfig


@dataclass(frozen=True)
class ProviderCapabilityState:
    supports_responses: bool | None = None


class ProviderCapabilityCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str | None, str], ProviderCapabilityState] = {}

    def _key(self, provider: ProviderConfig) -> tuple[str, str | None, str]:
        return (provider.id, provider.base_url, provider.api_key)

    def get(self, provider: ProviderConfig) -> ProviderCapabilityState:
        with self._lock:
            return self._states.get(self._key(provider), ProviderCapabilityState())

    def get_supports_responses(self, provider: ProviderConfig) -> bool | None:
        return self.get(provider).supports_responses

    def set_supports_responses(
        self,
        provider: ProviderConfig,
        supported: bool,
    ) -> None:
        with self._lock:
            self._states[self._key(provider)] = ProviderCapabilityState(
                supports_responses=supported,
            )


def should_cache_unsupported_responses(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()

    if status_code in {404, 405, 415, 501}:
        return True

    if status_code == 400 and any(
        needle in message
        for needle in (
            "/responses",
            "responses api",
            "unsupported",
            "not implemented",
            "unknown request url",
        )
    ):
        return True

    return False
