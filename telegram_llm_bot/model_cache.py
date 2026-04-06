import threading
import time
from dataclasses import dataclass

from .storage import ProviderConfig


@dataclass(frozen=True)
class ModelListCacheEntry:
    ids: tuple[str, ...]
    expires_at: float


class ProviderModelListCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = max(0, ttl_seconds)
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str | None, str], ModelListCacheEntry] = {}

    def _key(self, provider: ProviderConfig) -> tuple[str, str | None, str]:
        return (provider.id, provider.base_url, provider.api_key)

    def get(self, provider: ProviderConfig) -> list[str] | None:
        if self.ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(self._key(provider))
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(self._key(provider), None)
                return None
            return list(entry.ids)

    def set(self, provider: ProviderConfig, ids: list[str]) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[self._key(provider)] = ModelListCacheEntry(
                ids=tuple(ids),
                expires_at=time.monotonic() + self.ttl_seconds,
            )

    def invalidate(self, provider: ProviderConfig) -> None:
        with self._lock:
            self._entries.pop(self._key(provider), None)
