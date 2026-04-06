import unittest
from unittest.mock import patch

from telegram_llm_bot.model_cache import ProviderModelListCache
from telegram_llm_bot.storage import ProviderConfig


class ProviderModelListCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider_a = ProviderConfig(
            id="demo",
            name="Demo",
            base_url="https://example.com/v1",
            api_key="secret",
            default_model="a",
            current_model="a",
            reasoning_effort=None,
        )
        self.provider_b = ProviderConfig(
            id="demo",
            name="Demo",
            base_url="https://example.com/v1",
            api_key="secret",
            default_model="a",
            current_model="b",
            reasoning_effort="low",
        )

    def test_cache_key_is_stable_across_model_switches(self) -> None:
        cache = ProviderModelListCache(ttl_seconds=60)
        with patch("telegram_llm_bot.model_cache.time.monotonic", return_value=10.0):
            cache.set(self.provider_a, ["a", "b"])
        with patch("telegram_llm_bot.model_cache.time.monotonic", return_value=20.0):
            self.assertEqual(cache.get(self.provider_b), ["a", "b"])

    def test_cache_entry_expires_after_ttl(self) -> None:
        cache = ProviderModelListCache(ttl_seconds=5)
        with patch("telegram_llm_bot.model_cache.time.monotonic", return_value=10.0):
            cache.set(self.provider_a, ["a"])
        with patch("telegram_llm_bot.model_cache.time.monotonic", return_value=14.0):
            self.assertEqual(cache.get(self.provider_a), ["a"])
        with patch("telegram_llm_bot.model_cache.time.monotonic", return_value=16.0):
            self.assertIsNone(cache.get(self.provider_a))


if __name__ == "__main__":
    unittest.main()
