import unittest

from telegram_llm_bot.provider_capabilities import (
    ProviderCapabilityCache,
    should_cache_unsupported_responses,
)
from telegram_llm_bot.storage import ProviderConfig


class ProviderCapabilityCacheTests(unittest.TestCase):
    def test_cache_key_is_stable_across_model_switches(self) -> None:
        cache = ProviderCapabilityCache()
        provider_a = ProviderConfig(
            id="demo",
            name="Demo",
            base_url="https://example.com/v1",
            api_key="secret",
            default_model="a",
            current_model="a",
            reasoning_effort=None,
        )
        provider_b = ProviderConfig(
            id="demo",
            name="Demo",
            base_url="https://example.com/v1",
            api_key="secret",
            default_model="a",
            current_model="b",
            reasoning_effort="low",
        )

        cache.set_supports_responses(provider_a, False)
        self.assertFalse(cache.get_supports_responses(provider_b))

    def test_should_cache_unsupported_responses_uses_status_and_message(self) -> None:
        class UnsupportedError(Exception):
            status_code = 404

        class GenericBadRequest(Exception):
            status_code = 400

        self.assertTrue(should_cache_unsupported_responses(UnsupportedError("not found")))
        self.assertTrue(
            should_cache_unsupported_responses(
                GenericBadRequest("unsupported /responses endpoint")
            )
        )
        self.assertFalse(should_cache_unsupported_responses(GenericBadRequest("temporary issue")))


if __name__ == "__main__":
    unittest.main()
