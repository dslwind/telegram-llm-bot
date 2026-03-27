import os
import unittest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SQLITE_PATH", "/tmp/telegram-llm-bot-test-provider-wizard.db")

from telegram_llm_bot.provider_wizard import build_provider_draft_from_provider
from telegram_llm_bot.storage import ProviderConfig


class ProviderWizardTests(unittest.TestCase):
    def test_build_provider_draft_includes_reasoning_effort(self) -> None:
        provider = ProviderConfig(
            id="demo",
            name="Demo",
            base_url="https://example.com/v1",
            api_key="secret",
            default_model="gpt-4.1-mini",
            current_model="gpt-4.1-mini",
            reasoning_effort="high",
        )

        self.assertEqual(
            build_provider_draft_from_provider(provider),
            {
                "name": "Demo",
                "base_url": "https://example.com/v1",
                "api_key": "secret",
                "default_model": "gpt-4.1-mini",
                "reasoning_effort": "high",
            },
        )


if __name__ == "__main__":
    unittest.main()
