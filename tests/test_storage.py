import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from telegram_llm_bot.constants import CONFIG_VERSION
from telegram_llm_bot.storage import (
    ProviderConfig,
    RuntimeConfigStore,
    SQLiteChatStore,
    SlidingWindowRateLimiter,
)


class SQLiteChatStoreTests(unittest.TestCase):
    def test_append_read_and_clear_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "chat.db"
            store = SQLiteChatStore(str(db_path))
            self.addCleanup(store.close)

            store.append_message(1001, "user", "hello")
            store.append_message(1001, "assistant", "hi")
            store.append_message(2002, "user", "other")

            self.assertEqual(
                store.get_recent_messages(1001, 10),
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            )

            store.clear_user_history(1001)
            self.assertEqual(store.get_recent_messages(1001, 10), [])
            self.assertEqual(
                store.get_recent_messages(2002, 10),
                [{"role": "user", "content": "other"}],
            )


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def test_allow_enforces_limit_and_expires_old_events(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=5)

        with patch(
            "telegram_llm_bot.storage.time.monotonic",
            side_effect=[0.0, 1.0, 2.0, 6.1],
        ):
            self.assertEqual(limiter.allow(42), (True, 0))
            self.assertEqual(limiter.allow(42), (True, 0))
            self.assertEqual(limiter.allow(42), (False, 3))
            self.assertEqual(limiter.allow(42), (True, 0))


class RuntimeConfigStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = Path(self.temp_dir.name) / "config.json"
        self.bootstrap_provider = ProviderConfig(
            id="default",
            name="Default",
            base_url=None,
            api_key="bootstrap-key",
            default_model="gpt-4.1-mini",
            current_model="gpt-4.1-mini",
            reasoning_effort="low",
        )

    def test_bootstrap_creates_v2_config_file(self) -> None:
        store = RuntimeConfigStore(str(self.config_path), self.bootstrap_provider)

        current = store.current()
        self.assertEqual(current.version, CONFIG_VERSION)
        self.assertEqual(current.current_provider_id, "default")
        self.assertEqual(len(current.providers), 1)
        self.assertTrue(self.config_path.exists())

    def test_store_can_add_switch_update_and_delete_provider(self) -> None:
        store = RuntimeConfigStore(str(self.config_path), self.bootstrap_provider)

        added = store.add_provider(
            name="Backup Provider",
            base_url="https://example.com/v1",
            api_key="backup-key",
            default_model="model-a",
        )
        self.assertEqual(added.id, "backup-provider")

        current_provider = store.set_current_provider(added.id)
        self.assertEqual(current_provider.id, added.id)

        updated_provider = store.set_provider_current_model(added.id, "model-b")
        self.assertEqual(updated_provider.current_model, "model-b")

        updated_reasoning = store.set_provider_reasoning_effort(added.id, "high")
        self.assertEqual(updated_reasoning.reasoning_effort, "high")

        edited_provider = store.edit_provider(
            provider_id=added.id,
            name="Backup Provider Edited",
            base_url=None,
            api_key="edited-key",
            default_model="model-c",
        )
        self.assertEqual(edited_provider.name, "Backup Provider Edited")
        self.assertEqual(edited_provider.current_model, "model-c")
        self.assertEqual(edited_provider.reasoning_effort, "high")

        config_after_delete = store.delete_provider(added.id)
        self.assertEqual(config_after_delete.current_provider_id, "default")
        self.assertEqual([provider.id for provider in config_after_delete.providers], ["default"])

    def test_legacy_flat_config_is_migrated(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "openai_api_key": "legacy-key",
                    "openai_model": "legacy-model",
                    "openai_base_url": "https://legacy.example/v1",
                }
            ),
            encoding="utf-8",
        )

        store = RuntimeConfigStore(str(self.config_path), self.bootstrap_provider)

        current = store.current()
        self.assertEqual(current.version, CONFIG_VERSION)
        self.assertEqual(current.current_provider_id, "default")
        self.assertEqual(current.providers[0].api_key, "legacy-key")
        self.assertEqual(current.providers[0].current_model, "legacy-model")
        self.assertEqual(current.providers[0].reasoning_effort, "low")


if __name__ == "__main__":
    unittest.main()
