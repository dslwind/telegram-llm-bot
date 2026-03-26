import unittest
from types import SimpleNamespace

from telegram_llm_bot.state import (
    clear_models_menu_cache,
    clear_provider_wizard,
    get_models_menu_cache,
    get_provider_wizard,
    set_models_menu_cache,
)


class StateHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = SimpleNamespace(user_data={})

    def test_models_menu_cache_round_trip(self) -> None:
        set_models_menu_cache(self.context, "provider-a", ["model-1", "model-2"])
        self.assertEqual(
            get_models_menu_cache(self.context),
            {"provider_id": "provider-a", "ids": ["model-1", "model-2"]},
        )

        clear_models_menu_cache(self.context)
        self.assertIsNone(get_models_menu_cache(self.context))

    def test_provider_wizard_get_and_clear(self) -> None:
        self.context.user_data["provider_wizard"] = {"step": "name"}
        self.assertEqual(get_provider_wizard(self.context), {"step": "name"})

        clear_provider_wizard(self.context)
        self.assertIsNone(get_provider_wizard(self.context))

    def test_invalid_models_cache_payload_returns_none(self) -> None:
        self.context.user_data["models_menu"] = {"provider_id": "x", "ids": ["ok", 1]}
        self.assertIsNone(get_models_menu_cache(self.context))


if __name__ == "__main__":
    unittest.main()
