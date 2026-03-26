import os
import unittest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SQLITE_PATH", "/tmp/telegram-llm-bot-test-ui.db")

from telegram_llm_bot.ui import build_reply_html_chunks


class UiReplyChunkTests(unittest.TestCase):
    def test_build_reply_chunks_keeps_plain_answer_when_no_reasoning(self) -> None:
        chunks = build_reply_html_chunks("final answer")
        self.assertEqual(chunks, ["final answer"])

    def test_build_reply_chunks_embeds_reasoning_in_expandable_blockquote(self) -> None:
        chunks = build_reply_html_chunks(
            "final answer",
            raw_text="<think>first line\nsecond line</think>\nfinal answer",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("<blockquote expandable>", chunks[0])
        self.assertIn("<b>Reasoning</b>", chunks[0])
        self.assertIn("first line", chunks[0])
        self.assertIn("final answer", chunks[0])


if __name__ == "__main__":
    unittest.main()
