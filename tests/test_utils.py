import unittest

from telegram_llm_bot.utils import (
    REASONING_EFFORT_VALUES,
    ThinkTagFilter,
    extract_think_sections,
    format_reasoning_effort,
    markdown_to_telegram_html,
    normalize_reasoning_effort,
    slugify_provider_id,
    split_text_for_telegram,
    strip_think_tags,
)


class ThinkTagFilterTests(unittest.TestCase):
    def test_filter_removes_streamed_think_sections_across_chunks(self) -> None:
        filter_ = ThinkTagFilter()
        pieces = [
            filter_.feed("hello<th"),
            filter_.feed("ink>secret"),
            filter_.feed("</think> world"),
            filter_.finish(),
        ]
        self.assertEqual("".join(pieces), "hello world")

    def test_filter_drops_unclosed_think_section_on_finish(self) -> None:
        filter_ = ThinkTagFilter()
        pieces = [
            filter_.feed("visible<think>hidden"),
            filter_.finish(),
        ]
        self.assertEqual("".join(pieces), "visible")


class UtilsTests(unittest.TestCase):
    def test_strip_think_tags_removes_inline_content(self) -> None:
        text = "before<think>ignore this</think>after"
        self.assertEqual(strip_think_tags(text), "beforeafter")

    def test_extract_think_sections_returns_reasoning_and_answer(self) -> None:
        reasoning, answer = extract_think_sections(
            "before<think>step 1\nstep 2</think>after"
        )
        self.assertEqual(reasoning, "step 1\nstep 2")
        self.assertEqual(answer, "beforeafter")

    def test_reasoning_effort_helpers_normalize_and_format(self) -> None:
        self.assertEqual(normalize_reasoning_effort("HIGH"), "high")
        self.assertIsNone(normalize_reasoning_effort("default"))
        self.assertEqual(format_reasoning_effort(None), "default")
        self.assertEqual(format_reasoning_effort("low"), "low")
        self.assertIn("xhigh", REASONING_EFFORT_VALUES)

    def test_markdown_to_telegram_html_keeps_basic_formatting(self) -> None:
        rendered = markdown_to_telegram_html(
            "# Title\n**bold** `code` [link](https://example.com)"
        )
        self.assertIn("<b>Title</b>", rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn("<code>code</code>", rendered)
        self.assertIn('<a href="https://example.com">link</a>', rendered)

    def test_split_text_for_telegram_handles_empty_and_long_input(self) -> None:
        self.assertEqual(split_text_for_telegram(""), ["(empty)"])
        chunks = split_text_for_telegram("a" * 5000)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(len(chunk) for chunk in chunks), 5000)

    def test_slugify_provider_id_generates_unique_suffix(self) -> None:
        provider_id = slugify_provider_id("My Provider", {"my-provider"})
        self.assertEqual(provider_id, "my-provider-2")

    def test_slugify_provider_id_preserves_dots(self) -> None:
        provider_id = slugify_provider_id("sa.linux.yun", set())
        self.assertEqual(provider_id, "sa.linux.yun")

    def test_slugify_provider_id_strips_leading_trailing_dots(self) -> None:
        provider_id = slugify_provider_id(".my.provider.", set())
        self.assertEqual(provider_id, "my.provider")


if __name__ == "__main__":
    unittest.main()
