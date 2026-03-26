import html
import os
import re

from .constants import TELEGRAM_TEXT_LIMIT, THINK_CLOSE_TAG, THINK_OPEN_TAG

REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer env: {name}") from exc
    return max(minimum, value)


def get_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"Invalid float env: {name}") from exc
    return max(minimum, value)


def parse_user_id_set(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    user_ids: set[int] = set()
    for token in raw.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        try:
            user_ids.add(int(candidate))
        except ValueError as exc:
            raise RuntimeError(
                "WHITELIST_USER_IDS must be a comma-separated integer list"
            ) from exc
    return user_ids


def normalize_optional_config_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def normalize_reasoning_effort(value: object) -> str | None:
    normalized = normalize_optional_config_text(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered in {"default", "auto"}:
        return None
    if lowered not in REASONING_EFFORT_VALUES:
        raise RuntimeError(
            "Invalid reasoning effort. Expected one of: "
            + ", ".join(REASONING_EFFORT_VALUES)
        )
    return lowered


def normalize_required_text(value: object, field_name: str) -> str:
    normalized = normalize_optional_config_text(value)
    if normalized is None:
        raise RuntimeError(f"Missing required config field: {field_name}")
    return normalized


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    stripped = re.sub(
        r"(?is)<think>.*?(?:</think>|$)",
        "",
        text,
    )
    stripped = re.sub(r"(?is)</think>", "", stripped)
    return stripped


def extract_think_sections(text: str) -> tuple[str, str]:
    if not text:
        return "", text
    sections = [
        match.strip()
        for match in re.findall(r"(?is)<think>(.*?)(?:</think>|$)", text)
        if match.strip()
    ]
    reasoning = "\n\n".join(sections)
    answer = strip_think_tags(text).strip()
    return reasoning, answer


def _partial_tag_suffix_length(text: str, patterns: tuple[str, ...]) -> int:
    lower_text = text.lower()
    best = 0
    for pattern in patterns:
        limit = min(len(lower_text), len(pattern) - 1)
        for size in range(limit, 0, -1):
            if lower_text[-size:] == pattern[:size]:
                best = max(best, size)
                break
    return best


class ThinkTagFilter:
    def __init__(self) -> None:
        self._pending = ""
        self._inside_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._pending += chunk
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        output: list[str] = []
        while self._pending:
            lower_pending = self._pending.lower()
            if self._inside_think:
                close_index = lower_pending.find(THINK_CLOSE_TAG)
                if close_index == -1:
                    if final:
                        self._pending = ""
                    else:
                        keep = min(len(self._pending), len(THINK_CLOSE_TAG) - 1)
                        self._pending = self._pending[-keep:] if keep else ""
                    break
                self._pending = self._pending[close_index + len(THINK_CLOSE_TAG) :]
                self._inside_think = False
                continue

            open_index = lower_pending.find(THINK_OPEN_TAG)
            close_index = lower_pending.find(THINK_CLOSE_TAG)

            if close_index != -1 and (open_index == -1 or close_index < open_index):
                if close_index > 0:
                    output.append(self._pending[:close_index])
                self._pending = self._pending[close_index + len(THINK_CLOSE_TAG) :]
                continue

            if open_index != -1:
                if open_index > 0:
                    output.append(self._pending[:open_index])
                self._pending = self._pending[open_index + len(THINK_OPEN_TAG) :]
                self._inside_think = True
                continue

            if final:
                output.append(self._pending)
                self._pending = ""
            else:
                keep = _partial_tag_suffix_length(
                    self._pending,
                    (THINK_OPEN_TAG, THINK_CLOSE_TAG),
                )
                emit_end = len(self._pending) - keep
                if emit_end > 0:
                    output.append(self._pending[:emit_end])
                    self._pending = self._pending[emit_end:]
                break

        return "".join(output)


def mask_secret(value: str, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return "(missing)"
    if len(value) <= prefix + suffix:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


def format_base_url(base_url: str | None) -> str:
    return base_url or "https://api.openai.com/v1 (default)"


def format_reasoning_effort(reasoning_effort: str | None) -> str:
    return reasoning_effort or "default"


def slugify_provider_id(name: str, existing_ids: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "provider"
    base = base[:24].rstrip("-") or "provider"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def normalize_base_url_input(text: str) -> str | None:
    candidate = text.strip()
    if not candidate or candidate.lower() in {"-", "none", "null", "official", "default"}:
        return None
    return candidate


def is_yes_text(text: str) -> bool:
    return text.strip().lower() in {"y", "yes", "save", "confirm"}


def is_no_text(text: str) -> bool:
    return text.strip().lower() in {"n", "no", "cancel"}


def truncate_for_telegram(text: str) -> str:
    if len(text) <= TELEGRAM_TEXT_LIMIT:
        return text
    return text[: TELEGRAM_TEXT_LIMIT - 3] + "..."


def split_text_for_telegram(text: str) -> list[str]:
    if not text:
        return ["(empty)"]
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + TELEGRAM_TEXT_LIMIT, length)
        chunks.append(text[start:end])
        start = end
    return chunks


def markdown_to_telegram_html(text: str) -> str:
    if not text:
        return text

    slots: dict[str, str] = {}
    slot_index = 0

    def stash(value: str) -> str:
        nonlocal slot_index
        key = f"TGSLOT{slot_index}TOKEN"
        slot_index += 1
        slots[key] = value
        return key

    def replace_code_block(match: re.Match[str]) -> str:
        code_text = match.group(1).strip("\n")
        return stash(f"<pre><code>{html.escape(code_text)}</code></pre>")

    def replace_inline_code(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    def replace_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return stash(f'<a href="{url}">{label}</a>')

    text = re.sub(r"```(?:[^\n`]*)\n?(.*?)```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", replace_inline_code, text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", replace_link, text)

    escaped = html.escape(text)
    escaped = re.sub(r"(?m)^\s*#{1,6}\s+(.+)$", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "────────", escaped)
    escaped = re.sub(r"(?m)^(\s*)[-*]\s+(.+)$", r"\1• \2", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped, flags=re.DOTALL)
    escaped = re.sub(r"~~(.+?)~~", r"<s>\1</s>", escaped, flags=re.DOTALL)
    escaped = re.sub(
        r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)",
        r"<i>\1</i>",
        escaped,
        flags=re.DOTALL,
    )
    escaped = re.sub(
        r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)",
        r"<i>\1</i>",
        escaped,
        flags=re.DOTALL,
    )

    for key, value in slots.items():
        escaped = escaped.replace(key, value)

    return escaped
