"""Utilities for word counting and Markdown-aware section splitting."""

from __future__ import annotations

import logging
import re

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError:
    jieba = None  # type: ignore[assignment]


MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]*\S.*$", re.MULTILINE)
WORDLIKE_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
    re.UNICODE,
)
FALLBACK_WORD_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*"
    r"|[^\W_\d]+",
    re.UNICODE,
)


def count_words(text: str) -> int:
    """Count words in a way that fits English papers and mixed-language text."""
    if not text:
        return 0

    if jieba is not None:
        return sum(
            1
            for token in jieba.lcut(text)
            if token.strip() and WORDLIKE_TOKEN_PATTERN.search(token)
        )

    return len(FALLBACK_WORD_PATTERN.findall(text))


def is_markdown_heading(text: str) -> bool:
    """Return True when *text* is a standalone Markdown heading line."""
    stripped = text.strip()
    return "\n" not in stripped and bool(MARKDOWN_HEADING_PATTERN.match(stripped))


def split_markdown_sections(text: str) -> list[str]:
    """Split Markdown text into top-level sections while preserving heading lines."""
    if not text.strip():
        return []

    matches = list(MARKDOWN_HEADING_PATTERN.finditer(text))
    if not matches:
        return [text]

    sections: list[str] = []
    cursor = 0

    for index, match in enumerate(matches):
        start = match.start()
        if start > cursor:
            prefix = text[cursor:start].strip()
            if prefix:
                sections.append(prefix)

        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
        cursor = end

    return sections
