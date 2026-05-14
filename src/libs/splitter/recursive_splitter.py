"""Word-based paragraph splitter for Markdown academic documents."""

from __future__ import annotations

from typing import Any

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.text_utils import count_words, is_markdown_heading, split_markdown_sections


class RecursiveSplitter(BaseSplitter):
    """Markdown-aware word-based splitter.

    Behavior:
    - Split by Markdown headings first so chunks never cross section boundaries.
    - Within each section, only split on paragraph boundaries (``"\n\n"``).
    - Treat ``chunk_size`` as a soft target in words when combining paragraphs.
    - Preserve a whole paragraph even when a single paragraph exceeds ``chunk_size``.
    - Apply overlap in paragraph units, never by cutting through a paragraph.
    """

    DEFAULT_SEPARATORS = ["\n\n"]

    def __init__(
        self,
        settings: Any,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RecursiveSplitter."""
        self.settings = settings

        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    @classmethod
    def _count_words(cls, text: str) -> int:
        """Count words for chunk sizing and display."""
        return count_words(text)

    @classmethod
    def _count_tokens(cls, text: str) -> int:
        """Backward-compatible alias for older tests and callers."""
        return cls._count_words(text)

    def split_text(
        self,
        text: str,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Split text by heading-scoped paragraph groups measured in words."""
        self.validate_text(text)

        try:
            chunks: list[str] = []
            for section in split_markdown_sections(text):
                chunks.extend(self._split_section(section))

            if not chunks:
                chunks = [text]

            self.validate_chunks(chunks)
            return chunks

        except Exception as e:
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text words: {self._count_words(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e

    def _split_section(self, section_text: str) -> list[str]:
        """Split a single Markdown section by paragraph boundaries only."""
        separator = self.separators[0] if self.separators else self.DEFAULT_SEPARATORS[0]
        paragraphs = [part.strip() for part in section_text.split(separator) if part.strip()]

        if not paragraphs:
            return [section_text]

        if len(paragraphs) >= 2 and is_markdown_heading(paragraphs[0]):
            paragraphs = [separator.join(paragraphs[:2])] + paragraphs[2:]

        return self._merge_paragraphs(paragraphs, separator)

    def _merge_paragraphs(self, paragraphs: list[str], separator: str) -> list[str]:
        """Combine paragraphs into chunks without breaking a paragraph apart."""
        chunks: list[str] = []
        paragraph_word_counts = [self._count_words(paragraph) for paragraph in paragraphs]
        start = 0

        while start < len(paragraphs):
            end = start
            current_words = 0

            while end < len(paragraphs):
                paragraph_words = paragraph_word_counts[end]

                if end == start:
                    current_words = paragraph_words
                    end += 1
                    if paragraph_words > self.chunk_size:
                        break
                    continue

                if current_words + paragraph_words > self.chunk_size:
                    break

                current_words += paragraph_words
                end += 1

            chunks.append(separator.join(paragraphs[start:end]))

            if end >= len(paragraphs):
                break

            if self.chunk_overlap <= 0:
                start = end
                continue

            overlap_words = 0
            next_start = end

            while next_start - 1 > start and overlap_words < self.chunk_overlap:
                next_start -= 1
                overlap_words += paragraph_word_counts[next_start]

            start = next_start

        return chunks
