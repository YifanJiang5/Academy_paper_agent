"""Unit tests for RecursiveSplitter."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


def create_mock_settings(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Any:
    """Create a mock settings object with ingestion config."""
    settings = MagicMock()
    settings.ingestion = MagicMock()
    settings.ingestion.chunk_size = chunk_size
    settings.ingestion.chunk_overlap = chunk_overlap
    return settings


class TestRecursiveSplitterConfiguration:
    """Tests for RecursiveSplitter configuration and initialization."""

    def test_initialization_from_settings(self):
        settings = create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(settings=settings)

        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert isinstance(splitter, BaseSplitter)

    def test_initialization_with_overrides(self):
        settings = create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(
            settings=settings,
            chunk_size=300,
            chunk_overlap=50,
        )

        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50

    def test_initialization_with_custom_separators(self):
        settings = create_mock_settings()
        custom_separators = ["\n\n", "\n"]
        splitter = RecursiveSplitter(settings=settings, separators=custom_separators)

        assert splitter.separators == custom_separators

    def test_initialization_default_separators(self):
        settings = create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)

        assert splitter.separators == ["\n\n"]

    def test_initialization_missing_settings(self):
        settings = MagicMock()
        settings.ingestion = None

        with pytest.raises(ValueError, match="Missing ingestion configuration"):
            RecursiveSplitter(settings=settings)

    def test_initialization_invalid_chunk_size(self):
        settings = create_mock_settings(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)

    def test_initialization_invalid_chunk_overlap(self):
        settings = create_mock_settings(chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
            RecursiveSplitter(settings=settings)

    def test_initialization_overlap_cannot_reach_chunk_size(self):
        settings = create_mock_settings(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)


class TestRecursiveSplitterWordBasedSplitting:
    """Tests for word-based section-aware splitting behavior."""

    def test_word_counter_uses_words_not_characters(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=10, chunk_overlap=0))

        assert splitter._count_words("hello word") == 2
        assert splitter._count_words("supercalifragilisticexpialidocious") == 1

    def test_split_short_text(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=100, chunk_overlap=0))

        text = "This is a short text."
        chunks = splitter.split_text(text)

        assert chunks == [text]

    def test_chunk_size_uses_words_not_characters(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=3, chunk_overlap=0))

        text = ("a" * 50) + "\n\n" + ("b" * 50)
        chunks = splitter.split_text(text)

        assert len(text) > splitter.chunk_size
        assert splitter._count_words(text) <= splitter.chunk_size
        assert chunks == [text]

    def test_split_text_by_paragraph_targets_word_limit(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=5, chunk_overlap=0))

        text = "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta eta."
        chunks = splitter.split_text(text)

        assert chunks == [
            "Alpha beta.\n\nGamma delta.",
            "Epsilon zeta eta.",
        ]
        assert splitter._count_words(chunks[0]) == 4
        assert splitter._count_words(chunks[1]) == 3

    def test_split_keeps_markdown_sections_isolated(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=20, chunk_overlap=0))

        text = (
            "# Abstract\n\nAlpha beta gamma.\n\n"
            "# Introduction\n\nDelta epsilon zeta."
        )
        chunks = splitter.split_text(text)

        assert len(chunks) == 2
        assert chunks[0].startswith("# Abstract")
        assert "# Introduction" not in chunks[0]
        assert chunks[1].startswith("# Introduction")

    def test_heading_stays_with_first_paragraph(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=4, chunk_overlap=0))

        text = "# Abstract\n\nalpha beta gamma delta"
        chunks = splitter.split_text(text)

        assert chunks == [text]

    def test_split_does_not_fall_back_to_words_or_sentences(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=3, chunk_overlap=0))

        text = "Alpha one. Beta two. Gamma three."
        chunks = splitter.split_text(text)

        assert chunks == [text]

    def test_split_text_with_paragraph_overlap(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=4, chunk_overlap=2))

        text = "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta.\n\nEta theta."
        chunks = splitter.split_text(text)

        assert chunks == [
            "Alpha beta.\n\nGamma delta.",
            "Gamma delta.\n\nEpsilon zeta.",
            "Epsilon zeta.\n\nEta theta.",
        ]

    def test_single_paragraph_longer_than_chunk_size_is_preserved(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=3, chunk_overlap=0))

        text = "# Abstract\n\nalpha beta gamma delta epsilon"
        chunks = splitter.split_text(text)

        assert chunks == [text]
        assert splitter._count_words(chunks[0]) > splitter.chunk_size


class TestRecursiveSplitterMarkdownAndEdgeCases:
    """Tests for Markdown preservation, validation, and edge cases."""

    def test_split_markdown_code_blocks(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=150, chunk_overlap=0))

        text = """# Example

Some text before code.

```python
def example():
    return "code"
```

Some text after code."""

        chunks = splitter.split_text(text)
        all_text = "".join(chunks)

        assert "def example():" in all_text
        assert "Some text before code." in all_text
        assert "Some text after code." in all_text

    def test_split_markdown_lists(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=100, chunk_overlap=0))

        text = """# List Example

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5"""

        chunks = splitter.split_text(text)
        all_text = "".join(chunks)

        assert "- Item 1" in all_text
        assert "- Item 5" in all_text

    def test_split_very_long_text(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=100, chunk_overlap=20))

        text = "\n\n".join(f"Paragraph {i} words." for i in range(100))
        chunks = splitter.split_text(text)

        assert len(chunks) >= 2
        assert chunks[0].startswith("Paragraph 0")
        assert "Paragraph 99" in chunks[-1]

    def test_split_unicode_text(self):
        splitter = RecursiveSplitter(settings=create_mock_settings(chunk_size=50, chunk_overlap=0))

        text = "Hello 世界! Привет мир! 🌍🌎🌏"
        chunks = splitter.split_text(text)
        all_text = "".join(chunks)

        assert "世界" in all_text
        assert "мир" in all_text
        assert "🌍" in all_text

    def test_split_with_trace_parameter(self):
        splitter = RecursiveSplitter(settings=create_mock_settings())

        chunks = splitter.split_text("Some text to split.", trace=MagicMock())
        assert len(chunks) == 1

    def test_split_empty_string_validation(self):
        splitter = RecursiveSplitter(settings=create_mock_settings())

        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.split_text("   ")

    def test_split_non_string_validation(self):
        splitter = RecursiveSplitter(settings=create_mock_settings())

        with pytest.raises(ValueError, match="must be a string"):
            splitter.split_text(123)  # type: ignore[arg-type]


class TestRecursiveSplitterFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_can_create_recursive_splitter(self):
        SplitterFactory.register_provider("recursive", RecursiveSplitter)

        settings = create_mock_settings(chunk_size=500, chunk_overlap=100)
        settings.ingestion.splitter = "recursive"

        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
