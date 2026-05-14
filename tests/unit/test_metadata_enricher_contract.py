"""Contract tests for structural metadata normalization."""

from unittest.mock import Mock

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher


def _settings() -> Mock:
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {"use_llm": True}
    return settings


def test_removes_legacy_summary_tags_and_source_markers() -> None:
    chunk = Chunk(
        id="chunk_001",
        text="# Abstract\n\nThis paper studies retrieval.",
        metadata={
            "source_path": "paper.pdf",
            "title": "Paper Title",
            "summary": "old summary",
            "tags": ["old"],
            "enriched_by": "llm",
            "refined_by": "rule",
            "enrich_fallback_reason": "llm_failed",
        },
    )

    result = MetadataEnricher(_settings()).transform([chunk])

    metadata = result[0].metadata
    assert metadata["title"] == "Paper Title"
    assert metadata["sub-title"] == "Abstract"
    assert "summary" not in metadata
    assert "tags" not in metadata
    assert "enriched_by" not in metadata
    assert "refined_by" not in metadata
    assert "enrich_fallback_reason" not in metadata


def test_tracks_current_subtitle_across_chunks_without_heading() -> None:
    chunks = [
        Chunk(
            id="chunk_001",
            text="# Introduction\n\nFirst paragraph.",
            metadata={"source_path": "paper.pdf", "title": "Paper Title"},
        ),
        Chunk(
            id="chunk_002",
            text="Second paragraph in the same section.",
            metadata={"source_path": "paper.pdf", "title": "Paper Title"},
        ),
    ]

    result = MetadataEnricher(_settings()).transform(chunks)

    assert result[0].metadata["sub-title"] == "Introduction"
    assert result[1].metadata["sub-title"] == "Introduction"


def test_does_not_call_llm_even_when_enabled() -> None:
    llm = Mock()
    chunk = Chunk(
        id="chunk_001",
        text="# Methods\n\nContent.",
        metadata={"source_path": "paper.pdf", "title": "Paper Title"},
    )

    result = MetadataEnricher(_settings(), llm=llm).transform([chunk])

    assert result[0].metadata["title"] == "Paper Title"
    llm.chat.assert_not_called()


def test_trace_records_structural_normalization() -> None:
    chunk = Chunk(
        id="chunk_001",
        text="# Results\n\nContent.",
        metadata={"source_path": "paper.pdf", "title": "Paper Title", "summary": "old"},
    )
    trace = TraceContext(trace_id="test_trace")

    MetadataEnricher(_settings()).transform([chunk], trace=trace)

    stage_data = trace.get_stage_data("metadata_enricher")
    assert stage_data["method"] == "structural_metadata_normalization"
    assert stage_data["total_chunks"] == 1
    assert stage_data["deprecated_fields_removed"] == 1
    assert stage_data["llm_used"] is False
