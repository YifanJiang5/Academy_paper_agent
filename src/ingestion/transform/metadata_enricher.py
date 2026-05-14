"""Metadata normalization transform for parsed academic chunks.

The MinerU-based parser and Markdown-aware splitter now preserve enough document
structure that this stage should not synthesize semantic metadata with rules or
LLMs. It keeps only structural metadata needed downstream and removes legacy
summary/tag/refinement markers.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.observability.logger import get_logger

logger = get_logger(__name__)


MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}[ \t]*(?P<title>.+?)\s*$", re.MULTILINE)
DEPRECATED_METADATA_FIELDS = {
    "summary",
    "tags",
    "enriched_by",
    "refined_by",
    "enrich_fallback_reason",
    "refine_fallback_reason",
}


class MetadataEnricher(BaseTransform):
    """Normalize chunk metadata without rule-based or LLM enrichment.

    Output metadata intentionally excludes legacy semantic fields:
    ``summary``, ``tags``, ``enriched_by``, ``refined_by`` and fallback markers.
    The paper-level ``title`` is preserved from parser/chunker metadata, while
    ``sub-title`` tracks the current Markdown section heading.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.use_llm = False
        self._llm = None
        self._prompt_path = prompt_path
        if llm is not None:
            logger.debug("Ignoring metadata enrichment LLM because enrichment is structural only.")

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Remove legacy enrichment fields and ensure title/sub-title metadata."""
        if not chunks:
            return []

        document_title = self._document_title(chunks)
        current_subtitle = document_title
        normalized_chunks: list[Chunk] = []
        removed_fields = 0

        for chunk in chunks:
            heading = self._extract_first_heading(chunk.text)
            if heading:
                current_subtitle = heading

            metadata = dict(chunk.metadata or {})
            for field in DEPRECATED_METADATA_FIELDS:
                if field in metadata:
                    removed_fields += 1
                    metadata.pop(field, None)

            if document_title and not metadata.get("title"):
                metadata["title"] = document_title
            if current_subtitle and not metadata.get("sub-title"):
                metadata["sub-title"] = current_subtitle

            normalized_chunks.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text or "",
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            )

        if trace:
            trace.record_stage(
                "metadata_enricher",
                {
                    "method": "structural_metadata_normalization",
                    "total_chunks": len(chunks),
                    "success_count": len(normalized_chunks),
                    "deprecated_fields_removed": removed_fields,
                    "llm_used": False,
                },
            )

        logger.info(
            "Normalized metadata for %s chunks (removed %s legacy fields)",
            len(normalized_chunks),
            removed_fields,
        )
        return normalized_chunks

    def _document_title(self, chunks: list[Chunk]) -> str:
        """Determine paper-level title from existing metadata or first heading."""
        for chunk in chunks:
            title = str((chunk.metadata or {}).get("title") or "").strip()
            if title:
                return title

        for chunk in chunks:
            heading = self._extract_first_heading(chunk.text)
            if heading:
                return heading

        return ""

    def _extract_first_heading(self, text: str) -> str | None:
        """Extract first Markdown heading from chunk text."""
        if not text:
            return None
        match = MARKDOWN_HEADING_RE.search(text)
        if not match:
            return None
        return match.group("title").strip()
