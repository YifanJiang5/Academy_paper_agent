"""Image and table enhancement for academic chunks.

Images are described with a vision LLM using the parsed figure caption plus the
original image. HTML tables are described with a text LLM using the parsed table
caption plus the raw ``<table>...</table>`` block.
"""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.core.settings import Settings, resolve_path
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm import BaseLLM, LLMFactory, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.observability.logger import get_logger

logger = get_logger(__name__)


IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")
TABLE_BLOCK_PATTERN = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)
DEFAULT_MAX_WORKERS = 3
MAX_CONTEXT_CAPTION_CHARS = 1200
MAX_DESCRIPTION_ATTEMPTS = 3


class ImageCaptioner(BaseTransform):
    """Enhance chunks containing image placeholders or HTML tables."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseVisionLLM | None = None,
        text_llm: BaseLLM | None = None,
    ) -> None:
        self.settings = settings
        self.llm: BaseVisionLLM | None = None
        self._text_llm = text_llm
        self._text_llm_init_failed = False
        self._caption_cache: dict[str, str] = {}
        self._table_cache: dict[str, str] = {}
        self.last_failures: list[dict[str, Any]] = []
        self._cache_lock = threading.Lock()

        if self.settings.vision_llm and self.settings.vision_llm.enabled:
            try:
                self.llm = llm or LLMFactory.create_vision_llm(settings)
            except Exception as exc:
                logger.error(f"Failed to initialize Vision LLM: {exc}")
        else:
            logger.warning("Vision LLM is disabled or not configured. Image captioning will skip.")

        self.prompt = self._load_prompt("config/prompts/image_captioning.txt")
        self.table_prompt = self._load_prompt("config/prompts/table_captioning.txt")

    @property
    def text_llm(self) -> BaseLLM | None:
        """Lazy-load the text LLM for table descriptions."""
        if self._text_llm is not None:
            return self._text_llm
        if self._text_llm_init_failed:
            return None

        try:
            self._text_llm = LLMFactory.create(self.settings)
            return self._text_llm
        except Exception as exc:
            self._text_llm_init_failed = True
            logger.error(f"Failed to initialize text LLM for table enhancement: {exc}")
            return None

    def _load_prompt(self, relative_path: str) -> str:
        """Load a prompt from config, with a concise fallback."""
        prompt_path = resolve_path(relative_path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()

        if "table" in relative_path:
            return (
                "Describe this academic table for retrieval augmentation. "
                "Use the same language as the table caption, refer to the "
                "explicit table label from the caption when available, and "
                "output plain text only."
            )
        return (
            "Describe this academic figure for retrieval augmentation. "
            "Use the same language as the figure caption, refer to the "
            "explicit figure label from the caption when available, and "
            "output plain text only."
        )

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Add image/table descriptions while preserving pipeline robustness."""
        if not chunks:
            return []

        image_lookup = self._build_image_lookup(chunks)
        image_requests = self._collect_image_requests(chunks, image_lookup)

        with self._cache_lock:
            self._caption_cache.clear()
            self._table_cache.clear()
            self.last_failures.clear()

        if self.llm and image_requests:
            self._generate_captions_parallel(image_requests, trace)
        elif image_requests:
            logger.warning("Skipping %s image captions because Vision LLM is unavailable.", len(image_requests))

        processed_chunks: list[Chunk] = []
        image_captioned_chunks = 0
        table_enhanced_chunks = 0

        for chunk in chunks:
            enhanced_chunk, image_count = self._apply_image_captions(chunk)
            enhanced_chunk, table_count = self._apply_table_descriptions(enhanced_chunk, trace)

            if image_count:
                image_captioned_chunks += 1
            if table_count:
                table_enhanced_chunks += 1

            processed_chunks.append(enhanced_chunk)

        if trace:
            trace.record_stage(
                "image_captioner",
                {
                    "method": "image_table_description",
                    "unique_images": len(image_requests),
                    "image_captioned_chunks": image_captioned_chunks,
                    "table_enhanced_chunks": table_enhanced_chunks,
                    "vision_llm_available": self.llm is not None,
                    "text_llm_available": self._text_llm is not None,
                    "failures": list(self.last_failures),
                },
            )

        logger.info(
            "Enhanced %s chunks with image captions and %s chunks with table descriptions",
            image_captioned_chunks,
            table_enhanced_chunks,
        )
        return processed_chunks

    def _build_image_lookup(self, chunks: list[Chunk]) -> dict[str, dict]:
        """Build image_id -> metadata from chunk metadata."""
        image_lookup: dict[str, dict] = {}
        for chunk in chunks:
            images = (chunk.metadata or {}).get("images", [])
            if not isinstance(images, list):
                continue
            for img_meta in images:
                if not isinstance(img_meta, dict):
                    continue
                img_id = img_meta.get("id")
                if img_id and img_id not in image_lookup:
                    image_lookup[str(img_id)] = img_meta
        return image_lookup

    def _collect_image_requests(
        self,
        chunks: list[Chunk],
        image_lookup: dict[str, dict],
    ) -> dict[str, dict[str, str]]:
        """Collect unique image caption requests with nearest figure captions."""
        requests: dict[str, dict[str, str]] = {}
        for chunk in chunks:
            for match in IMAGE_PLACEHOLDER_PATTERN.finditer(chunk.text or ""):
                img_id = match.group(1).strip()
                if img_id in requests:
                    continue
                img_meta = image_lookup.get(img_id)
                if not img_meta or not img_meta.get("path"):
                    continue
                requests[img_id] = {
                    "path": str(img_meta.get("path")),
                    "source_caption": self._extract_following_caption(chunk.text, match.end()),
                }
        return requests

    def _generate_captions_parallel(
        self,
        image_requests: dict[str, dict[str, str]],
        trace: TraceContext | None = None,
    ) -> None:
        """Generate captions for unique images in parallel."""
        max_workers = min(DEFAULT_MAX_WORKERS, len(image_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._get_caption,
                    img_id,
                    request["path"],
                    request.get("source_caption", ""),
                    trace,
                ): img_id
                for img_id, request in image_requests.items()
            }

            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Failed to generate caption for {img_id}: {exc}")

    def _get_caption(
        self,
        img_id: str,
        img_path: str,
        source_caption: str = "",
        trace: TraceContext | None = None,
    ) -> str | None:
        """Get a caption for one image, using cache if available."""
        with self._cache_lock:
            if img_id in self._caption_cache:
                return self._caption_cache[img_id]

        if not img_path or not Path(img_path).exists():
            error = f"Image path not found: {img_path}"
            logger.warning(error)
            self._record_failure(
                {
                    "type": "image",
                    "target": img_id,
                    "image_id": img_id,
                    "path": img_path,
                    "caption": source_caption,
                    "attempts": 0,
                    "error": error,
                }
            )
            return None
        if not self.llm:
            return None

        last_error = ""
        for attempt in range(1, MAX_DESCRIPTION_ATTEMPTS + 1):
            try:
                response = self.llm.chat_with_image(
                    text=self._build_image_prompt(source_caption),
                    image=ImageInput(path=img_path),
                    trace=trace,
                    temperature=self._generation_temperature(vision=True),
                )
                caption = response.content.strip()
            except Exception as exc:
                last_error = str(exc)
                if attempt < MAX_DESCRIPTION_ATTEMPTS:
                    logger.warning(
                        "Failed to caption image %s on attempt %s/%s: %s",
                        img_path,
                        attempt,
                        MAX_DESCRIPTION_ATTEMPTS,
                        exc,
                    )
                    continue

                logger.error(
                    "Failed to caption image %s after %s attempts: %s",
                    img_path,
                    MAX_DESCRIPTION_ATTEMPTS,
                    exc,
                )
                break

            if caption:
                with self._cache_lock:
                    self._caption_cache[img_id] = caption
                return caption

            last_error = "Vision LLM returned an empty caption"
            if attempt < MAX_DESCRIPTION_ATTEMPTS:
                logger.warning(
                    "Empty caption for image %s on attempt %s/%s",
                    img_path,
                    attempt,
                    MAX_DESCRIPTION_ATTEMPTS,
                )

        self._record_failure(
            {
                "type": "image",
                "target": img_id,
                "image_id": img_id,
                "path": img_path,
                "caption": source_caption,
                "attempts": MAX_DESCRIPTION_ATTEMPTS,
                "error": last_error,
            }
        )
        return None

    def _apply_image_captions(self, chunk: Chunk) -> tuple[Chunk, int]:
        """Replace image placeholders in chunk text and append captions in metadata text."""
        text = chunk.text or ""
        matches = list(IMAGE_PLACEHOLDER_PATTERN.finditer(text))
        if not matches:
            return chunk, 0

        new_text = text
        metadata = dict(chunk.metadata or {})
        metadata_text = str(metadata.get("text") or text)
        image_captions = self._coerce_caption_dict(metadata.get("image_captions"))
        caption_count = 0

        for match in matches:
            img_id = match.group(1).strip()
            with self._cache_lock:
                caption = self._caption_cache.get(img_id)
            if not caption:
                continue

            placeholder = match.group(0)
            new_text = new_text.replace(placeholder, caption, 1)
            metadata_text = metadata_text.replace(placeholder, f"{placeholder}\n{caption}", 1)
            image_captions[img_id] = caption
            caption_count += 1

        if caption_count == 0:
            return chunk, 0

        metadata["text"] = metadata_text
        metadata["image_captions"] = image_captions
        return (
            Chunk(
                id=chunk.id,
                text=new_text,
                metadata=metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            ),
            caption_count,
        )

    def _apply_table_descriptions(
        self,
        chunk: Chunk,
        trace: TraceContext | None = None,
    ) -> tuple[Chunk, int]:
        """Replace HTML tables in chunk text with generated table descriptions."""
        text = chunk.text or ""
        matches = list(TABLE_BLOCK_PATTERN.finditer(text))
        if not matches:
            return chunk, 0

        table_descriptions: list[dict[str, str]] = []

        def replace_table(match: re.Match[str]) -> str:
            html_table = match.group(0)
            source_caption = self._extract_preceding_caption(text, match.start())
            description = self._get_table_description(
                chunk.id,
                source_caption,
                html_table,
                trace,
            )
            if not description:
                return html_table

            table_descriptions.append(
                {
                    "caption": source_caption,
                    "description": description,
                }
            )
            return description

        new_text = TABLE_BLOCK_PATTERN.sub(replace_table, text)
        if not table_descriptions:
            return chunk, 0

        metadata = dict(chunk.metadata or {})
        existing = metadata.get("table_descriptions", [])
        if not isinstance(existing, list):
            existing = []
        metadata["table_descriptions"] = existing + table_descriptions

        return (
            Chunk(
                id=chunk.id,
                text=new_text,
                metadata=metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            ),
            len(table_descriptions),
        )

    def _get_table_description(
        self,
        chunk_id: str,
        source_caption: str,
        html_table: str,
        trace: TraceContext | None = None,
    ) -> str | None:
        """Generate a retrieval-oriented description for an HTML table."""
        cache_key = hashlib.sha256(f"{source_caption}\n{html_table}".encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._table_cache:
                return self._table_cache[cache_key]

        llm = self.text_llm
        if not llm:
            return None

        last_error = ""
        for attempt in range(1, MAX_DESCRIPTION_ATTEMPTS + 1):
            try:
                response = llm.chat(
                    [Message(role="user", content=self._build_table_prompt(source_caption, html_table))],
                    trace=trace,
                    temperature=self._generation_temperature(vision=False),
                )
                description = response.content.strip() if hasattr(response, "content") else str(response).strip()
            except Exception as exc:
                last_error = str(exc)
                if attempt < MAX_DESCRIPTION_ATTEMPTS:
                    logger.warning(
                        "Failed to describe table in chunk %s on attempt %s/%s: %s",
                        chunk_id,
                        attempt,
                        MAX_DESCRIPTION_ATTEMPTS,
                        exc,
                    )
                    continue

                logger.error(
                    "Failed to describe table in chunk %s after %s attempts: %s",
                    chunk_id,
                    MAX_DESCRIPTION_ATTEMPTS,
                    exc,
                )
                break

            if description:
                with self._cache_lock:
                    self._table_cache[cache_key] = description
                return description

            last_error = "Text LLM returned an empty table description"
            if attempt < MAX_DESCRIPTION_ATTEMPTS:
                logger.warning(
                    "Empty table description in chunk %s on attempt %s/%s",
                    chunk_id,
                    attempt,
                    MAX_DESCRIPTION_ATTEMPTS,
                )

        self._record_failure(
            {
                "type": "table",
                "target": self._extract_visual_label(source_caption, kind="table") or chunk_id,
                "chunk_id": chunk_id,
                "caption": source_caption,
                "attempts": MAX_DESCRIPTION_ATTEMPTS,
                "error": last_error,
                "table_preview": html_table[:240],
            }
        )
        return None

    def _build_image_prompt(self, source_caption: str) -> str:
        """Combine the configured image prompt with the figure caption."""
        parts = [self.prompt]
        label = self._extract_visual_label(source_caption, kind="figure")
        if label:
            parts.append(f"Detected figure label:\n{label}")
        if source_caption:
            parts.append(f"Figure caption:\n{source_caption.strip()}")
        return "\n\n".join(part for part in parts if part)

    def _build_table_prompt(self, source_caption: str, html_table: str) -> str:
        """Combine the configured table prompt with caption and HTML."""
        parts = [self.table_prompt]
        label = self._extract_visual_label(source_caption, kind="table")
        if label:
            parts.append(f"Detected table label:\n{label}")
        if source_caption:
            parts.append(f"Table caption:\n{source_caption.strip()}")
        parts.append(f"Table HTML:\n{html_table.strip()}")
        return "\n\n".join(parts)

    def _generation_temperature(self, *, vision: bool) -> float:
        """Return a provider-safe temperature for enhancement calls."""
        model = ""
        configured_temperature: float | None = None

        if vision and self.settings.vision_llm:
            model = self.settings.vision_llm.model
            configured_temperature = getattr(self.settings.vision_llm, "temperature", None)
        if not model and getattr(self.settings, "llm", None):
            model = self.settings.llm.model
        if configured_temperature is None and getattr(self.settings, "llm", None):
            configured_temperature = self.settings.llm.temperature

        if "kimi-k2.5" in str(model).lower():
            return 1.0
        return float(configured_temperature if configured_temperature is not None else 0.0)

    def _extract_following_caption(self, text: str, placeholder_end: int) -> str:
        """Extract the figure caption immediately after an image placeholder."""
        tail = text[placeholder_end:].lstrip()
        if not tail:
            return ""

        stop_positions = [len(tail)]
        blank = re.search(r"\n\s*\n", tail)
        if blank:
            stop_positions.append(blank.start())
        heading = MARKDOWN_HEADING_PATTERN.search(tail)
        if heading:
            stop_positions.append(heading.start())
        next_image = IMAGE_PLACEHOLDER_PATTERN.search(tail)
        if next_image:
            stop_positions.append(next_image.start())
        table = TABLE_BLOCK_PATTERN.search(tail)
        if table:
            stop_positions.append(table.start())

        caption = tail[: max(0, min(stop_positions))].strip()
        return caption[:MAX_CONTEXT_CAPTION_CHARS]

    def _extract_preceding_caption(self, text: str, table_start: int) -> str:
        """Extract the table caption immediately before a table block."""
        before = text[:table_start].rstrip()
        if not before:
            return ""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", before) if part.strip()]
        if not paragraphs:
            return ""

        caption = paragraphs[-1]
        if TABLE_BLOCK_PATTERN.search(caption):
            return ""
        return caption[-MAX_CONTEXT_CAPTION_CHARS:]

    def _coerce_caption_dict(self, value: Any) -> dict[str, str]:
        """Normalize existing image caption metadata to ``{image_id: caption}``."""
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, list):
            result: dict[str, str] = {}
            for item in value:
                if isinstance(item, dict) and item.get("id") and item.get("caption"):
                    result[str(item["id"])] = str(item["caption"])
            return result
        return {}

    def _record_failure(self, failure: dict[str, Any]) -> None:
        """Record an enhancement failure for trace/dashboard reporting."""
        with self._cache_lock:
            self.last_failures.append(failure)

    def _extract_visual_label(self, caption: str, *, kind: str) -> str:
        """Extract an explicit Figure/Table label from a caption when present."""
        if not caption:
            return ""

        if kind == "table":
            patterns = (
                r"\bTable\s+\d+(?:[\w.-]*|[A-Za-z])",
                r"\bTab\.\s*\d+(?:[\w.-]*|[A-Za-z])",
                r"表\s*\d+(?:[\w.-]*|[A-Za-z])",
            )
        else:
            patterns = (
                r"\bFigure\s+\d+(?:[\w.-]*|[A-Za-z])",
                r"\bFig\.\s*\d+(?:[\w.-]*|[A-Za-z])",
                r"\bFig\s+\d+(?:[\w.-]*|[A-Za-z])",
                r"图\s*\d+(?:[\w.-]*|[A-Za-z])",
            )

        for pattern in patterns:
            match = re.search(pattern, caption, flags=re.IGNORECASE)
            if match:
                return re.sub(r"\s+", " ", match.group(0)).strip()
        return ""
