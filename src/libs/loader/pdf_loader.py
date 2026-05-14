"""PDF Loader implementation with MinerU primary parsing and MarkItDown fallback.

This module keeps the public ``PdfLoader`` contract unchanged while replacing the
primary PDF parsing path with MinerU's precision parsing API. MinerU returns a
Markdown result package (``full.md`` plus extracted images), which is normalized
to the project's canonical ``Document`` format.

Features:
- MinerU API parsing for PDF -> Markdown with table/image preservation
- Image reference normalization to ``[IMAGE: {image_id}]`` placeholders
- Compatible ``metadata.images`` generation for downstream image captioning
- MarkItDown fallback when MinerU is unavailable or fails
"""

from __future__ import annotations

import hashlib
import html
import http.client
import io
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


MINERU_API_BASE_URL = "https://mineru.net/api/v4"
MINERU_API_KEY_ENV = "MINERU_API_KEY"
MINERU_COMPLETED_STATES = {"done", "success", "completed"}
MINERU_FAILED_STATES = {"failed", "fail", "error", "canceled", "cancelled"}
MINERU_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((?P<src>[^)]+)\)")
MINERU_HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MINERU_HTML_SRC_PATTERN = re.compile(r"\bsrc\s*=\s*(['\"])(?P<src>.*?)\1", re.IGNORECASE)


@dataclass
class MinerUParsedDocument:
    """Normalized MinerU parse output used internally by ``PdfLoader``."""

    text: str
    images: List[Dict[str, Any]]
    full_md_path: Path
    output_dir: Path
    raw_result: Dict[str, Any] = field(default_factory=dict)


class PdfLoader(BaseLoader):
    """PDF Loader using MinerU as the primary parser.

    The loader still returns the same project-level ``Document`` shape:
    Markdown text, ``source_path`` metadata, ``doc_type=pdf``, stable document
    hash, and optional ``metadata.images`` entries. If MinerU cannot be used,
    the loader can fall back to the previous MarkItDown + PyMuPDF behavior.

    Configuration:
        extract_images: Enable/disable image metadata normalization.
        image_storage_dir: Base directory for stored extracted images.
        use_mineru: Enable MinerU parsing. Defaults to True.
        mineru_api_key: API token. Defaults to ``MINERU_API_KEY`` env var.
        fallback_to_markitdown: Use MarkItDown if MinerU is unavailable/fails.
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        *,
        use_mineru: bool = True,
        mineru_api_key: Optional[str] = None,
        mineru_api_base_url: str = MINERU_API_BASE_URL,
        mineru_output_dir: str | Path = "data/mineru",
        mineru_poll_interval: float = 3.0,
        mineru_timeout: float = 600.0,
        mineru_enable_formula: bool = True,
        mineru_enable_table: bool = True,
        mineru_language: str = "ch",
        mineru_layout_model: str = "vlm",
        mineru_is_ocr: bool = True,
        fallback_to_markitdown: bool = True,
    ):
        """Initialize PDF Loader.

        Args:
            extract_images: Whether to normalize extracted image references.
            image_storage_dir: Base directory for storing extracted images.
            use_mineru: Whether to prefer MinerU for PDF parsing.
            mineru_api_key: MinerU API token. Reads MINERU_API_KEY by default.
            mineru_api_base_url: MinerU API base URL.
            mineru_output_dir: Directory for downloaded MinerU result packages.
            mineru_poll_interval: Seconds between result polling attempts.
            mineru_timeout: Max seconds to wait for MinerU parsing.
            mineru_enable_formula: Whether MinerU should parse formulas.
            mineru_enable_table: Whether MinerU should parse tables.
            mineru_language: MinerU language hint.
            mineru_layout_model: MinerU layout model version.
            mineru_is_ocr: Whether to request OCR mode for uploaded PDFs.
            fallback_to_markitdown: Whether to use MarkItDown if MinerU fails.
        """
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self.use_mineru = use_mineru
        self.mineru_api_key = mineru_api_key or os.getenv(MINERU_API_KEY_ENV)
        self.mineru_api_base_url = mineru_api_base_url.rstrip("/")
        self.mineru_output_dir = Path(mineru_output_dir)
        self.mineru_poll_interval = mineru_poll_interval
        self.mineru_timeout = mineru_timeout
        self.mineru_enable_formula = mineru_enable_formula
        self.mineru_enable_table = mineru_enable_table
        self.mineru_language = mineru_language
        self.mineru_layout_model = mineru_layout_model
        self.mineru_is_ocr = mineru_is_ocr
        self.fallback_to_markitdown = fallback_to_markitdown

        self._markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None

        if not self._should_use_mineru() and not self._markitdown:
            raise ImportError(
                "PdfLoader requires either MinerU API credentials or MarkItDown. "
                f"Set {MINERU_API_KEY_ENV} or install with: pip install markitdown"
            )

        if self.fallback_to_markitdown and not self._markitdown:
            logger.warning("MarkItDown is unavailable; MinerU fallback is disabled.")

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document with Markdown text and metadata.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically and fallback is unavailable.
        """
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }

        parse_warning: Optional[str] = None
        if self._should_use_mineru():
            try:
                parsed = self._parse_with_mineru(path, doc_hash)
                text_content = parsed.text
                metadata["parser"] = "mineru"
                metadata["mineru_full_md_path"] = str(parsed.full_md_path)
                metadata["mineru_output_dir"] = str(parsed.output_dir)
                if parsed.images:
                    metadata["images"] = parsed.images
            except Exception as exc:
                parse_warning = f"MinerU parsing failed, falling back to MarkItDown: {exc}"
                logger.warning(parse_warning)
                if not self.fallback_to_markitdown:
                    raise RuntimeError(f"PDF parsing failed with MinerU: {exc}") from exc
                text_content = self._parse_with_markitdown(path)
                metadata["parser"] = "markitdown"
                metadata["parser_fallback_reason"] = str(exc)
                metadata = self._add_markitdown_images(path, text_content, doc_hash, metadata)
                text_content = metadata.pop("_text_content")
        else:
            if self.use_mineru and not self.mineru_api_key:
                parse_warning = (
                    f"{MINERU_API_KEY_ENV} is not configured; using MarkItDown fallback."
                )
                logger.info(parse_warning)
            text_content = self._parse_with_markitdown(path)
            metadata["parser"] = "markitdown"
            metadata = self._add_markitdown_images(path, text_content, doc_hash, metadata)
            text_content = metadata.pop("_text_content")

        if parse_warning:
            metadata["parser_warning"] = parse_warning

        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        return Document(id=doc_id, text=text_content, metadata=metadata)

    def _should_use_mineru(self) -> bool:
        """Return whether MinerU parsing should be attempted."""
        return bool(self.use_mineru and self.mineru_api_key)

    def _parse_with_markitdown(self, path: Path) -> str:
        """Parse PDF with MarkItDown and return Markdown text."""
        if not self._markitdown:
            raise RuntimeError("MarkItDown fallback is unavailable.")

        try:
            result = self._markitdown.convert(str(path))
            return result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as exc:
            logger.error(f"Failed to parse PDF {path} with MarkItDown: {exc}")
            raise RuntimeError(f"PDF parsing failed: {exc}") from exc

    def _add_markitdown_images(
        self,
        path: Path,
        text_content: str,
        doc_hash: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the legacy PyMuPDF image extraction path after MarkItDown parsing."""
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as exc:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {exc}"
                )
        metadata["_text_content"] = text_content
        return metadata

    def _parse_with_mineru(self, pdf_path: Path, doc_hash: str) -> MinerUParsedDocument:
        """Parse PDF via MinerU and normalize the downloaded result package."""
        batch_id, upload_url = self._mineru_create_upload_task(pdf_path, doc_hash)
        self._mineru_upload_file(upload_url, pdf_path)
        result = self._mineru_wait_for_result(batch_id, pdf_path.name, doc_hash)

        full_zip_url = result.get("full_zip_url")
        if not full_zip_url:
            raise RuntimeError("MinerU result did not include full_zip_url.")

        output_dir = self._download_and_extract_mineru_zip(full_zip_url, doc_hash)
        full_md_path = self._find_full_md(output_dir)
        text_content = full_md_path.read_text(encoding="utf-8")
        if not text_content.strip():
            raise RuntimeError(f"MinerU full.md is empty: {full_md_path}")

        images_metadata: List[Dict[str, Any]] = []
        if self.extract_images:
            text_content, images_metadata = self._replace_mineru_image_links(
                text_content, full_md_path, doc_hash
            )

        return MinerUParsedDocument(
            text=text_content,
            images=images_metadata,
            full_md_path=full_md_path,
            output_dir=output_dir,
            raw_result=result,
        )

    def _mineru_create_upload_task(self, pdf_path: Path, doc_hash: str) -> tuple[str, str]:
        """Create a MinerU upload task and return ``(batch_id, upload_url)``."""
        payload = {
            "enable_formula": self.mineru_enable_formula,
            "enable_table": self.mineru_enable_table,
            "language": self.mineru_language,
            "layout_model": self.mineru_layout_model,
            "files": [
                {
                    "name": pdf_path.name,
                    "is_ocr": self.mineru_is_ocr,
                    "data_id": doc_hash,
                }
            ],
        }
        response = self._mineru_post_json("/file-urls/batch", payload)
        data = self._mineru_response_data(response)

        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or data.get("files") or []
        if not batch_id:
            raise RuntimeError("MinerU upload task response missing batch_id.")
        if not file_urls:
            raise RuntimeError("MinerU upload task response missing file upload URL.")

        first_file = file_urls[0]
        if isinstance(first_file, dict):
            upload_url = first_file.get("url") or first_file.get("upload_url")
        else:
            upload_url = str(first_file)
        if not upload_url:
            raise RuntimeError("MinerU upload task response missing upload URL.")

        return str(batch_id), str(upload_url)

    def _mineru_wait_for_result(
        self,
        batch_id: str,
        file_name: str,
        doc_hash: str,
    ) -> Dict[str, Any]:
        """Poll MinerU until parsing is done or failed."""
        deadline = time.monotonic() + self.mineru_timeout
        last_state = "unknown"

        while time.monotonic() < deadline:
            response = self._mineru_get_json(f"/extract-results/batch/{batch_id}")
            data = self._mineru_response_data(response)
            result = self._select_mineru_result(data, file_name, doc_hash)
            if result:
                if result.get("full_zip_url"):
                    return result
                state = str(result.get("state", "")).lower()
                last_state = state or "unknown"
                if state in MINERU_COMPLETED_STATES:
                    return result
                if state in MINERU_FAILED_STATES:
                    err_msg = result.get("err_msg") or result.get("message") or result
                    raise RuntimeError(f"MinerU parsing failed: {err_msg}")

            time.sleep(self.mineru_poll_interval)

        raise TimeoutError(
            f"Timed out waiting for MinerU batch {batch_id}; last state={last_state}."
        )

    def _select_mineru_result(
        self,
        data: Dict[str, Any],
        file_name: str,
        doc_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Select this PDF's result item from MinerU batch result data."""
        results = data.get("extract_result") or data.get("results") or data.get("files") or []
        if isinstance(results, dict):
            results = [results]

        if not isinstance(results, list):
            return None

        normalized_name = Path(file_name).name
        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("data_id") == doc_hash:
                return item
            if item.get("file_name") == normalized_name or item.get("name") == normalized_name:
                return item

        return results[0] if results else None

    def _mineru_response_data(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MinerU envelope and return its data mapping."""
        code = response.get("code")
        if code not in (0, "0", None):
            message = response.get("msg") or response.get("message") or "unknown error"
            raise RuntimeError(f"MinerU API error {code}: {message}")

        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("MinerU API response missing data object.")
        return data

    def _mineru_post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to MinerU and return decoded response."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.mineru_api_base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.mineru_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._open_json_request(request)

    def _mineru_get_json(self, path: str) -> Dict[str, Any]:
        """GET JSON from MinerU and return decoded response."""
        request = urllib.request.Request(
            url=f"{self.mineru_api_base_url}{path}",
            headers={"Authorization": f"Bearer {self.mineru_api_key}"},
            method="GET",
        )
        return self._open_json_request(request)

    def _open_json_request(self, request: urllib.request.Request) -> Dict[str, Any]:
        """Execute a small JSON HTTP request with sanitized errors."""
        try:
            with urllib.request.urlopen(request, timeout=min(self.mineru_timeout, 60)) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MinerU HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MinerU request failed: {exc.reason}") from exc

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("MinerU response was not valid JSON.") from exc

        if not isinstance(decoded, dict):
            raise RuntimeError("MinerU response JSON was not an object.")
        return decoded

    def _mineru_upload_file(self, upload_url: str, pdf_path: Path) -> None:
        """Upload a local PDF to MinerU's pre-signed URL."""
        parsed = urllib.parse.urlsplit(upload_url)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError(f"Unsupported MinerU upload URL scheme: {parsed.scheme}")

        path_with_query = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_cls(parsed.netloc, timeout=min(self.mineru_timeout, 120))

        try:
            body = pdf_path.read_bytes()
            connection.request(
                "PUT",
                path_with_query,
                body=body,
                headers={"Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            response_body = response.read(500).decode("utf-8", errors="replace")
            if response.status not in {200, 201, 204}:
                raise RuntimeError(
                    f"MinerU file upload failed with HTTP {response.status}: {response_body}"
                )
        finally:
            connection.close()

    def _download_and_extract_mineru_zip(self, full_zip_url: str, doc_hash: str) -> Path:
        """Download MinerU result zip and safely extract it."""
        output_dir = (self.mineru_output_dir / doc_hash).resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        zip_path = output_dir / "mineru_result.zip"
        request = urllib.request.Request(full_zip_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.mineru_timeout, 120)) as response:
                with zip_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to download MinerU result zip: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download MinerU result zip: {exc.reason}") from exc

        try:
            with zipfile.ZipFile(zip_path) as archive:
                self._safe_extract_zip(archive, output_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError("MinerU result download was not a valid zip file.") from exc

        return output_dir

    def _safe_extract_zip(self, archive: zipfile.ZipFile, destination: Path) -> None:
        """Extract a zip file without allowing path traversal."""
        destination = destination.resolve()
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()
            if destination not in target_path.parents and target_path != destination:
                raise RuntimeError(f"Unsafe path in MinerU zip: {member.filename}")
            archive.extract(member, destination)

    def _find_full_md(self, output_dir: Path) -> Path:
        """Find MinerU's full.md in an extracted result package."""
        matches = sorted(output_dir.rglob("full.md"))
        if not matches:
            raise RuntimeError(f"MinerU result package did not contain full.md: {output_dir}")
        return matches[0]

    def _replace_mineru_image_links(
        self,
        text_content: str,
        full_md_path: Path,
        doc_hash: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Convert MinerU image links to canonical placeholders and metadata."""
        content_index = self._load_mineru_content_index(full_md_path)
        image_dir = self.image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        images_metadata: List[Dict[str, Any]] = []
        output_parts: List[str] = []
        last_end = 0
        sequence = 0

        for match in self._iter_image_matches(text_content):
            image_src = match["src"]
            source_image = self._resolve_mineru_image_path(image_src, full_md_path)
            if not source_image:
                logger.warning(f"MinerU image path not found: {image_src}")
                continue

            output_parts.append(text_content[last_end : match["start"]])

            sequence += 1
            index_item = self._lookup_mineru_content_item(content_index, image_src, source_image)
            page = self._mineru_page_number(index_item)
            image_id = self._generate_image_id(doc_hash, page or 0, sequence)
            image_ext = source_image.suffix or ".png"
            stored_image = image_dir / f"{image_id}{image_ext}"
            if source_image.resolve() != stored_image.resolve():
                shutil.copy2(source_image, stored_image)

            width, height = self._read_image_size(stored_image)
            placeholder = f"[IMAGE: {image_id}]"
            text_offset = sum(len(part) for part in output_parts)
            output_parts.append(placeholder)

            metadata = {
                "id": image_id,
                "path": self._display_image_path(stored_image),
                "page": page or 0,
                "text_offset": text_offset,
                "text_length": len(placeholder),
                "position": {
                    "width": width,
                    "height": height,
                    "page": page or 0,
                    "index": sequence,
                    "source": "mineru",
                    "original_src": image_src,
                },
            }

            if index_item:
                bbox = index_item.get("bbox") or index_item.get("poly")
                if bbox:
                    metadata["position"]["bbox"] = bbox
                item_type = index_item.get("type")
                if item_type:
                    metadata["position"]["type"] = item_type

            images_metadata.append(metadata)
            last_end = match["end"]

        output_parts.append(text_content[last_end:])
        return "".join(output_parts), images_metadata

    def _iter_image_matches(self, text_content: str) -> Iterable[Dict[str, Any]]:
        """Yield Markdown and HTML image matches in source order."""
        matches: List[Dict[str, Any]] = []

        for match in MINERU_MARKDOWN_IMAGE_PATTERN.finditer(text_content):
            raw_src = self._clean_markdown_image_src(match.group("src"))
            if raw_src:
                matches.append({"start": match.start(), "end": match.end(), "src": raw_src})

        for match in MINERU_HTML_IMAGE_PATTERN.finditer(text_content):
            src_match = MINERU_HTML_SRC_PATTERN.search(match.group(0))
            if src_match:
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "src": html.unescape(src_match.group("src")),
                    }
                )

        matches.sort(key=lambda item: item["start"])
        for item in matches:
            yield item

    def _clean_markdown_image_src(self, src: str) -> str:
        """Extract a path-like image target from Markdown image syntax."""
        src = html.unescape(src.strip())
        if src.startswith("<") and src.endswith(">"):
            return src[1:-1].strip()
        if " " in src:
            return src.split(" ", 1)[0].strip()
        return src

    def _resolve_mineru_image_path(self, src: str, full_md_path: Path) -> Optional[Path]:
        """Resolve a MinerU image reference to a local file path."""
        if self._is_external_image_ref(src):
            return None

        clean_src = self._clean_image_ref_path(src)
        candidate = Path(clean_src)
        candidates: List[Path] = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend(
                [
                    full_md_path.parent / candidate,
                    full_md_path.parent / "images" / candidate.name,
                    full_md_path.parent.parent / candidate,
                    full_md_path.parent.parent / "images" / candidate.name,
                ]
            )

        for item in candidates:
            try:
                if item.exists() and item.is_file():
                    return item.resolve()
            except OSError:
                continue

        for item in full_md_path.parent.rglob(candidate.name):
            if item.is_file():
                return item.resolve()

        return None

    def _is_external_image_ref(self, src: str) -> bool:
        """Return whether an image reference points outside the MinerU package."""
        parsed = urllib.parse.urlsplit(src)
        return parsed.scheme in {"http", "https", "data", "mailto"}

    def _clean_image_ref_path(self, src: str) -> str:
        """Remove URL quoting, query strings, and fragments from image path refs."""
        no_fragment = src.split("#", 1)[0]
        no_query = no_fragment.split("?", 1)[0]
        return urllib.parse.unquote(no_query).strip()

    def _load_mineru_content_index(self, full_md_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load nearby MinerU content-list JSON files as an image metadata index."""
        index: Dict[str, Dict[str, Any]] = {}
        for json_path in sorted(full_md_path.parent.rglob("*content_list*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug(f"Skipping unreadable MinerU content list {json_path}: {exc}")
                continue

            for item in self._walk_mineru_content_items(payload):
                img_path = item.get("img_path") or item.get("image_path")
                if not img_path:
                    continue
                normalized = self._normalize_index_key(str(img_path))
                index[normalized] = item
                index[Path(normalized).name] = item

        return index

    def _walk_mineru_content_items(self, payload: Any) -> Iterable[Dict[str, Any]]:
        """Yield JSON dicts that look like MinerU content items."""
        if isinstance(payload, dict):
            if "img_path" in payload or "image_path" in payload:
                yield payload
            for value in payload.values():
                yield from self._walk_mineru_content_items(value)
        elif isinstance(payload, list):
            for value in payload:
                yield from self._walk_mineru_content_items(value)

    def _lookup_mineru_content_item(
        self,
        content_index: Dict[str, Dict[str, Any]],
        original_src: str,
        image_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """Find content-list metadata for an image reference."""
        keys = [
            self._normalize_index_key(original_src),
            Path(self._normalize_index_key(original_src)).name,
            image_path.name,
        ]
        for key in keys:
            if key in content_index:
                return content_index[key]
        return None

    def _normalize_index_key(self, value: str) -> str:
        """Normalize MinerU content-list image path keys."""
        cleaned = self._clean_image_ref_path(value)
        return cleaned.replace("\\", "/").lstrip("./")

    def _mineru_page_number(self, item: Optional[Dict[str, Any]]) -> Optional[int]:
        """Convert MinerU page index metadata to a 1-based page number."""
        if not item:
            return None

        for key in ("page", "page_num"):
            value = item.get(key)
            if isinstance(value, int):
                return value

        page_idx = item.get("page_idx")
        if isinstance(page_idx, int):
            return page_idx + 1

        return None

    def _read_image_size(self, image_path: Path) -> tuple[int, int]:
        """Read image dimensions, returning zeros if unavailable."""
        if not PIL_AVAILABLE or Image is None:
            return 0, 0
        try:
            with Image.open(image_path) as image:
                return image.size
        except Exception:
            return 0, 0

    def _display_image_path(self, image_path: Path) -> str:
        """Return an image path that downstream components can open."""
        absolute = image_path.resolve()
        try:
            return str(absolute.relative_to(Path.cwd()))
        except ValueError:
            return str(absolute)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line.

        Args:
            text: Markdown text content.

        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split("\n")

        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line

        return None

    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Extract images from PDF and insert placeholders.

        This is the legacy MarkItDown fallback image path. MinerU result images
        are handled by ``_replace_mineru_image_links`` instead.
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []

        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []

        images_metadata = []
        modified_text = text_content

        try:
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)

            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        try:
                            if not PIL_AVAILABLE or Image is None:
                                raise RuntimeError("Pillow is unavailable")
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0

                        placeholder = f"[IMAGE: {image_id}]"
                        insert_position = len(modified_text)
                        modified_text += f"\n{placeholder}\n"

                        image_metadata = {
                            "id": image_id,
                            "path": self._display_image_path(image_path),
                            "page": page_num + 1,
                            "text_offset": insert_position + 1,
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index,
                            },
                        }
                        images_metadata.append(image_metadata)

                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")

                    except Exception as exc:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num + 1}: {exc}"
                        )
                        continue

            doc.close()

            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")

            return modified_text, images_metadata

        except Exception as exc:
            logger.warning(f"Image extraction failed for {pdf_path}: {exc}")
            return text_content, []

    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID.

        Args:
            doc_hash: Document hash.
            page: Page number.
            sequence: Image sequence.

        Returns:
            Unique image ID string.
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
