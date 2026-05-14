"""Tests for image and table enhancement transforms."""

from types import SimpleNamespace

from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_llm import ChatResponse


class FakeVisionLLM:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.calls = []
        self.failures_before_success = failures_before_success

    def chat_with_image(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures_before_success:
            raise TimeoutError("Request timed out after 60 seconds")
        return ChatResponse(content="A concise figure description.", model="kimi-k2.5")


class FakeTextLLM:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.calls = []
        self.failures_before_success = failures_before_success

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if len(self.calls) <= self.failures_before_success:
            raise TimeoutError("Request timed out after 60 seconds")
        return ChatResponse(content="A concise table description.", model="kimi-k2.5")


def _settings():
    return SimpleNamespace(
        llm=SimpleNamespace(model="kimi-k2.5", temperature=0.0, max_tokens=4096, provider="openai"),
        vision_llm=SimpleNamespace(
            enabled=True,
            provider="openai",
            model="kimi-k2.5",
            max_image_size=2048,
            temperature=1.0,
        ),
    )


def test_image_caption_uses_figure_caption_and_replaces_placeholder(tmp_path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake image bytes")
    vision_llm = FakeVisionLLM()
    captioner = ImageCaptioner(_settings(), llm=vision_llm)
    chunk = Chunk(
        id="chunk_001",
        text="[IMAGE: img_001]\nFigure 1: Overview of the proposed RAG pipeline.\n\nBody text.",
        metadata={
            "source_path": "paper.pdf",
            "images": [{"id": "img_001", "path": str(image_path), "page": 1}],
        },
    )

    result = captioner.transform([chunk])[0]

    assert "[IMAGE: img_001]" not in result.text
    assert "A concise figure description." in result.text
    assert "[IMAGE: img_001]\nA concise figure description." in result.metadata["text"]
    assert result.metadata["image_captions"]["img_001"] == "A concise figure description."
    assert "Figure 1: Overview" in vision_llm.calls[0]["text"]
    assert "Detected figure label:\nFigure 1" in vision_llm.calls[0]["text"]
    assert vision_llm.calls[0]["temperature"] == 1.0


def test_table_description_uses_caption_and_replaces_html() -> None:
    text_llm = FakeTextLLM()
    captioner = ImageCaptioner(_settings(), llm=FakeVisionLLM(), text_llm=text_llm)
    html_table = "<table><tr><th>Model</th><th>Score</th></tr><tr><td>A</td><td>92</td></tr></table>"
    chunk = Chunk(
        id="chunk_001",
        text=f"Table 1: Main retrieval results.\n\n{html_table}\n\nFollowing text.",
        metadata={"source_path": "paper.pdf"},
    )

    result = captioner.transform([chunk])[0]

    assert html_table not in result.text
    assert "A concise table description." in result.text
    assert result.metadata["table_descriptions"][0]["description"] == "A concise table description."
    prompt = text_llm.calls[0]["messages"][0].content
    assert "Table 1: Main retrieval results." in prompt
    assert "Detected table label:\nTable 1" in prompt
    assert html_table in prompt
    assert text_llm.calls[0]["temperature"] == 1.0
    assert "text" not in result.metadata


def test_image_caption_retries_before_success(tmp_path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake image bytes")
    vision_llm = FakeVisionLLM(failures_before_success=1)
    captioner = ImageCaptioner(_settings(), llm=vision_llm)
    chunk = Chunk(
        id="chunk_001",
        text="[IMAGE: img_001]\nFigure 2: Retrieval architecture.",
        metadata={
            "source_path": "paper.pdf",
            "images": [{"id": "img_001", "path": str(image_path), "page": 1}],
        },
    )

    result = captioner.transform([chunk])[0]

    assert len(vision_llm.calls) == 2
    assert "A concise figure description." in result.text
    assert captioner.last_failures == []


def test_image_caption_records_failure_after_three_attempts(tmp_path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake image bytes")
    vision_llm = FakeVisionLLM(failures_before_success=3)
    captioner = ImageCaptioner(_settings(), llm=vision_llm)
    chunk = Chunk(
        id="chunk_001",
        text="[IMAGE: img_001]\nFigure 3: Ablation results.",
        metadata={
            "source_path": "paper.pdf",
            "images": [{"id": "img_001", "path": str(image_path), "page": 1}],
        },
    )

    result = captioner.transform([chunk])[0]

    assert len(vision_llm.calls) == 3
    assert result.text.startswith("[IMAGE: img_001]")
    assert captioner.last_failures[0]["type"] == "image"
    assert captioner.last_failures[0]["image_id"] == "img_001"
    assert captioner.last_failures[0]["attempts"] == 3


def test_table_description_retries_before_success() -> None:
    text_llm = FakeTextLLM(failures_before_success=1)
    captioner = ImageCaptioner(_settings(), llm=FakeVisionLLM(), text_llm=text_llm)
    html_table = "<table><tr><th>Model</th><th>Score</th></tr><tr><td>A</td><td>92</td></tr></table>"
    chunk = Chunk(
        id="chunk_001",
        text=f"Table 2: Secondary retrieval results.\n\n{html_table}",
        metadata={"source_path": "paper.pdf"},
    )

    result = captioner.transform([chunk])[0]

    assert len(text_llm.calls) == 2
    assert html_table not in result.text
    assert captioner.last_failures == []
