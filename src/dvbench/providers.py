"""API wrappers for the proprietary models evaluated in DVBench."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from .videos import sample_frames

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GPT_MODEL = "gpt-5.4"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_KIMI_MODEL = "kimi-k2.5"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"

MODEL_DEFAULTS = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "gpt": DEFAULT_GPT_MODEL,
    "claude": DEFAULT_CLAUDE_MODEL,
    "kimi": DEFAULT_KIMI_MODEL,
}

__all__ = ["Claude", "Gemini", "GPT", "Kimi", "MODEL_DEFAULTS"]


def _normalize_path_key(file_path: str | Path) -> str:
    return str(Path(file_path).expanduser().resolve())


def _read_bytes(path: str | Path, max_bytes: int | None = None) -> bytes:
    source = Path(path).expanduser().resolve()
    if max_bytes is not None and source.stat().st_size > max_bytes:
        raise ValueError(f"file exceeds configured limit of {max_bytes} bytes: {source}")
    return source.read_bytes()


def _data_url(path: str | Path, data: bytes, media_type: str | None = None) -> str:
    suffix = Path(path).suffix.lstrip(".").lower() or "mp4"
    mime = media_type or f"video/{suffix}"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _sampled_frame_data(video_path: str | Path, num_frames: int) -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory(prefix="dvbench-frames-") as directory:
        frames = sample_frames(video_path, directory, num_frames=num_frames)
        return [
            ("image/jpeg", base64.b64encode(frame.path.read_bytes()).decode("ascii"))
            for frame in frames
        ]


class Gemini:
    """Gemini 3.1 Pro with native video upload."""

    input_mode = "video"
    num_frames = None

    def __init__(
        self,
        uploaded_files_map: dict[str, str] | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.0,
        top_p: float = 1.0,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("install google-genai to use Gemini") from exc
            client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.client = client
        self.uploaded_files_map = {
            _normalize_path_key(key): value for key, value in (uploaded_files_map or {}).items()
        }
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str, video_path: str | Path) -> str:
        normalized_key = _normalize_path_key(video_path)
        if normalized_key in self.uploaded_files_map:
            file_obj = self.client.files.get(name=self.uploaded_files_map[normalized_key])
        else:
            file_obj = self.client.files.upload(file=normalized_key)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[file_obj, prompt],
            config={"temperature": self.temperature, "top_p": self.top_p},
        )
        return str(response.text or "")


class GPT:
    """GPT-5.4 with 64 uniformly sampled frames, matching the paper."""

    input_mode = "uniform_frames"
    num_frames = 64

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GPT_MODEL,
        temperature: float = 0.0,
        top_p: float = 1.0,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("install openai to use GPT") from exc
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str, video_path: str | Path) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
            for media_type, data in _sampled_frame_data(video_path, self.num_frames)
        )
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return str(response.output_text or "")


class Claude:
    """Claude Sonnet 4.6 with 100 uniformly sampled frames, matching the paper."""

    input_mode = "uniform_frames"
    num_frames = 100

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError("install anthropic to use Claude") from exc
            client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, video_path: str | Path) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
            for media_type, data in _sampled_frame_data(video_path, self.num_frames)
        ]
        content.append({"type": "text", "text": prompt})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            str(block.text) for block in response.content
            if getattr(block, "type", None) == "text"
        )


class Kimi:
    """Kimi K2.5 using Moonshot's experimental ``video_url`` extension."""

    input_mode = "video"
    num_frames = None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_KIMI_MODEL,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 180.0,
        max_video_bytes: int | None = None,
        client: Any | None = None,
    ):
        key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = base_url or os.environ.get("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("install openai to use Kimi") from exc
            client = OpenAI(api_key=key.strip() if key else key, base_url=base)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_video_bytes = max_video_bytes

    def generate(self, prompt: str, video_path: str | Path) -> str:
        video_data = _read_bytes(video_path, self.max_video_bytes)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 Kimi。"},
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": _data_url(video_path, video_data)}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
        )
        try:
            return str(completion.choices[0].message.content or "")
        except (AttributeError, IndexError, TypeError):
            return str(getattr(completion, "text", completion))
