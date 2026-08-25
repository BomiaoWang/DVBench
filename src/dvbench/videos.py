"""Local video path resolution, probing, and deterministic frame sampling."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_seconds: float
    width: int | None
    height: int | None


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    timestamp_seconds: float


def _video_stem(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("video value is empty")
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    name = Path(raw).name
    if name != raw or raw in {".", ".."}:
        raise ValueError(f"video value must be a file name or ID, got {raw!r}")
    return name[:-4] if name.lower().endswith(".mp4") else name


def resolve_video_path(video: object, videos_dir: str | Path) -> Path:
    """Resolve numeric IDs and padded/unpadded MP4 names within ``videos_dir``."""
    directory = Path(videos_dir).expanduser().resolve()
    stem = _video_stem(video)
    names = [f"{stem}.mp4"]
    if stem.isdigit():
        names.extend((f"{int(stem)}.mp4", f"{stem.zfill(3)}.mp4"))
    for name in dict.fromkeys(names):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"video not found in {directory}: tried {list(dict.fromkeys(names))}")


def _executable(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"{name} is required but was not found on PATH")
    return value


def probe_video(path: str | Path, *, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Read basic stream metadata with ffprobe."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [
        _executable(ffprobe), "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height", "-of", "json", str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload: dict[str, Any] = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration") or 0)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), {})
    if duration <= 0 or not stream:
        raise ValueError(f"no usable video stream or duration in {source}")
    return VideoMetadata(source, duration, stream.get("width"), stream.get("height"))


def uniform_timestamps(duration_seconds: float, num_frames: int) -> list[float]:
    """Return deterministic midpoint timestamps, avoiding fragile end-of-file seeks."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return [duration_seconds * (index + 0.5) / num_frames for index in range(num_frames)]


def sample_frames(
    path: str | Path,
    output_dir: str | Path,
    *,
    num_frames: int = 8,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
) -> list[SampledFrame]:
    """Extract uniformly sampled JPEG frames with ffmpeg and return timestamps."""
    metadata = probe_video(path, ffprobe=ffprobe)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    executable = _executable(ffmpeg)
    sampled = []
    for index, timestamp in enumerate(uniform_timestamps(metadata.duration_seconds, num_frames)):
        target = destination / f"frame_{index:04d}_{timestamp:.3f}s.jpg"
        if target.exists() and not overwrite:
            sampled.append(SampledFrame(target, timestamp))
            continue
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
            "-ss", f"{timestamp:.6f}", "-i", str(metadata.path), "-frames:v", "1",
            "-q:v", "2", str(target),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg did not produce frame at {timestamp:.3f}s")
        sampled.append(SampledFrame(target, timestamp))
    return sampled
