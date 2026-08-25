"""Canonical DVBench JSONL loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

QUESTION_TYPES = frozenset({"EM", "MCQ_single", "MCQ_multiple", "Open_ended"})
DIMENSIONS = frozenset({"Narrative", "Animation", "Chart Perception", "Chart Reasoning", "Alignment"})
REQUIRED_FIELDS = ("question_id", "question_type", "video", "dimension", "question", "answer")
_PACKAGE_DATA_PATH = Path(__file__).resolve().parent / "data" / "DVBench_QA.jsonl"
_SOURCE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "DVBench_QA.jsonl"
DEFAULT_DATA_PATH = _PACKAGE_DATA_PATH if _PACKAGE_DATA_PATH.exists() else _SOURCE_DATA_PATH


class ValidationError(ValueError):
    """Raised when a canonical DVBench record is invalid."""


@dataclass(frozen=True)
class DVBenchRecord:
    question_id: str
    question_type: str
    video: str
    dimension: str
    question: str
    answer: str
    distractor1: str = ""
    distractor2: str = ""
    distractor3: str = ""
    chart_type: str = ""
    animation_editorial_layer: str = ""
    chart_reas_type: str = ""
    alignment_semantic_label: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, location: str = "record") -> "DVBenchRecord":
        missing = [key for key in REQUIRED_FIELDS if key not in value]
        if missing:
            raise ValidationError(f"{location}: missing required fields: {', '.join(missing)}")
        data = {field: value.get(field, "") for field in cls.__dataclass_fields__}
        for key, item in data.items():
            if item is None:
                item = ""
            if not isinstance(item, (str, int)):
                raise ValidationError(f"{location}: {key!r} must be a string or integer")
            data[key] = str(item).strip()
        if not data["question_id"] or not data["video"]:
            raise ValidationError(f"{location}: question_id and video must be non-empty")
        if data["question_type"] not in QUESTION_TYPES:
            raise ValidationError(f"{location}: unsupported question_type {data['question_type']!r}")
        if data["dimension"] not in DIMENSIONS:
            raise ValidationError(f"{location}: unsupported dimension {data['dimension']!r}")
        if (data["question_type"] == "Open_ended") != (data["dimension"] == "Alignment"):
            raise ValidationError(f"{location}: Open_ended and Alignment must occur together")
        if data["question_type"] == "MCQ_multiple" and len(split_answers(data["answer"])) < 2:
            raise ValidationError(f"{location}: MCQ_multiple requires at least two semicolon-separated answers")
        return cls(**data)

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def split_answers(answer: str) -> list[str]:
    """Split the canonical semicolon-separated multi-answer representation."""
    return [part.strip() for part in str(answer).split(";") if part.strip()]


def iter_jsonl(path: str | Path = DEFAULT_DATA_PATH) -> Iterator[DVBenchRecord]:
    source = Path(path)
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{source}:{line_number}: expected a JSON object")
            record = DVBenchRecord.from_mapping(value, location=f"{source}:{line_number}")
            if record.question_id in seen:
                raise ValidationError(f"{source}:{line_number}: duplicate question_id {record.question_id!r}")
            seen.add(record.question_id)
            yield record


def load_jsonl(path: str | Path = DEFAULT_DATA_PATH) -> list[DVBenchRecord]:
    return list(iter_jsonl(path))


def validate_records(records: Iterable[Mapping[str, Any] | DVBenchRecord]) -> list[DVBenchRecord]:
    validated, seen = [], set()
    for index, value in enumerate(records, 1):
        record = value if isinstance(value, DVBenchRecord) else DVBenchRecord.from_mapping(value, location=f"record {index}")
        if record.question_id in seen:
            raise ValidationError(f"record {index}: duplicate question_id {record.question_id!r}")
        seen.add(record.question_id)
        validated.append(record)
    return validated


def incomplete_records(records: Iterable[DVBenchRecord]) -> list[dict[str, str]]:
    """Return records with blank or explicitly unfinished release fields."""
    issues = []
    sentinels = {"to be completed", "to be comleted"}
    for record in records:
        missing = [
            field for field in ("question", "answer")
            if not getattr(record, field) or getattr(record, field).casefold() in sentinels
        ]
        if missing:
            issues.append({"question_id": record.question_id, "missing": ",".join(missing)})
    return issues


def file_sha256(path: str | Path = DEFAULT_DATA_PATH) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
