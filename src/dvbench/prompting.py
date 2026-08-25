"""Deterministic prompt construction with explicit option-label provenance."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from .data import DVBenchRecord, split_answers

EM_PROMPT = """Based on the video, please answer the following question:
{question}
Please provide the answer directly in <answer></answer> without any units, symbols, or explanation. For example, if the answer is 15% or $1, output only <answer>15</answer> or <answer>1</answer>.
Answer:
"""
MCQ_SINGLE_PROMPT = """Based on the video, please answer the following question:
{question}
{option_lines}Output only the answer label (e.g., A, B, C, or D) between <answer> and </answer> tags (e.g., <answer>A</answer>).
Answer:
"""
MCQ_MULTIPLE_PROMPT = """Based on the video, please answer the following question:
{question}
{option_lines}This question has more than one correct answer. Output all correct answer labels separated by semicolons between <answer> and </answer> tags (e.g., <answer>A;B;C</answer>).
Answer:
"""
ALIGNMENT_PROMPT = """Please analyze the video and the surrounding subtitle context to identify the missing segment:
Subtitle Context:
{subtitle}

Based on the video content and the subtitle context, please fill in the missing subtitle segment for [Insert Subtitle Here], and output the filled content directly between the <answer> and </answer> tags.
Answer:
"""


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    question_id: str
    question_type: str
    seed: int
    label_to_option: dict[str, str] = field(default_factory=dict)
    correct_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "seed": self.seed,
            "label_to_option": dict(self.label_to_option),
            "correct_labels": list(self.correct_labels),
        }


def _record(value: Mapping[str, Any] | DVBenchRecord) -> DVBenchRecord:
    return value if isinstance(value, DVBenchRecord) else DVBenchRecord.from_mapping(value)


def _item_seed(question_id: str, seed: int) -> int:
    payload = f"dvbench-prompt-v1\0{seed}\0{question_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_prompt(value: Mapping[str, Any] | DVBenchRecord, *, seed: int = 0, shuffle: bool = True) -> PromptResult:
    record = _record(value)
    if record.question_type == "Open_ended":
        return PromptResult(ALIGNMENT_PROMPT.format(subtitle=record.question), record.question_id, record.question_type, seed)
    if record.question_type == "EM":
        return PromptResult(EM_PROMPT.format(question=record.question), record.question_id, record.question_type, seed)

    correct_options = split_answers(record.answer) if record.question_type == "MCQ_multiple" else [record.answer]
    options = correct_options + [x for x in (record.distractor1, record.distractor2, record.distractor3) if x]
    if len(set(options)) != len(options):
        raise ValueError(f"{record.question_id}: duplicate MCQ option text is ambiguous")
    if len(options) > 26:
        raise ValueError(f"{record.question_id}: at most 26 options are supported")
    if shuffle:
        random.Random(_item_seed(record.question_id, seed)).shuffle(options)
    mapping = {chr(65 + index): option for index, option in enumerate(options)}
    correct = tuple(label for label, option in mapping.items() if option in set(correct_options))
    lines = "".join(f"{label}. {option}\n" for label, option in mapping.items())
    template = MCQ_SINGLE_PROMPT if record.question_type == "MCQ_single" else MCQ_MULTIPLE_PROMPT
    return PromptResult(template.format(question=record.question, option_lines=lines), record.question_id, record.question_type, seed, mapping, correct)


def build_prompt_for_row(row: Mapping[str, Any], *, seed: int = 0, shuffle: bool = True) -> str:
    """Compatibility helper returning only prompt text."""
    return build_prompt(row, seed=seed, shuffle=shuffle).prompt
