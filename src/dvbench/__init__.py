"""Public DVBench data, prompting, and evaluation API."""

from .data import DEFAULT_DATA_PATH, DVBenchRecord, ValidationError, incomplete_records, iter_jsonl, load_jsonl
from .evaluation import evaluate_predictions, extract_answer, normalize_exact_match, score_closed_ended
from .prompting import PromptResult, build_prompt, build_prompt_for_row

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DATA_PATH",
    "DVBenchRecord",
    "PromptResult",
    "ValidationError",
    "build_prompt",
    "build_prompt_for_row",
    "evaluate_predictions",
    "extract_answer",
    "incomplete_records",
    "iter_jsonl",
    "load_jsonl",
    "normalize_exact_match",
    "score_closed_ended",
]
