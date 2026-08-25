"""Paper-consistent DVBench scoring and structured evaluation outputs."""

from __future__ import annotations

import argparse
import json
import platform
import re
import string
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .data import DEFAULT_DATA_PATH, DVBenchRecord, file_sha256, incomplete_records, load_jsonl
from .prompting import build_prompt

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
LABEL_RE = re.compile(r"(?<![A-Za-z])[A-Z](?![A-Za-z])")


def extract_answer(response: Any) -> str:
    if response is None:
        return ""
    text = str(response).strip()
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else text


def normalize_exact_match(value: Any) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""
    text = str(value).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def parse_labels(value: Any, valid_labels: Iterable[str] = tuple(string.ascii_uppercase)) -> tuple[str, ...]:
    valid = set(valid_labels)
    labels = LABEL_RE.findall(extract_answer(value).upper())
    return tuple(dict.fromkeys(label for label in labels if label in valid))


def score_closed_ended(
    question_type: str,
    prediction: Any,
    answer: Any,
    *,
    correct_labels: Sequence[str] | None = None,
    valid_labels: Iterable[str] = tuple(string.ascii_uppercase),
) -> bool:
    """Score EM or MCQ; multiple choice uses exact complete-set match."""
    pred = extract_answer(prediction)
    if not pred or pred.upper().startswith("ERROR"):
        return False
    if question_type == "EM":
        return normalize_exact_match(pred) == normalize_exact_match(answer)
    if question_type not in {"MCQ_single", "MCQ_multiple"}:
        raise ValueError(f"not a closed-ended question type: {question_type!r}")
    if correct_labels is None:
        raise ValueError("correct_labels are required to score MCQ predictions")
    predicted = parse_labels(pred, valid_labels)
    expected = tuple(str(label).upper() for label in correct_labels)
    if question_type == "MCQ_single":
        return len(predicted) == 1 and predicted[0] == expected[0]
    return len(predicted) == len(expected) and set(predicted) == set(expected)


def _versions(names: Iterable[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def alignment_metrics(predictions: Sequence[str], references: Sequence[str], *, include_bertscore: bool = True) -> dict[str, Any]:
    """Compute paper metrics that are installed; unavailable metrics include reasons."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    available: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from nltk.translate.meteor_score import meteor_score

        tokenized = [
            (nltk.word_tokenize(str(p), preserve_line=True), nltk.word_tokenize(str(r), preserve_line=True))
            for p, r in zip(predictions, references)
        ]
        smooth = SmoothingFunction().method1
        available["bleu_2"] = 100 * fmean(
            sentence_bleu([ref], pred, weights=(0.5, 0.5), smoothing_function=smooth)
            for pred, ref in tokenized
        ) if tokenized else 0.0
        try:
            available["meteor"] = 100 * fmean(meteor_score([ref], pred) for pred, ref in tokenized) if tokenized else 0.0
        except LookupError as exc:
            unavailable["meteor"] = f"missing NLTK resource: {exc.args[0].splitlines()[0]}"
    except (ImportError, LookupError) as exc:
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        unavailable.update({"bleu_2": reason, "meteor": reason})

    if include_bertscore:
        try:
            from bert_score import score as bert_score

            if predictions:
                _, _, f1 = bert_score(list(predictions), list(references), lang="en", verbose=False)
                available["bertscore_f1"] = 100 * float(f1.mean())
            else:
                available["bertscore_f1"] = 0.0
        except Exception as exc:  # optional model/package/network dependency
            unavailable["bertscore_f1"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    else:
        unavailable["bertscore_f1"] = "disabled"
    return {"scores": {k: round(v, 6) for k, v in available.items()}, "unavailable": unavailable}


def evaluate_predictions(
    records: Sequence[DVBenchRecord],
    predictions: Mapping[str, Any],
    *,
    prompt_seed: int = 0,
    include_bertscore: bool = True,
    dataset_path: str | Path | None = None,
    model: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    incomplete = incomplete_records(records)
    if incomplete:
        summary = ", ".join(f"{item['question_id']} ({item['missing']})" for item in incomplete)
        raise ValueError(f"benchmark contains incomplete records that cannot be scored: {summary}")

    expected_ids = {record.question_id for record in records}
    provided_ids = set(predictions)
    missing_ids = sorted(expected_ids - provided_ids)
    unexpected_ids = sorted(provided_ids - expected_ids)
    if (missing_ids or unexpected_ids) and not allow_partial:
        details = []
        if missing_ids:
            details.append(f"{len(missing_ids)} missing prediction(s)")
        if unexpected_ids:
            details.append(f"{len(unexpected_ids)} unknown question ID(s)")
        raise ValueError("prediction coverage mismatch: " + ", ".join(details) + "; pass --allow-partial to score available rows")

    rows: list[dict[str, Any]] = []
    closed_by_dimension: dict[str, list[bool]] = defaultdict(list)
    alignment_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        raw = predictions.get(record.question_id, "")
        extracted = extract_answer(raw)
        result: dict[str, Any] = {
            "question_id": record.question_id,
            "question_type": record.question_type,
            "dimension": record.dimension,
            "prediction": extracted,
        }
        if record.question_type == "Open_ended":
            pair = (extracted, record.answer)
            alignment_groups["overall"].append(pair)
            if record.alignment_semantic_label:
                alignment_groups[record.alignment_semantic_label].append(pair)
            result["reference"] = record.answer
        else:
            prompt = build_prompt(record, seed=prompt_seed)
            correct = score_closed_ended(
                record.question_type, extracted, record.answer,
                correct_labels=prompt.correct_labels,
                valid_labels=prompt.label_to_option,
            )
            result.update({"is_correct": correct, "correct_labels": list(prompt.correct_labels)})
            closed_by_dimension[record.dimension].append(correct)
        rows.append(result)

    closed = {}
    all_values: list[bool] = []
    for dimension, values in sorted(closed_by_dimension.items()):
        all_values.extend(values)
        closed[dimension] = {"accuracy": round(100 * fmean(values), 6), "correct": sum(values), "total": len(values)}
    closed["overall_micro"] = {"accuracy": round(100 * fmean(all_values), 6), "correct": sum(all_values), "total": len(all_values)}
    closed["paper_average"] = round(fmean(value["accuracy"] for key, value in closed.items() if key != "overall_micro"), 6)

    alignment = {}
    for group, pairs in sorted(alignment_groups.items()):
        alignment[group] = {"total": len(pairs), **alignment_metrics([x[0] for x in pairs], [x[1] for x in pairs], include_bertscore=include_bertscore)}

    source = Path(dataset_path) if dataset_path else DEFAULT_DATA_PATH
    return {
        "schema_version": "1.0",
        "benchmark": "DVBench",
        "metrics": {"closed_ended": closed, "alignment": alignment},
        "coverage": {
            "expected": len(expected_ids),
            "provided": len(expected_ids & provided_ids),
            "missing": len(missing_ids),
            "unexpected": len(unexpected_ids),
            "missing_question_ids": missing_ids,
            "unexpected_question_ids": unexpected_ids,
            "is_complete": not missing_ids and not unexpected_ids,
        },
        "items": rows,
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(source.resolve()),
            "dataset_sha256": file_sha256(source) if source.exists() else None,
            "prompt_seed": prompt_seed,
            "model": model,
            "python": platform.python_version(),
            "packages": _versions(("dvbench", "nltk", "bert-score")),
        },
    }


def load_predictions(path: str | Path) -> dict[str, Any]:
    result = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if not question_id or question_id in result:
                raise ValueError(f"{path}:{number}: missing or duplicate question_id")
            if "prediction" not in row and "raw_response" not in row and "extracted_answer" not in row:
                raise ValueError(f"{path}:{number}: missing prediction/raw_response/extracted_answer")
            result[question_id] = row.get("prediction", row.get("raw_response", row.get("extracted_answer")))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DVBench JSONL predictions")
    parser.add_argument("predictions", help="JSONL with question_id and prediction")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default="dvbench_results.json")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Score incomplete prediction files and report coverage")
    args = parser.parse_args(argv)
    report = evaluate_predictions(
        load_jsonl(args.data),
        load_predictions(args.predictions),
        prompt_seed=args.prompt_seed,
        include_bertscore=not args.no_bertscore,
        dataset_path=args.data,
        model=args.model,
        allow_partial=args.allow_partial,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
