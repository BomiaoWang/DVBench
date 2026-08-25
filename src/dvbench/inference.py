"""Run DVBench inference and write evaluator-ready JSONL predictions."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dvbench.data import DEFAULT_DATA_PATH, DVBenchRecord, file_sha256, load_jsonl
from dvbench.prompting import build_prompt
from dvbench.videos import resolve_video_path


def resolve_local_video_path(video_value: object, videos_dir: Path) -> Path:
    """Backward-compatible alias for shared video path resolution."""
    return resolve_video_path(video_value, videos_dir)


def build_client(args: argparse.Namespace) -> Any:
    """Import provider SDK wrappers only when a real request is requested."""
    from dvbench.providers import Claude, Gemini, GPT, Kimi

    common = {"model": args.model_id, "temperature": args.temperature}
    if args.provider == "gemini":
        return Gemini(**common, top_p=args.top_p)
    if args.provider == "gpt":
        return GPT(**common, top_p=args.top_p)
    if args.provider == "claude":
        return Claude(**common, max_tokens=args.max_tokens)
    if args.provider == "kimi":
        return Kimi(
            **common,
            top_p=args.top_p,
            timeout=args.timeout,
            max_video_bytes=args.max_video_bytes,
        )
    raise ValueError(f"unsupported provider: {args.provider}")


def _mock_response(record: DVBenchRecord, correct_labels: tuple[str, ...]) -> str:
    value = ";".join(correct_labels) if record.question_type.startswith("MCQ") else record.answer
    return f"<answer>{value}</answer>"


def _call(client: Any, prompt: str, video_path: Path) -> str:
    return client.generate(prompt, video_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run API-backed models from the DVBench paper")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Canonical DVBench QA JSONL")
    parser.add_argument("--videos-dir", default="data/videos", help="Directory containing local MP4 files")
    parser.add_argument("--output", default="predictions.jsonl", help="Evaluator-ready JSONL output")
    parser.add_argument(
        "--provider",
        "--model",
        dest="provider",
        choices=["gemini", "gpt", "claude", "kimi"],
        default="kimi",
    )
    parser.add_argument("--model-id", help="Provider model ID; defaults to the paper configuration")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Claude output-token limit")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-video-bytes", type=int, help="Reject oversized Kimi inputs before base64 encoding")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Maximum selected questions; omitted means all")
    parser.add_argument("--question-id", help="Run one question ID")
    parser.add_argument("--mock", action="store_true", help="Generate local reference-shaped responses")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first item error")
    args = parser.parse_args(argv)

    data_path = Path(args.data).expanduser().resolve()
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    records = load_jsonl(data_path)
    if args.question_id:
        records = [record for record in records if record.question_id == args.question_id]
        if not records:
            parser.error(f"question ID not found: {args.question_id}")
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        records = records[:args.limit]

    if args.model_id is None:
        from dvbench.providers import MODEL_DEFAULTS
        args.model_id = MODEL_DEFAULTS[args.provider]

    client = None if args.mock else build_client(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": "mock" if args.mock else args.provider,
        "model": "mock" if args.mock else args.model_id,
        "temperature": args.temperature,
        "top_p": args.top_p if args.provider != "claude" else None,
        "input_mode": "mock" if args.mock else client.input_mode,
        "num_frames": None if args.mock else client.num_frames,
        "prompt_seed": args.prompt_seed,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "videos_dir": str(videos_dir),
        "python": platform.python_version(),
    }
    succeeded = failed = 0
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            started = time.monotonic()
            prompt_result = build_prompt(record, seed=args.prompt_seed)
            row: dict[str, Any] = {
                "schema_version": "1.0",
                "question_id": record.question_id,
                "video": record.video,
                "question_type": record.question_type,
                "prediction": "",
                "raw_response": "",
                "status": "error",
                "error": None,
                "correct_labels": list(prompt_result.correct_labels),
                "run": run,
            }
            try:
                if args.mock:
                    response = _mock_response(record, prompt_result.correct_labels)
                    video_path = None
                else:
                    video_path = resolve_video_path(record.video, videos_dir)
                    response = _call(client, prompt_result.prompt, video_path)
                row.update({"prediction": response, "raw_response": response, "status": "ok"})
                if video_path is not None:
                    row["video_path"] = str(video_path)
                succeeded += 1
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                failed += 1
                if args.fail_fast:
                    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    raise
            row["elapsed_seconds"] = round(time.monotonic() - started, 6)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    print(json.dumps({"output": str(output_path), "ok": succeeded, "error": failed, "run_id": run["run_id"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
