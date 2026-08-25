"""Configurable Hugging Face downloader for DVBench video assets."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Sequence

DEFAULT_FILENAME = "videos.zip"
DEFAULT_LOCAL_DIR = Path("data")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename!r}")
        handle.extractall(destination)


def download_videos(
    repo_id: str,
    *,
    filename: str = DEFAULT_FILENAME,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    revision: str | None = None,
    token: str | None = None,
    extract: bool = True,
    force_download: bool = False,
) -> Path:
    """Download a configured dataset asset and optionally extract a ZIP archive."""
    if not repo_id or repo_id.startswith("<"):
        raise ValueError("set a real Hugging Face dataset repo ID; no public default is assumed")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to download video assets") from exc

    destination = Path(local_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
        local_dir=str(destination),
        force_download=force_download,
    )).resolve()
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise RuntimeError(f"downloaded asset is missing or empty: {downloaded}")
    if extract:
        if not zipfile.is_zipfile(downloaded):
            raise ValueError(f"--extract requires a ZIP asset: {downloaded}")
        _safe_extract(downloaded, destination)
    return downloaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download DVBench videos from a configured Hugging Face dataset")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("DVBENCH_HF_REPO_ID"),
        help="Hugging Face dataset repo ID (or set DVBENCH_HF_REPO_ID); intentionally has no guessed default",
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Asset path within the dataset repo")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Download and extraction directory")
    parser.add_argument("--revision", help="Branch, tag, or commit to pin")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token; defaults to HF_TOKEN")
    parser.add_argument("--no-extract", action="store_true", help="Keep the downloaded archive without extraction")
    parser.add_argument("--force", action="store_true", help="Force a fresh download")
    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("--repo-id is required (or set DVBENCH_HF_REPO_ID)")
    path = download_videos(
        args.repo_id, filename=args.filename, local_dir=args.local_dir,
        revision=args.revision, token=args.token, extract=not args.no_extract,
        force_download=args.force,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
