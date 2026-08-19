"""Select and download a reproducible local video demo set."""

import hashlib
import json
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_demo(source: Path, destination: Path, limit: int = 10) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = [
        item
        for item in read_jsonl(source)
        if item.get("validation_status") == "valid" and item.get("watch_url")
    ][:limit]
    if len(selected) < limit:
        raise ValueError(f"requested {limit} videos, but only {len(selected)} are valid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    return selected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_videos(
    manifest_path: Path,
    output_root: Path,
    max_height: int = 720,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for item in read_jsonl(manifest_path):
        video_id = str(item["video_id"])
        video_dir = output_root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = video_dir / "download.json"
        existing = list(video_dir.glob("source.*"))
        if metadata_path.exists() and existing:
            results.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue

        options: dict[str, Any] = {
            "format": (f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b"),
            "merge_output_format": "mp4",
            "outtmpl": str(video_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(str(item["watch_url"]), download=True)
            prepared = Path(ydl.prepare_filename(info))

        candidates = list(video_dir.glob("source.*"))
        video_path = next((path for path in candidates if path.suffix == ".mp4"), prepared)
        if not video_path.exists():
            raise FileNotFoundError(f"download completed but output is missing for {video_id}")
        result = {
            "video_id": video_id,
            "youtube_id": item.get("youtube_id"),
            "source_url": item["watch_url"],
            "path": video_path.as_posix(),
            "bytes": video_path.stat().st_size,
            "sha256": file_sha256(video_path),
            "duration_sec": info.get("duration"),
            "width": info.get("width"),
            "height": info.get("height"),
            "extractor": info.get("extractor_key"),
            "status": "downloaded",
        }
        metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
    return results
