"""Check YouTube links in a media-info directory without downloading videos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL


def load_items(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "file": path.name,
                    "video_id": data.get("video_id"),
                    "watch_url": data.get("watch_url"),
                    "title": data.get("title"),
                }
            )
        except Exception as exc:  # keep malformed files visible in the report
            items.append(
                {
                    "file": path.name,
                    "video_id": None,
                    "watch_url": None,
                    "title": None,
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return items


def check_one(item: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    started = time.time()
    base = {
        "file": item["file"],
        "video_id": item.get("video_id"),
        "watch_url": item.get("watch_url"),
        "title": item.get("title"),
    }
    if item.get("load_error"):
        return {
            **base,
            "status": "metadata_error",
            "error_type": "MetadataError",
            "error": item["load_error"],
            "elapsed_sec": round(time.time() - started, 3),
        }
    if not item.get("watch_url"):
        return {
            **base,
            "status": "invalid_url",
            "error_type": "MissingURL",
            "error": "watch_url is empty",
            "elapsed_sec": round(time.time() - started, 3),
        }

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "retries": retries,
        "extractor_retries": retries,
        "file_access_retries": retries,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(str(item["watch_url"]), download=False)
        extracted_id = info.get("id")
        return {
            **base,
            "status": "ok",
            "extracted_id": extracted_id,
            "availability": info.get("availability"),
            "live_status": info.get("live_status"),
            "duration_sec": info.get("duration"),
            "resolved_url": info.get("webpage_url") or info.get("original_url"),
            "extracted_title": info.get("title"),
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:  # yt-dlp exposes useful access details in the message
        return {
            **base,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    items = load_items(args.root)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(check_one, item, args.timeout, args.retries) for item in items]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or index == len(futures):
                print(f"checked {index}/{len(futures)}", file=sys.stderr, flush=True)

    results.sort(key=lambda row: row["file"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    summary: dict[str, int] = {}
    for row in results:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    print(json.dumps({"total": len(results), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
