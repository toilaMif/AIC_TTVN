"""Lightweight availability check for YouTube links using the oEmbed endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


def load_items(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "file": path.name,
                    "video_id": data.get("video_id"),
                    "watch_url": data.get("watch_url"),
                    "title": data.get("title"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": path.name,
                    "video_id": None,
                    "watch_url": None,
                    "title": None,
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def check_one(row: dict[str, Any], timeout: int, attempts: int) -> dict[str, Any]:
    started = time.time()
    base = {
        "file": row["file"],
        "video_id": row.get("video_id"),
        "watch_url": row.get("watch_url"),
        "title": row.get("title"),
    }
    if row.get("load_error"):
        return {**base, "status": "metadata_error", "error": row["load_error"]}
    url = row.get("watch_url")
    if not url:
        return {**base, "status": "invalid_url", "error": "watch_url is empty"}

    endpoint = "https://www.youtube.com/oembed?url=" + quote(str(url), safe="") + "&format=json"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; media-link-check/1.0)"}
    last_error = ""
    last_status: int | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = requests.get(endpoint, headers=headers, timeout=timeout)
            last_status = response.status_code
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    return {
                        **base,
                        "status": "invalid_response",
                        "http_status": response.status_code,
                        "error": f"Invalid JSON: {exc}",
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                return {
                    **base,
                    "status": "ok",
                    "http_status": response.status_code,
                    "oembed_title": payload.get("title"),
                    "oembed_author": payload.get("author_name"),
                    "oembed_url": payload.get("url"),
                    "elapsed_sec": round(time.time() - started, 3),
                }
            # 429 and transient server failures may be caused by rate limiting.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = response.text[:300]
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (2**attempt))
                    continue
            body = response.text[:300].replace("\n", " ")
            return {
                **base,
                "status": "access_error" if response.status_code in {400, 401, 403, 404} else "http_error",
                "http_status": response.status_code,
                "error": body,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts:
                time.sleep(1.5 * (2**attempt))
                continue
    return {
        **base,
        "status": "network_error" if last_status is None else "rate_limited",
        "http_status": last_status,
        "error": last_error,
        "elapsed_sec": round(time.time() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    rows = load_items(args.root)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(check_one, row, args.timeout, args.attempts) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(futures):
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
