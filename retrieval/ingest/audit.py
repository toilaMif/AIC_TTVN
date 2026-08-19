"""Audit the two BTC source directories before any video download."""

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REQUIRED_COLUMNS = ("n", "pts_time", "fps", "frame_idx")
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    video_id: str | None = None
    row: int | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def youtube_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        return None
    value = (
        parsed.path.strip("/")
        if parsed.netloc.lower() == "youtu.be"
        else parse_qs(parsed.query).get("v", [""])[0]
    )
    return value if YOUTUBE_ID.fullmatch(value) else None


def audit(
    mapping_root: Path, media_info_root: Path
) -> tuple[list[dict[str, object]], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    metadata_paths = {path.stem: path for path in media_info_root.glob("*.json")}
    mapping_paths = {path.stem: path for path in mapping_root.glob("*.csv")}
    manifests: list[dict[str, object]] = []

    for video_id in sorted(metadata_paths.keys() | mapping_paths.keys()):
        metadata_path = metadata_paths.get(video_id)
        mapping_path = mapping_paths.get(video_id)
        if not metadata_path or not mapping_path:
            issues.append(AuditIssue("error", "MISSING_PAIR", "JSON/CSV pair is missing", video_id))
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(AuditIssue("error", "INVALID_JSON", str(exc), video_id))
            continue
        yt_id = youtube_id(str(metadata.get("watch_url", "")))
        if not yt_id:
            issues.append(
                AuditIssue(
                    "error", "INVALID_WATCH_URL", "watch_url is not a valid YouTube URL", video_id
                )
            )

        with mapping_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                issues.append(
                    AuditIssue(
                        "error", "INVALID_COLUMNS", "CSV columns must match contract", video_id
                    )
                )
                continue
            rows = list(reader)

        previous_n = previous_frame = -1
        previous_time = -1.0
        for row_number, row in enumerate(rows, start=2):
            try:
                ordinal = int(row["n"])
                frame = int(row["frame_idx"])
                pts_time = float(row["pts_time"])
                fps = float(row["fps"])
            except ValueError:
                issues.append(
                    AuditIssue(
                        "error", "INVALID_VALUE", "mapping row is not numeric", video_id, row_number
                    )
                )
                continue
            if (
                ordinal <= previous_n
                or frame < previous_frame
                or pts_time < previous_time
                or frame < 0
                or pts_time < 0
                or fps <= 0
            ):
                issues.append(
                    AuditIssue(
                        "error",
                        "INVALID_MONOTONICITY",
                        "mapping order or value is invalid",
                        video_id,
                        row_number,
                    )
                )
            elif frame == previous_frame:
                issues.append(
                    AuditIssue(
                        "warning",
                        "DUPLICATE_FRAME",
                        "consecutive samples map to the same source frame",
                        video_id,
                        row_number,
                    )
                )
            previous_n, previous_frame, previous_time = ordinal, frame, pts_time

        manifests.append(
            {
                "video_id": video_id,
                "youtube_id": yt_id,
                "watch_url": metadata.get("watch_url"),
                "title": metadata.get("title"),
                "duration_expected_sec": metadata.get("length"),
                "mapping_path": str(mapping_path).replace("\\", "/"),
                "media_info_path": str(metadata_path).replace("\\", "/"),
                "keyframe_count": len(rows),
                "mapping_sha256": sha256(mapping_path),
                "metadata_sha256": sha256(metadata_path),
                "validation_status": "valid" if yt_id else "invalid",
            }
        )
    return manifests, issues


def write_report(
    manifests: list[dict[str, object]], issues: list[AuditIssue], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifests), encoding="utf-8"
    )
    report = {
        "summary": {
            "valid_videos": sum(item["validation_status"] == "valid" for item in manifests),
            "invalid_videos": sum(item["validation_status"] != "valid" for item in manifests),
            "keyframes": sum(int(item["keyframe_count"]) for item in manifests),
            "errors": sum(issue.severity == "error" for issue in issues),
            "warnings": sum(issue.severity == "warning" for issue in issues),
        },
        "issues": [asdict(issue) for issue in issues],
    }
    (output_dir / "audit-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
