"""Idempotent import of audited BTC metadata into PostgreSQL."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg

from retrieval.config import settings
from retrieval.domain import keyframe_id


def import_metadata(manifest_path: Path) -> tuple[int, int, str]:
    manifests = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    run_id = f"metadata-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    keyframe_count = 0

    with (
        psycopg.connect(settings.database_url.replace("+psycopg", "")) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "INSERT INTO ingest_runs (run_id, run_type, status, video_count, keyframe_count, error_count, started_at) VALUES (%s, 'metadata', 'running', %s, 0, 0, %s)",
            (run_id, len(manifests), datetime.now(UTC)),
        )
        for item in manifests:
            metadata_path = Path(item["media_info_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cur.execute(
                """
                    INSERT INTO videos (video_id, youtube_id, watch_url, title, description,
                                        duration_expected_sec, availability_status, source_metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'unchecked', %s::jsonb, %s)
                    ON CONFLICT (video_id) DO UPDATE SET
                      youtube_id=EXCLUDED.youtube_id, watch_url=EXCLUDED.watch_url,
                      title=EXCLUDED.title, description=EXCLUDED.description,
                      duration_expected_sec=EXCLUDED.duration_expected_sec,
                      source_metadata=EXCLUDED.source_metadata
                    """,
                (
                    item["video_id"],
                    item["youtube_id"],
                    item["watch_url"],
                    metadata.get("title"),
                    metadata.get("description"),
                    metadata.get("length"),
                    json.dumps(metadata, ensure_ascii=False),
                    datetime.now(UTC),
                ),
            )

            with Path(item["mapping_path"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            keyframe_count += len(rows)
            cur.execute("DELETE FROM keyframes WHERE video_id=%s", (item["video_id"],))
            with cur.copy(
                "COPY keyframes (keyframe_id, video_id, n, frame_idx, pts_time, fps) FROM STDIN"
            ) as copy:
                for row in rows:
                    copy.write_row(
                        (
                            keyframe_id(item["video_id"], int(row["n"])),
                            item["video_id"],
                            int(row["n"]),
                            int(row["frame_idx"]),
                            float(row["pts_time"]),
                            float(row["fps"]),
                        )
                    )
        cur.execute(
            "UPDATE ingest_runs SET status='completed', keyframe_count=%s, completed_at=%s WHERE run_id=%s",
            (keyframe_count, datetime.now(UTC), run_id),
        )
    return len(manifests), keyframe_count, run_id
