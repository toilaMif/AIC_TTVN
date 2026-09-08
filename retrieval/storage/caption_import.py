"""Import frame-level captions and search text from the V2 fusion pipeline."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg

from retrieval.config import settings
from retrieval.provenance import dataset_group
from retrieval.search.caption_semantic import embed_and_upsert_captions


def _value(row: object, name: str, default: object = None) -> object:
    value = getattr(row, name, default)
    return default if pd.isna(value) else value


def import_captions(root: Path, artifact_batch: str | None = None) -> dict[str, int | str]:
    path = root / "search-documents.parquet"
    summary_path = root / "frame-understanding-summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise ValueError(f"Missing frame-understanding artifact in {root}")
    frame = pd.read_parquet(path)
    required = {
        "keyframe_id",
        "video_id",
        "frame_idx",
        "pts_time",
        "search_text_vi",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame-understanding artifact missing columns: {sorted(missing)}")
    if frame["keyframe_id"].duplicated().any():
        raise ValueError("Duplicate caption keyframe_id")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_ids = {str(value) for value in frame["video_id"].dropna().unique()}
    if len(video_ids) != 1:
        raise ValueError(f"Frame-understanding artifact must contain exactly one video: {sorted(video_ids)}")
    video_id = video_ids.pop()
    if str(summary.get("video_id")) != video_id:
        raise ValueError("Frame-understanding summary video_id does not match parquet")
    group = dataset_group(video_id)
    pipeline_version = str(summary["pipeline_version"])
    run_id = f"caption-{pipeline_version}-{video_id.lower()}"
    keyframe_ids = frame["keyframe_id"].astype(str).tolist()

    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SELECT keyframe_id FROM keyframes WHERE keyframe_id = ANY(%s)", (keyframe_ids,))
        present = {str(row[0]) for row in cur.fetchall()}
        missing_keyframes = sorted(set(keyframe_ids) - present)
        if missing_keyframes:
            raise ValueError(f"Frame-understanding references unknown keyframes: {missing_keyframes[:3]}")
        cur.execute(
            """
            INSERT INTO ingest_runs
              (run_id, run_type, status, video_count, keyframe_count, error_count, started_at,
               dataset_group, artifact_batch)
            VALUES (%s, 'caption', 'running', 1, %s, 0, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
              status='running', keyframe_count=EXCLUDED.keyframe_count,
              error_count=0, started_at=EXCLUDED.started_at, completed_at=NULL
            """,
            (run_id, len(frame), datetime.now(UTC), group, artifact_batch),
        )
        cur.execute("DELETE FROM frame_captions WHERE video_id=%s", (video_id,))
        with cur.copy(
            """COPY frame_captions
            (keyframe_id, video_id, shot_id, frame_idx, pts_time, caption_vi_short,
             caption_vi_detail, search_text_vi, asr_text, ocr_scene_text, object_labels_text,
             scene_labels_json, description_source, confidence, pipeline_version,
             dataset_group, artifact_batch) FROM STDIN"""
        ) as copy:
            for row in frame.itertuples(index=False):
                copy.write_row(
                    (
                        str(row.keyframe_id),
                        video_id,
                        _value(row, "shot_id"),
                        int(row.frame_idx),
                        float(row.pts_time),
                        _value(row, "caption_vi_short"),
                        _value(row, "caption_vi_detail"),
                        str(_value(row, "search_text_vi", "")),
                        _value(row, "asr_text"),
                        _value(row, "ocr_scene_text"),
                        _value(row, "object_labels_text"),
                        _value(row, "scene_labels_json"),
                        _value(row, "description_source"),
                        _value(row, "confidence"),
                        pipeline_version,
                        group,
                        artifact_batch,
                    )
                )
        cur.execute(
            "UPDATE ingest_runs SET status='completed', completed_at=%s WHERE run_id=%s",
            (datetime.now(UTC), run_id),
        )
    embedded = embed_and_upsert_captions(frame, video_id)
    return {
        "video_id": video_id,
        "records": len(frame),
        "pipeline": pipeline_version,
        "embedded": embedded,
    }
