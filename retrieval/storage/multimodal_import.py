"""Import OCR and object-detection artifacts into PostgreSQL."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg

from retrieval.config import settings
from retrieval.provenance import dataset_group


def _value(row: object, name: str, default: object = None) -> object:
    value = getattr(row, name, default)
    return default if pd.isna(value) else value


def _validate_video(frame: pd.DataFrame, summary: dict[str, object], stage: str) -> str:
    if frame.empty:
        raise ValueError(f"{stage} artifact is empty")
    video_ids = {str(value) for value in frame["video_id"].dropna().unique()}
    if len(video_ids) != 1:
        raise ValueError(f"{stage} artifact must contain exactly one video: {sorted(video_ids)}")
    video_id = video_ids.pop()
    if str(summary.get("video_id")) != video_id:
        raise ValueError(f"{stage} summary video_id does not match parquet")
    return video_id


def _start_run(
    cur: psycopg.Cursor[object], run_id: str, run_type: str, row_count: int,
    group: str, artifact_batch: str | None,
) -> None:
    cur.execute(
        """
        INSERT INTO ingest_runs
          (run_id, run_type, status, video_count, keyframe_count, error_count, started_at,
           dataset_group, artifact_batch)
        VALUES (%s, %s, 'running', 1, %s, 0, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
          status='running', keyframe_count=EXCLUDED.keyframe_count,
          error_count=0, started_at=EXCLUDED.started_at, completed_at=NULL
        """,
        (run_id, run_type, row_count, datetime.now(UTC), group, artifact_batch),
    )


def _finish_run(cur: psycopg.Cursor[object], run_id: str) -> None:
    cur.execute(
        "UPDATE ingest_runs SET status='completed', completed_at=%s WHERE run_id=%s",
        (datetime.now(UTC), run_id),
    )


def _ensure_keyframes(cur: psycopg.Cursor[object], keyframe_ids: list[str], stage: str) -> None:
    cur.execute("SELECT keyframe_id FROM keyframes WHERE keyframe_id = ANY(%s)", (keyframe_ids,))
    present = {str(row[0]) for row in cur.fetchall()}
    missing = sorted(set(keyframe_ids) - present)
    if missing:
        raise ValueError(f"{stage} references unknown keyframes: {missing[:3]}")


def import_ocr(root: Path, artifact_batch: str | None = None) -> dict[str, int | str]:
    path = root / "ocr.parquet"
    summary_path = root / "ocr-summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise ValueError(f"Missing OCR artifact in {root}")
    frame = pd.read_parquet(path)
    required = {
        "keyframe_id",
        "video_id",
        "frame_idx",
        "pts_time",
        "text",
        "pipeline_version",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OCR artifact missing columns: {sorted(missing)}")
    if frame["keyframe_id"].duplicated().any():
        raise ValueError("Duplicate OCR keyframe_id")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_id = _validate_video(frame, summary, "OCR")
    group = dataset_group(video_id)
    pipeline_version = str(summary["pipeline_version"])
    run_id = f"ocr-{pipeline_version}-{video_id.lower()}"
    keyframe_ids = frame["keyframe_id"].astype(str).tolist()

    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        _ensure_keyframes(cur, keyframe_ids, "OCR")
        _start_run(cur, run_id, "ocr", len(frame), group, artifact_batch)
        cur.execute("DELETE FROM ocr_records WHERE video_id=%s", (video_id,))
        with cur.copy(
            """COPY ocr_records
            (keyframe_id, video_id, frame_idx, pts_time, shot_id, text, raw_text,
             normalized_text, search_text, shot_text, shot_search_text, ocr_error,
             pipeline_version, detections_json, merged_detections_json,
             dataset_group, artifact_batch) FROM STDIN"""
        ) as copy:
            for row in frame.itertuples(index=False):
                copy.write_row(
                    (
                        str(row.keyframe_id),
                        video_id,
                        int(row.frame_idx),
                        float(row.pts_time),
                        _value(row, "shot_id"),
                        str(_value(row, "text", "")),
                        _value(row, "raw_text"),
                        _value(row, "normalized_text"),
                        _value(row, "search_text"),
                        _value(row, "shot_text"),
                        _value(row, "shot_search_text"),
                        _value(row, "ocr_error"),
                        str(row.pipeline_version),
                        _value(row, "detections_json"),
                        _value(row, "merged_detections_json"),
                        group,
                        artifact_batch,
                    )
                )
        _finish_run(cur, run_id)
    return {"video_id": video_id, "records": len(frame), "pipeline": pipeline_version}


def import_objects(root: Path, artifact_batch: str | None = None) -> dict[str, int | str]:
    path = root / "objects.parquet"
    summary_path = root / "object-summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise ValueError(f"Missing object-detection artifact in {root}")
    frame = pd.read_parquet(path)
    required = {
        "detection_id",
        "keyframe_id",
        "video_id",
        "frame_idx",
        "pts_time",
        "model",
        "class_id",
        "class_name",
        "confidence",
        "x1_norm",
        "y1_norm",
        "x2_norm",
        "y2_norm",
        "bbox_area_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Object artifact missing columns: {sorted(missing)}")
    if frame["detection_id"].duplicated().any():
        raise ValueError("Duplicate object detection_id")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_id = _validate_video(frame, summary, "Object detection")
    group = dataset_group(video_id)
    pipeline_version = str(summary["pipeline_version"])
    run_id = f"object-{pipeline_version}-{video_id.lower()}"
    keyframe_ids = frame["keyframe_id"].astype(str).drop_duplicates().tolist()

    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        _ensure_keyframes(cur, keyframe_ids, "Object detection")
        _start_run(cur, run_id, "object", len(frame), group, artifact_batch)
        cur.execute("DELETE FROM object_detections WHERE video_id=%s", (video_id,))
        with cur.copy(
            """COPY object_detections
            (detection_id, keyframe_id, video_id, frame_idx, pts_time, model, class_id,
             class_name, confidence, x1_norm, y1_norm, x2_norm, y2_norm, bbox_area_ratio,
             dataset_group, artifact_batch)
             FROM STDIN"""
        ) as copy:
            for row in frame.itertuples(index=False):
                copy.write_row(
                    (
                        str(row.detection_id),
                        str(row.keyframe_id),
                        video_id,
                        int(row.frame_idx),
                        float(row.pts_time),
                        str(row.model),
                        int(row.class_id),
                        str(row.class_name),
                        float(row.confidence),
                        float(row.x1_norm),
                        float(row.y1_norm),
                        float(row.x2_norm),
                        float(row.y2_norm),
                        float(row.bbox_area_ratio),
                        group,
                        artifact_batch,
                    )
                )
        _finish_run(cur, run_id)
    return {"video_id": video_id, "detections": len(frame), "pipeline": pipeline_version}
