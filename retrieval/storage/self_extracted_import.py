"""Validated, idempotent import of Kaggle self-extracted artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from minio import Minio

from retrieval.config import settings
from retrieval.indexes.milvus import stable_milvus_pk, upsert_visual_vectors
from retrieval.provenance import dataset_group

FRAME_BUCKET = "aic-frames"
ARTIFACT_BUCKET = "aic-artifacts"


def validate_output(root: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, np.ndarray]:
    required = [
        "summary.json",
        "frame-mapping-report.json",
        "shots.parquet",
        "keyframes.parquet",
        "embedding-records.parquet",
        "visual.npy",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing output artifacts: {missing}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "frame-mapping-report.json").read_text(encoding="utf-8"))
    shots = pd.read_parquet(root / "shots.parquet")
    keyframes = pd.read_parquet(root / "keyframes.parquet")
    records = pd.read_parquet(root / "embedding-records.parquet")
    vectors = np.load(root / "visual.npy", allow_pickle=False)
    if not summary.get("quality_report", {}).get("passed"):
        raise ValueError("quality_report.passed is not true")
    # Batch Kaggle exports may omit optional BTC checkpoint rows; the keyframe
    # parquet itself remains authoritative when the report is empty.
    if mapping and not all(row.get("passed") for row in mapping):
        raise ValueError("frame mapping validation failed")
    if len(keyframes) != len(records) or len(keyframes) != len(vectors):
        raise ValueError("keyframe/embedding counts differ")
    if keyframes["keyframe_id"].tolist() != records["keyframe_id"].tolist():
        raise ValueError("embedding records are not aligned with keyframes")
    if records["embedding_row"].tolist() != list(range(len(records))):
        raise ValueError("embedding_row is not contiguous")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or Inf")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("vectors are not L2 normalized")
    if not keyframes["keyframe_id"].is_unique or not shots["shot_id"].is_unique:
        raise ValueError("duplicate shot/keyframe IDs")
    if not shots["shot_id"].isin(keyframes["shot_id"]).all():
        raise ValueError("one or more shots have no keyframe")
    absent = [path for path in keyframes["frame_path"] if not (root / path).is_file()]
    if absent:
        raise ValueError(f"missing frame files: {absent[:3]}")
    return summary, shots, keyframes, vectors.astype("float32", copy=False)


def _minio_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=False,
    )


def import_self_extracted(
    root: Path, collection_name: str, artifact_batch: str | None = None
) -> dict[str, object]:
    summary, shots, keyframes, vectors = validate_output(root)
    video_id = str(summary["video_id"])
    group = dataset_group(video_id)
    version = str(summary["pipeline_version"])
    run_id = str(summary["run_id"])
    embedding_version = f"{summary['embedding_model']}:{version}:{video_id}"
    job_id = f"visual-{run_id}"

    minio = _minio_client()
    for bucket in (FRAME_BUCKET, ARTIFACT_BUCKET):
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
    frame_keys: dict[str, str] = {}
    for row in keyframes.itertuples(index=False):
        source = root / row.frame_path
        object_key = f"self-extracted/{version}/{video_id}/frames/{source.name}"
        minio.fput_object(FRAME_BUCKET, object_key, str(source), content_type="image/webp")
        frame_keys[row.keyframe_id] = object_key
    for name in (
        "summary.json",
        "frame-mapping-report.json",
        "shots.parquet",
        "keyframes.parquet",
        "embedding-records.parquet",
        "visual.npy",
    ):
        minio.fput_object(
            ARTIFACT_BUCKET,
            f"self-extracted/{version}/{video_id}/{name}",
            str(root / name),
        )

    connection_url = settings.database_url.replace("+psycopg", "")
    now = datetime.now(UTC)
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM videos WHERE video_id=%s", (video_id,))
        if cur.fetchone() is None:
            raise ValueError(f"Video {video_id} is not present; run import-metadata first")
        cur.execute(
            "UPDATE videos SET dataset_group=%s, artifact_batch=%s WHERE video_id=%s",
            (group, artifact_batch, video_id),
        )
        cur.execute(
            """
            INSERT INTO ingest_runs
              (run_id, run_type, status, video_count, keyframe_count, error_count, started_at,
               dataset_group, artifact_batch)
            VALUES (%s, 'self_extracted', 'running', 1, %s, 0, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET status='running', error_count=0
            """,
            (run_id, len(keyframes), now, group, artifact_batch),
        )
        cur.execute("DELETE FROM feature_records WHERE job_id=%s", (job_id,))
        cur.execute("DELETE FROM feature_jobs WHERE job_id=%s", (job_id,))
        cur.execute(
            "DELETE FROM keyframes WHERE video_id=%s AND pipeline_version=%s", (video_id, version)
        )
        cur.execute(
            "DELETE FROM shots WHERE video_id=%s AND pipeline_version=%s", (video_id, version)
        )
        with cur.copy(
            """COPY shots (shot_id, video_id, shot_index, start_frame, end_frame,
            start_time, end_time, duration_sec, detector, pipeline_version, run_id) FROM STDIN"""
        ) as copy:
            for row in shots.itertuples(index=False):
                copy.write_row(
                    (
                        row.shot_id,
                        video_id,
                        int(row.shot_index),
                        int(row.start_frame),
                        int(row.end_frame),
                        float(row.start_time),
                        float(row.end_time),
                        float(row.duration_sec),
                        row.detector,
                        version,
                        run_id,
                    )
                )
        with cur.copy(
            """COPY keyframes (keyframe_id, video_id, n, frame_idx, pts_time, fps, shot_id,
            window_id, frame_object_key, source, pipeline_version, run_id, quality_score,
            quality_fallback, dataset_group, artifact_batch) FROM STDIN"""
        ) as copy:
            for row in keyframes.itertuples(index=False):
                copy.write_row(
                    (
                        row.keyframe_id,
                        video_id,
                        int(row.ordinal),
                        int(row.frame_idx),
                        float(row.pts_time),
                        float(row.fps),
                        row.shot_id,
                        str(row.window_id),
                        frame_keys[row.keyframe_id],
                        "self_extracted",
                        version,
                        run_id,
                        float(row.quality_score),
                        bool(row.quality_fallback),
                        group,
                        artifact_batch,
                    )
                )
        cur.execute(
            """INSERT INTO feature_jobs (job_id, model_name, model_version, embedding_version,
            dimension, dtype, expected_count, completed_count, status, dataset_group, artifact_batch)
            VALUES (%s, %s, %s, %s, %s, 'float32', %s, %s, 'completed', %s, %s)""",
            (
                job_id,
                str(summary["embedding_model"]),
                version,
                embedding_version,
                int(vectors.shape[1]),
                len(vectors),
                group,
                artifact_batch,
                len(vectors),
            ),
        )
        with cur.copy(
            """COPY feature_records (job_id, keyframe_id, embedding_version,
            milvus_collection, milvus_pk, import_status) FROM STDIN"""
        ) as copy:
            for row in keyframes.itertuples(index=False):
                copy.write_row(
                    (
                        job_id,
                        row.keyframe_id,
                        embedding_version,
                        collection_name,
                        stable_milvus_pk(row.keyframe_id),
                        "pending",
                    )
                )

    milvus_records = [
        {
            "pk": stable_milvus_pk(row.keyframe_id),
            "keyframe_id": row.keyframe_id,
            "video_id": video_id,
            "frame_idx": int(row.frame_idx),
            "embedding": vectors[index].tolist(),
        }
        for index, row in enumerate(keyframes.itertuples(index=False))
    ]
    upsert_visual_vectors(settings.milvus_uri, collection_name, milvus_records, vectors.shape[1])
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE feature_records SET import_status='imported' WHERE job_id=%s", (job_id,)
        )
        cur.execute(
            "UPDATE ingest_runs SET status='completed', completed_at=%s WHERE run_id=%s",
            (datetime.now(UTC), run_id),
        )
    return {
        "run_id": run_id,
        "video_id": video_id,
        "shots": len(shots),
        "keyframes": len(keyframes),
        "vectors": len(vectors),
        "collection": collection_name,
    }
