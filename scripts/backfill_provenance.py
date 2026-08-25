"""Backfill dataset-group and artifact-batch provenance without rebuilding indexes."""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from retrieval.config import settings
from retrieval.provenance import artifact_batch_for_video, dataset_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=settings.aic_kaggle_artifact_root,
        help="Root containing <batch>/01-shot-keyframes archives.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SELECT video_id, artifact_batch FROM videos ORDER BY video_id")
        videos = cur.fetchall()
        updates: list[tuple[str, str, str]] = []
        ambiguous: list[str] = []
        for video_id, current_batch in videos:
            batch = artifact_batch_for_video(str(video_id), args.artifact_root)
            if batch is None:
                continue
            if current_batch and str(current_batch).lower() != batch.lower():
                ambiguous.append(f"{video_id}: {current_batch} -> {batch}")
            updates.append((dataset_group(str(video_id)), batch, str(video_id)))
        if ambiguous:
            raise RuntimeError("Conflicting artifact batches:\n" + "\n".join(ambiguous[:20]))
        if not args.dry_run:
            cur.executemany(
                "UPDATE videos SET dataset_group=%s, artifact_batch=%s WHERE video_id=%s",
                updates,
            )
            cur.execute(
                """
                UPDATE keyframes k
                SET dataset_group=v.dataset_group, artifact_batch=v.artifact_batch
                FROM videos v WHERE k.video_id=v.video_id
                """
            )
            cur.execute(
                """
                UPDATE asr_segments a
                SET dataset_group=v.dataset_group, artifact_batch=v.artifact_batch
                FROM videos v WHERE a.video_id=v.video_id
                """
            )
            cur.execute(
                """
                UPDATE ocr_records o
                SET dataset_group=v.dataset_group, artifact_batch=v.artifact_batch
                FROM videos v WHERE o.video_id=v.video_id
                """
            )
            cur.execute(
                """
                UPDATE object_detections d
                SET dataset_group=v.dataset_group, artifact_batch=v.artifact_batch
                FROM videos v WHERE d.video_id=v.video_id
                """
            )
            cur.execute(
                """
                UPDATE ingest_runs r
                SET dataset_group=source.dataset_group, artifact_batch=source.artifact_batch
                FROM (
                    SELECT run_id, max(dataset_group) AS dataset_group,
                           max(artifact_batch) AS artifact_batch
                    FROM keyframes WHERE run_id IS NOT NULL GROUP BY run_id
                ) source WHERE source.run_id=r.run_id
                """
            )
            cur.execute(
                """
                UPDATE feature_jobs j
                SET dataset_group=source.dataset_group, artifact_batch=source.artifact_batch
                FROM (
                    SELECT fr.job_id, max(k.dataset_group) AS dataset_group,
                           max(k.artifact_batch) AS artifact_batch
                    FROM feature_records fr JOIN keyframes k ON k.keyframe_id=fr.keyframe_id
                    GROUP BY fr.job_id
                ) source WHERE source.job_id=j.job_id
                """
            )
    print(f"videos={len(videos)} mapped={len(updates)} dry_run={args.dry_run}")
    if len(updates) != len(videos):
        print(f"unmapped={len(videos) - len(updates)}")


if __name__ == "__main__":
    main()
