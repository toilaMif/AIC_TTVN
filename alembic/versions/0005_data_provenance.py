"""Track corpus and artifact-batch provenance across imported data."""

import sqlalchemy as sa

from alembic import op

revision = "0005_data_provenance"
down_revision = "0004_ocr_objects"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("dataset_group", sa.String(32)))
    op.add_column("videos", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("ingest_runs", sa.Column("dataset_group", sa.String(32)))
    op.add_column("ingest_runs", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("keyframes", sa.Column("dataset_group", sa.String(32)))
    op.add_column("keyframes", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("asr_segments", sa.Column("dataset_group", sa.String(32)))
    op.add_column("asr_segments", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("ocr_records", sa.Column("dataset_group", sa.String(32)))
    op.add_column("ocr_records", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("object_detections", sa.Column("dataset_group", sa.String(32)))
    op.add_column("object_detections", sa.Column("artifact_batch", sa.String(64)))
    op.add_column("feature_jobs", sa.Column("dataset_group", sa.String(32)))
    op.add_column("feature_jobs", sa.Column("artifact_batch", sa.String(64)))

    op.execute(
        """
        UPDATE videos
        SET dataset_group = substring(video_id from '^(L[0-9]+)')
        WHERE dataset_group IS NULL
        """
    )
    op.execute(
        """
        UPDATE keyframes k
        SET dataset_group = v.dataset_group
        FROM videos v
        WHERE k.video_id = v.video_id AND k.dataset_group IS NULL
        """
    )
    op.execute(
        """
        UPDATE asr_segments a
        SET dataset_group = v.dataset_group
        FROM videos v
        WHERE a.video_id = v.video_id AND a.dataset_group IS NULL
        """
    )
    op.execute(
        """
        UPDATE ocr_records o
        SET dataset_group = v.dataset_group
        FROM videos v
        WHERE o.video_id = v.video_id AND o.dataset_group IS NULL
        """
    )
    op.execute(
        """
        UPDATE object_detections d
        SET dataset_group = v.dataset_group
        FROM videos v
        WHERE d.video_id = v.video_id AND d.dataset_group IS NULL
        """
    )
    op.execute(
        """
        UPDATE feature_jobs j
        SET dataset_group = source.dataset_group
        FROM (
            SELECT fr.job_id, max(k.dataset_group) AS dataset_group
            FROM feature_records fr
            JOIN keyframes k ON k.keyframe_id = fr.keyframe_id
            GROUP BY fr.job_id
        ) AS source
        WHERE source.job_id = j.job_id AND j.dataset_group IS NULL
        """
    )
    op.create_index("ix_videos_dataset_group", "videos", ["dataset_group"])
    op.create_index("ix_videos_artifact_batch", "videos", ["artifact_batch"])
    op.create_index("ix_keyframes_dataset_group", "keyframes", ["dataset_group"])
    op.create_index("ix_keyframes_artifact_batch", "keyframes", ["artifact_batch"])
    op.create_index("ix_asr_segments_dataset_group", "asr_segments", ["dataset_group"])
    op.create_index("ix_ocr_records_dataset_group", "ocr_records", ["dataset_group"])
    op.create_index("ix_object_detections_dataset_group", "object_detections", ["dataset_group"])


def downgrade() -> None:
    for index, table in (
        ("ix_object_detections_dataset_group", "object_detections"),
        ("ix_ocr_records_dataset_group", "ocr_records"),
        ("ix_asr_segments_dataset_group", "asr_segments"),
        ("ix_keyframes_artifact_batch", "keyframes"),
        ("ix_keyframes_dataset_group", "keyframes"),
        ("ix_videos_artifact_batch", "videos"),
        ("ix_videos_dataset_group", "videos"),
    ):
        op.drop_index(index, table_name=table)
    for table, columns in {
        "feature_jobs": ("artifact_batch", "dataset_group"),
        "object_detections": ("artifact_batch", "dataset_group"),
        "ocr_records": ("artifact_batch", "dataset_group"),
        "asr_segments": ("artifact_batch", "dataset_group"),
        "keyframes": ("artifact_batch", "dataset_group"),
        "ingest_runs": ("artifact_batch", "dataset_group"),
        "videos": ("artifact_batch", "dataset_group"),
    }.items():
        for column in columns:
            op.drop_column(table, column)
