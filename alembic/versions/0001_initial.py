"""Initial source metadata and feature lifecycle schema."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("video_id", sa.String(32), primary_key=True),
        sa.Column("youtube_id", sa.String(32), nullable=False, unique=True),
        sa.Column("watch_url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("duration_expected_sec", sa.Float()),
        sa.Column("availability_status", sa.String(32), nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "ingest_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("run_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("video_count", sa.Integer(), nullable=False),
        sa.Column("keyframe_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "keyframes",
        sa.Column("keyframe_id", sa.String(64), primary_key=True),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("frame_idx", sa.BigInteger(), nullable=False),
        sa.Column("pts_time", sa.Float(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("shot_id", sa.String(64)),
        sa.Column("window_id", sa.String(64)),
        sa.Column("frame_object_key", sa.Text()),
        sa.UniqueConstraint("video_id", "n"),
        sa.UniqueConstraint("video_id", "frame_idx", "pts_time"),
    )
    op.create_index("ix_keyframes_video_id", "keyframes", ["video_id"])
    op.create_table(
        "feature_jobs",
        sa.Column("job_id", sa.String(64), primary_key=True),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("embedding_version", sa.String(128), nullable=False, unique=True),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("dtype", sa.String(16), nullable=False),
        sa.Column("expected_count", sa.Integer(), nullable=False),
        sa.Column("completed_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
    )
    op.create_table(
        "feature_records",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("feature_jobs.job_id"), nullable=False),
        sa.Column(
            "keyframe_id", sa.String(64), sa.ForeignKey("keyframes.keyframe_id"), nullable=False
        ),
        sa.Column("embedding_version", sa.String(128), nullable=False),
        sa.Column("milvus_collection", sa.String(128), nullable=False),
        sa.Column("milvus_pk", sa.BigInteger(), nullable=False),
        sa.Column("import_status", sa.String(32), nullable=False),
        sa.UniqueConstraint("keyframe_id", "embedding_version"),
        sa.UniqueConstraint("milvus_collection", "milvus_pk"),
    )


def downgrade() -> None:
    op.drop_table("feature_records")
    op.drop_table("feature_jobs")
    op.drop_index("ix_keyframes_video_id", table_name="keyframes")
    op.drop_table("keyframes")
    op.drop_table("ingest_runs")
    op.drop_table("videos")
