"""Support versioned self-extracted shots and keyframes."""

import sqlalchemy as sa

from alembic import op

revision = "0002_self_extracted"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("keyframes_video_id_n_key", "keyframes", type_="unique")
    op.drop_constraint("keyframes_video_id_frame_idx_pts_time_key", "keyframes", type_="unique")
    op.add_column(
        "keyframes", sa.Column("source", sa.String(32), nullable=False, server_default="btc")
    )
    op.add_column(
        "keyframes",
        sa.Column("pipeline_version", sa.String(32), nullable=False, server_default="btc"),
    )
    op.add_column("keyframes", sa.Column("run_id", sa.String(64)))
    op.add_column("keyframes", sa.Column("quality_score", sa.Float()))
    op.add_column("keyframes", sa.Column("quality_fallback", sa.Boolean()))
    op.create_foreign_key("fk_keyframes_run_id", "keyframes", "ingest_runs", ["run_id"], ["run_id"])
    op.create_unique_constraint(
        "uq_keyframes_versioned_ordinal",
        "keyframes",
        ["video_id", "source", "pipeline_version", "n"],
    )
    op.create_table(
        "shots",
        sa.Column("shot_id", sa.String(64), primary_key=True),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("shot_index", sa.Integer(), nullable=False),
        sa.Column("start_frame", sa.BigInteger(), nullable=False),
        sa.Column("end_frame", sa.BigInteger(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("detector", sa.String(64), nullable=False),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("ingest_runs.run_id"), nullable=False),
        sa.UniqueConstraint("video_id", "pipeline_version", "shot_index"),
    )
    op.create_index("ix_shots_video_id", "shots", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_shots_video_id", table_name="shots")
    op.drop_table("shots")
    op.drop_constraint("uq_keyframes_versioned_ordinal", "keyframes", type_="unique")
    op.drop_constraint("fk_keyframes_run_id", "keyframes", type_="foreignkey")
    op.drop_column("keyframes", "quality_fallback")
    op.drop_column("keyframes", "quality_score")
    op.drop_column("keyframes", "run_id")
    op.drop_column("keyframes", "pipeline_version")
    op.drop_column("keyframes", "source")
    op.create_unique_constraint("keyframes_video_id_n_key", "keyframes", ["video_id", "n"])
    op.create_unique_constraint(
        "keyframes_video_id_frame_idx_pts_time_key",
        "keyframes",
        ["video_id", "frame_idx", "pts_time"],
    )
