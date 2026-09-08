"""Store frame-level captions and search text from the V2 fusion pipeline."""

import sqlalchemy as sa

from alembic import op

revision = "0010_frame_captions"
down_revision = "0009_question_answers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "frame_captions",
        sa.Column(
            "keyframe_id",
            sa.String(64),
            sa.ForeignKey("keyframes.keyframe_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("shot_id", sa.String(64)),
        sa.Column("frame_idx", sa.BigInteger(), nullable=False),
        sa.Column("pts_time", sa.Float(), nullable=False),
        sa.Column("caption_vi_short", sa.Text()),
        sa.Column("caption_vi_detail", sa.Text()),
        sa.Column("search_text_vi", sa.Text(), nullable=False),
        sa.Column("asr_text", sa.Text()),
        sa.Column("ocr_scene_text", sa.Text()),
        sa.Column("object_labels_text", sa.Text()),
        sa.Column("scene_labels_json", sa.Text()),
        sa.Column("description_source", sa.String(32)),
        sa.Column("confidence", sa.Float()),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("dataset_group", sa.String(32)),
        sa.Column("artifact_batch", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_frame_captions_video_time", "frame_captions", ["video_id", "pts_time"])
    op.create_index("ix_frame_captions_dataset_group", "frame_captions", ["dataset_group"])
    # A real GIN index up front avoids the sequential-scan behavior ocr_records
    # has today (to_tsvector computed at query time with no supporting index).
    op.execute(
        """
        CREATE INDEX ix_frame_captions_search_text_gin
        ON frame_captions
        USING GIN (to_tsvector('simple', coalesce(search_text_vi, '')))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_frame_captions_search_text_gin")
    op.drop_index("ix_frame_captions_dataset_group", table_name="frame_captions")
    op.drop_index("ix_frame_captions_video_time", table_name="frame_captions")
    op.drop_table("frame_captions")
