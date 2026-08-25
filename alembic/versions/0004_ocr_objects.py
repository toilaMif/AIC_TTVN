"""Store OCR records and object detections."""

import sqlalchemy as sa

from alembic import op

revision = "0004_ocr_objects"
down_revision = "0003_asr_segments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_records",
        sa.Column(
            "keyframe_id",
            sa.String(64),
            sa.ForeignKey("keyframes.keyframe_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("frame_idx", sa.BigInteger(), nullable=False),
        sa.Column("pts_time", sa.Float(), nullable=False),
        sa.Column("shot_id", sa.String(64)),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text()),
        sa.Column("normalized_text", sa.Text()),
        sa.Column("search_text", sa.Text()),
        sa.Column("shot_text", sa.Text()),
        sa.Column("shot_search_text", sa.Text()),
        sa.Column("ocr_error", sa.Text()),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("detections_json", sa.Text()),
        sa.Column("merged_detections_json", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_ocr_records_video_time", "ocr_records", ["video_id", "pts_time"])

    op.create_table(
        "object_detections",
        sa.Column("detection_id", sa.String(160), primary_key=True),
        sa.Column(
            "keyframe_id",
            sa.String(64),
            sa.ForeignKey("keyframes.keyframe_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("frame_idx", sa.BigInteger(), nullable=False),
        sa.Column("pts_time", sa.Float(), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("x1_norm", sa.Float(), nullable=False),
        sa.Column("y1_norm", sa.Float(), nullable=False),
        sa.Column("x2_norm", sa.Float(), nullable=False),
        sa.Column("y2_norm", sa.Float(), nullable=False),
        sa.Column("bbox_area_ratio", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_object_detections_video_frame",
        "object_detections",
        ["video_id", "frame_idx"],
    )
    op.create_index("ix_object_detections_class", "object_detections", ["class_name"])


def downgrade() -> None:
    op.drop_index("ix_object_detections_class", table_name="object_detections")
    op.drop_index("ix_object_detections_video_frame", table_name="object_detections")
    op.drop_table("object_detections")
    op.drop_index("ix_ocr_records_video_time", table_name="ocr_records")
    op.drop_table("ocr_records")
