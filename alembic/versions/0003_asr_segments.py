"""Store timestamped Vietnamese ASR segments."""

import sqlalchemy as sa

from alembic import op

revision = "0003_asr_segments"
down_revision = "0002_self_extracted"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asr_segments",
        sa.Column("segment_id", sa.String(96), primary_key=True),
        sa.Column("video_id", sa.String(32), sa.ForeignKey("videos.video_id"), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(16), nullable=False, server_default="vi"),
        sa.Column("avg_logprob", sa.Float()),
        sa.Column("no_speech_prob", sa.Float()),
        sa.Column("words_json", sa.Text()),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("video_id", "segment_index"),
    )
    op.create_index("ix_asr_segments_video_time", "asr_segments", ["video_id", "start_time", "end_time"])


def downgrade() -> None:
    op.drop_index("ix_asr_segments_video_time", table_name="asr_segments")
    op.drop_table("asr_segments")
