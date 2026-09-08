"""Widen frame_captions.pipeline_version -- newer VLM pipeline tags (e.g.
'aicv3-frame-understanding-production-visualprop-tokens420-batch8-px512') run
past the original 64-char limit.
"""

import sqlalchemy as sa

from alembic import op

revision = "0011_widen_pipeline_version"
down_revision = "0010_frame_captions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "frame_captions",
        "pipeline_version",
        type_=sa.String(128),
        existing_type=sa.String(64),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "frame_captions",
        "pipeline_version",
        type_=sa.String(64),
        existing_type=sa.String(128),
        existing_nullable=False,
    )
