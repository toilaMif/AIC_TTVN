"""Widen ingest_runs.run_id -- run_id embeds the full pipeline_version tag
(e.g. 'caption-aicv3-frame-understanding-production-visualprop-tokens420-batch8-px512-l21_v001',
87 chars) which already exceeds the original 64-char limit on its own.
"""

import sqlalchemy as sa

from alembic import op

revision = "0012_widen_ingest_runs_run_id"
down_revision = "0011_widen_pipeline_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "ingest_runs",
        "run_id",
        type_=sa.String(160),
        existing_type=sa.String(64),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "ingest_runs",
        "run_id",
        type_=sa.String(64),
        existing_type=sa.String(160),
        existing_nullable=False,
    )
