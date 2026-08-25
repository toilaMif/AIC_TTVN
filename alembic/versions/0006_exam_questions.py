"""Imported exam question sets (KIS/QA/TRAKE) and per-question progress."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006_exam_questions"
down_revision = "0005_data_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "question_sets",
        sa.Column("set_id", sa.String(128), primary_key=True),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "exam_questions",
        sa.Column("question_id", sa.String(160), primary_key=True),
        sa.Column("set_id", sa.String(128), sa.ForeignKey("question_sets.set_id"), nullable=False),
        sa.Column("part", sa.String(16), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("qtype", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("events", postgresql.JSONB()),
        sa.Column("done", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("done_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("set_id", "part", "number"),
    )
    op.create_index("ix_exam_questions_set_id", "exam_questions", ["set_id"])


def downgrade() -> None:
    op.drop_index("ix_exam_questions_set_id", table_name="exam_questions")
    op.drop_table("exam_questions")
    op.drop_table("question_sets")
