"""Track which account is currently working on which exam question."""

import sqlalchemy as sa

from alembic import op

revision = "0007_question_presence"
down_revision = "0006_exam_questions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("exam_questions", sa.Column("in_progress_by", sa.Text()))
    op.add_column("exam_questions", sa.Column("in_progress_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("exam_questions", "in_progress_at")
    op.drop_column("exam_questions", "in_progress_by")
