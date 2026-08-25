"""Track which account marked an exam question as done."""

import sqlalchemy as sa

from alembic import op

revision = "0008_question_done_by"
down_revision = "0007_question_presence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("exam_questions", sa.Column("done_by", sa.Text()))


def downgrade() -> None:
    op.drop_column("exam_questions", "done_by")
