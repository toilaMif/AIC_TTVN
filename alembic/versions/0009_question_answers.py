"""Store the selected answer frame(s)/text per exam question."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009_question_answers"
down_revision = "0008_question_done_by"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("exam_questions", sa.Column("answer_frames", postgresql.JSONB()))
    op.add_column("exam_questions", sa.Column("answer_text", sa.Text()))


def downgrade() -> None:
    op.drop_column("exam_questions", "answer_text")
    op.drop_column("exam_questions", "answer_frames")
