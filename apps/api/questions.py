"""Import exam question sets (KIS/QA/TRAKE .txt files bundled in a .zip),
track per-question progress ("done") for the competition day, and record the
chosen answer frame(s)/text — writing each question's submission CSV to disk
in the exact format required by the organizers (see data/thể lệ)."""

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import psycopg
from psycopg.types.json import Json

from retrieval.config import settings

_FILENAME_RE = re.compile(r"^query-(p\d+)-(\d+)-(kis|qa|trake)\.txt$", re.IGNORECASE)
_EVENT_RE = re.compile(r"^(E\d+)\s+(.*)$")
_SUBMISSION_ROOT = Path(__file__).resolve().parents[2] / "submission"


@dataclass(frozen=True)
class ParsedQuestion:
    part: str
    number: int
    qtype: str
    text: str
    events: list[dict] | None


@dataclass(frozen=True)
class ParsedQuestionSet:
    questions: list[ParsedQuestion]
    skipped_files: list[str]


def _parse_trake(body: str) -> tuple[str, list[dict]]:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    intro = lines[0] if lines else ""
    events = []
    for line in lines[1:]:
        match = _EVENT_RE.match(line)
        if match:
            events.append({"label": match.group(1), "text": match.group(2).strip()})
    return intro, events


def parse_question_zip(data: bytes, source_filename: str) -> ParsedQuestionSet:
    questions: list[ParsedQuestion] = []
    skipped: list[str] = []
    with ZipFile(BytesIO(data)) as archive:
        for name in archive.namelist():
            base_name = name.rsplit("/", 1)[-1]
            match = _FILENAME_RE.match(base_name)
            if not match:
                if base_name:
                    skipped.append(base_name)
                continue
            part, number_str, qtype = match.groups()
            qtype = qtype.lower()
            body = archive.read(name).decode("utf-8").strip()
            if qtype == "trake":
                intro, events = _parse_trake(body)
                questions.append(ParsedQuestion(part, int(number_str), qtype, intro, events))
            else:
                questions.append(ParsedQuestion(part, int(number_str), qtype, body, None))
    return ParsedQuestionSet(questions=questions, skipped_files=skipped)


def import_question_set(parsed: ParsedQuestionSet, source_filename: str) -> dict:
    set_id = re.sub(r"\.zip$", "", source_filename, flags=re.IGNORECASE).strip() or "bo-de"
    now = datetime.now(timezone.utc)
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO question_sets (set_id, source_filename, imported_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (set_id) DO UPDATE SET
                source_filename = EXCLUDED.source_filename,
                imported_at = EXCLUDED.imported_at
            """,
            (set_id, source_filename, now),
        )
        for question in parsed.questions:
            question_id = f"{set_id}:{question.part}-{question.number}"
            cur.execute(
                """
                INSERT INTO exam_questions
                    (question_id, set_id, part, number, qtype, text, events)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (set_id, part, number) DO UPDATE SET
                    qtype = EXCLUDED.qtype,
                    text = EXCLUDED.text,
                    events = EXCLUDED.events
                """,
                (
                    question_id,
                    set_id,
                    question.part,
                    question.number,
                    question.qtype,
                    question.text,
                    Json(question.events) if question.events is not None else None,
                ),
            )
        conn.commit()
    return {"set_id": set_id, "imported": len(parsed.questions), "skipped_files": parsed.skipped_files}


_ROW_FIELDS = (
    "question_id", "set_id", "part", "number", "qtype", "text", "events",
    "done", "done_at", "done_by", "in_progress_by", "in_progress_at",
    "answer_frames", "answer_text",
)
_ROW_SELECT = (
    "question_id, set_id, part, number, qtype, text, events, done, done_at, done_by, "
    "in_progress_by, in_progress_at, answer_frames, answer_text"
)


def _row_to_dict(row) -> dict:
    record = dict(zip(_ROW_FIELDS, row))
    record["done_at"] = record["done_at"].isoformat() if record["done_at"] else None
    record["in_progress_at"] = record["in_progress_at"].isoformat() if record["in_progress_at"] else None
    return record


def list_questions(set_id: str | None) -> list[dict]:
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        if not set_id:
            cur.execute("SELECT set_id FROM question_sets ORDER BY imported_at DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return []
            set_id = row[0]
        cur.execute(
            f"""
            SELECT {_ROW_SELECT}
            FROM exam_questions
            WHERE set_id = %s
            ORDER BY part, number
            """,
            (set_id,),
        )
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def delete_question_set(set_id: str) -> bool:
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM exam_questions WHERE set_id = %s", (set_id,))
        cur.execute("DELETE FROM question_sets WHERE set_id = %s", (set_id,))
        deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def set_question_done(question_id: str, done: bool, username: str) -> dict | None:
    connection_url = settings.database_url.replace("+psycopg", "")
    now = datetime.now(timezone.utc) if done else None
    done_by = username if done else None
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE exam_questions
            SET done = %s, done_at = %s, done_by = %s,
                in_progress_by = CASE WHEN %s THEN NULL ELSE in_progress_by END,
                in_progress_at = CASE WHEN %s THEN NULL ELSE in_progress_at END
            WHERE question_id = %s
            RETURNING {_ROW_SELECT}
            """,
            (done, now, done_by, done, done, question_id),
        )
        row = cur.fetchone()
        conn.commit()
    return _row_to_dict(row) if row else None


def select_question(question_id: str, username: str) -> dict | None:
    """Mark `username` as currently working on `question_id`, releasing any other
    question in the same set that they previously held (one active pick at a time)."""
    connection_url = settings.database_url.replace("+psycopg", "")
    now = datetime.now(timezone.utc)
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT set_id FROM exam_questions WHERE question_id = %s", (question_id,))
        row = cur.fetchone()
        if not row:
            return None
        set_id = row[0]
        cur.execute(
            """
            UPDATE exam_questions SET in_progress_by = NULL, in_progress_at = NULL
            WHERE set_id = %s AND in_progress_by = %s AND question_id != %s
            """,
            (set_id, username, question_id),
        )
        cur.execute(
            f"""
            UPDATE exam_questions SET in_progress_by = %s, in_progress_at = %s
            WHERE question_id = %s
            RETURNING {_ROW_SELECT}
            """,
            (username, now, question_id),
        )
        row = cur.fetchone()
        conn.commit()
    return _row_to_dict(row) if row else None


_MAX_GUESSES = 100  # organizers cap each submission CSV at 100 rows


def write_submission_csv(question: dict) -> None:
    """(Re)write, or remove, this question's submission/query-<N>-<type>.csv on
    disk so it always mirrors the currently saved answer(s) — matching the exact
    per-row format required by the organizers (data/thể lệ)."""
    path = _SUBMISSION_ROOT / f"query-{question['number']}-{question['qtype']}.csv"
    frames = question.get("answer_frames") or []
    qtype = question["qtype"]

    rows: list[list[str]] = []
    if qtype == "kis":
        rows = [[frame["video_id"], str(frame["frame_idx"])] for frame in frames]
    elif qtype == "qa" and question.get("answer_text"):
        rows = [[frame["video_id"], str(frame["frame_idx"]), question["answer_text"]] for frame in frames]
    elif qtype == "trake" and question.get("events") and len(frames) == len(question["events"]):
        rows = [[frames[0]["video_id"], *[str(frame["frame_idx"]) for frame in frames]]]

    if not rows:
        path.unlink(missing_ok=True)
        return
    _SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, quoting=csv.QUOTE_MINIMAL).writerows(rows)


def toggle_answer_frame(question_id: str, video_id: str, frame_idx: int, pts_time: float) -> dict | None:
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {_ROW_SELECT} FROM exam_questions WHERE question_id = %s", (question_id,))
        row = cur.fetchone()
        if not row:
            return None
        question = _row_to_dict(row)
        frames: list[dict] = list(question.get("answer_frames") or [])
        already_selected = any(f["video_id"] == video_id and f["frame_idx"] == frame_idx for f in frames)

        if already_selected:
            frames = [f for f in frames if not (f["video_id"] == video_id and f["frame_idx"] == frame_idx)]
        elif question["qtype"] in ("kis", "qa"):
            if len(frames) >= _MAX_GUESSES:
                raise ValueError(f"Đã đủ tối đa {_MAX_GUESSES} khung cho câu hỏi này.")
            frames.append({"video_id": video_id, "frame_idx": frame_idx, "pts_time": pts_time})
            frames.sort(key=lambda f: f["pts_time"])
        elif question["qtype"] == "trake":
            max_events = len(question["events"] or [])
            if len(frames) >= max_events:
                raise ValueError(f"Đã đủ {max_events} khung cho câu hỏi này, bỏ chọn 1 khung trước khi thêm.")
            frames.append({"video_id": video_id, "frame_idx": frame_idx, "pts_time": pts_time})
            frames.sort(key=lambda f: f["pts_time"])
        else:
            raise ValueError(f"Loại câu hỏi không hỗ trợ: {question['qtype']}")

        cur.execute(
            f"""
            UPDATE exam_questions SET answer_frames = %s
            WHERE question_id = %s
            RETURNING {_ROW_SELECT}
            """,
            (Json(frames), question_id),
        )
        row = cur.fetchone()
        conn.commit()
    updated = _row_to_dict(row)
    write_submission_csv(updated)
    return updated


def set_answer_text(question_id: str, text: str) -> dict | None:
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE exam_questions SET answer_text = %s
            WHERE question_id = %s
            RETURNING {_ROW_SELECT}
            """,
            (text.strip() or None, question_id),
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        return None
    updated = _row_to_dict(row)
    write_submission_csv(updated)
    return updated
