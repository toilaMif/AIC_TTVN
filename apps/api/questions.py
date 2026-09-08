"""Import exam question sets (KIS/QA/TRAKE .txt files bundled in a .zip),
track per-question progress ("done") for the competition day, and record the
chosen answer frame(s)/text — writing each question's submission CSV to disk
in the exact format required by the organizers (see data/thể lệ)."""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import psycopg
from psycopg.types.json import Json

from retrieval.config import settings

_FILENAME_RE = re.compile(r"^query-(p\d+)-(\d+)-(kis|qa|trake)\.txt$", re.IGNORECASE)
_EVENT_RE = re.compile(r"^(E\d+)\s+(.*)$")
_SUBMISSION_ROOT = Path(__file__).resolve().parents[2] / "submission"
_SUBMISSION_ZIP_FILENAME = "team_TTVN_round1.zip"


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


def _submission_filename(question: dict) -> str:
    return f"query-{question['part']}-{question['number']}-{question['qtype']}.csv"


def _validate_frame(frame: dict, position: int) -> list[str]:
    errors: list[str] = []
    video_id = str(frame.get("video_id") or "").strip()
    frame_idx = frame.get("frame_idx")
    if not video_id:
        errors.append(f"Frame {position}: thiếu video_id.")
    elif video_id.lower().endswith(".mp4"):
        errors.append(f"Frame {position}: video_id không được chứa đuôi .mp4.")
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
        errors.append(f"Frame {position}: frame_idx phải là số nguyên không âm.")
    return errors


def _review_question(question: dict) -> dict:
    frames = list(question.get("answer_frames") or [])
    events = list(question.get("events") or [])
    qtype = str(question.get("qtype") or "").lower()
    missing: list[str] = []
    errors: list[str] = []
    part = str(question.get("part") or "").lower()
    number = question.get("number")
    if not re.fullmatch(r"p\d+", part):
        errors.append("Mã phần thi không hợp lệ (cần có dạng p1, p2, ...).")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        errors.append("Số thứ tự câu hỏi không hợp lệ.")
    if qtype not in {"kis", "qa", "trake"}:
        errors.append(f"Loại câu hỏi không được hỗ trợ: {qtype or '(trống)' }.")
    for position, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            errors.append(f"Frame {position}: dữ liệu không hợp lệ.")
        else:
            errors.extend(_validate_frame(frame, position))

    if qtype in {"kis", "qa"}:
        if not frames:
            missing.append("Chưa chọn frame đáp án.")
        if len(frames) > _MAX_GUESSES:
            errors.append(f"Vượt quá giới hạn {_MAX_GUESSES} dòng đáp án.")

    answer_text = str(question.get("answer_text") or "").strip()
    if qtype == "qa":
        if not answer_text:
            missing.append("Chưa nhập nội dung trả lời QA.")
        elif len(answer_text) > 100:
            errors.append("Nội dung trả lời QA vượt quá 100 ký tự.")

    if qtype == "trake":
        if not events:
            errors.append("Câu TRAKE không có danh sách sự kiện.")
        if not frames:
            missing.append(f"Chưa chọn frame cho {len(events)} sự kiện.")
        elif len(frames) < len(events):
            missing.append(f"Thiếu {len(events) - len(frames)} frame cho chuỗi sự kiện.")
        elif len(frames) > len(events):
            errors.append(f"Thừa {len(frames) - len(events)} frame so với số sự kiện.")
        video_ids = {str(frame.get("video_id") or "").strip() for frame in frames if isinstance(frame, dict)}
        if len(video_ids) > 1:
            errors.append("Tất cả frame của câu TRAKE phải thuộc cùng một video.")
        pts_times: list[float] = []
        for position, frame in enumerate(frames, start=1):
            pts_time = frame.get("pts_time") if isinstance(frame, dict) else None
            if isinstance(pts_time, bool) or not isinstance(pts_time, (int, float)) or not math.isfinite(pts_time):
                errors.append(f"Frame {position}: thiếu thời gian hợp lệ để kiểm tra thứ tự sự kiện.")
                pts_times = []
                break
            pts_times.append(float(pts_time))
        if pts_times and any(current < previous for previous, current in zip(pts_times, pts_times[1:])):
            errors.append("Các frame TRAKE chưa theo đúng thứ tự thời gian của sự kiện.")

    messages = [*missing, *errors]
    status = "invalid" if errors else "missing" if missing else "ready"
    row_count = len(frames) if qtype in {"kis", "qa"} else (1 if status == "ready" else 0)
    return {
        "question_id": question.get("question_id"),
        "part": part,
        "number": number,
        "qtype": qtype,
        "text": question.get("text") or "",
        "events": events,
        "answer_frames": frames,
        "answer_text": answer_text,
        "done": bool(question.get("done")),
        "done_by": question.get("done_by"),
        "filename": _submission_filename({"part": part, "number": number, "qtype": qtype}),
        "row_count": row_count,
        "status": status,
        "messages": messages,
    }


def build_submission_review(questions: list[dict], set_id: str | None = None) -> dict:
    reviewed = [_review_question(question) for question in questions]
    summary = {
        "total": len(reviewed),
        "ready": sum(item["status"] == "ready" for item in reviewed),
        "missing": sum(item["status"] == "missing" for item in reviewed),
        "invalid": sum(item["status"] == "invalid" for item in reviewed),
    }
    return {
        "set_id": set_id or (questions[0].get("set_id") if questions else None),
        "download_filename": _SUBMISSION_ZIP_FILENAME,
        "valid": bool(reviewed) and summary["ready"] == summary["total"],
        "summary": summary,
        "messages": [] if reviewed else ["Chưa có bộ đề để đóng gói."],
        "questions": reviewed,
    }


def get_submission_review(set_id: str | None = None) -> dict:
    return build_submission_review(list_questions(set_id), set_id)


def _quoted_csv_value(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _render_submission_csv(question: dict) -> bytes:
    frames = question.get("answer_frames") or []
    qtype = question["qtype"]
    if qtype == "kis":
        lines = [f"{frame['video_id']},{frame['frame_idx']}" for frame in frames]
    elif qtype == "qa":
        answer = _quoted_csv_value(str(question.get("answer_text") or "").strip())
        lines = [f"{frame['video_id']},{frame['frame_idx']},{answer}" for frame in frames]
    else:
        lines = [f"{frames[0]['video_id']},{','.join(str(frame['frame_idx']) for frame in frames)}"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def build_submission_zip(questions: list[dict], set_id: str | None = None) -> tuple[bytes, dict]:
    review = build_submission_review(questions, set_id)
    if not review["valid"]:
        raise ValueError("Bộ đáp án còn thiếu hoặc không hợp lệ.")
    question_by_id = {question.get("question_id"): question for question in questions}
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission/", b"")
        for item in review["questions"]:
            archive.writestr(f"submission/{item['filename']}", _render_submission_csv(question_by_id[item["question_id"]]))
    return buffer.getvalue(), review


def get_submission_zip(set_id: str | None = None) -> tuple[bytes, dict]:
    return build_submission_zip(list_questions(set_id), set_id)


def get_submission_file(question_id: str) -> tuple[bytes, str] | None:
    """Render just this one question's submission CSV for a single-file download
    (the review screen's per-file "Tải CSV này" button) instead of the full ZIP."""
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {_ROW_SELECT} FROM exam_questions WHERE question_id = %s", (question_id,))
        row = cur.fetchone()
    if not row:
        return None
    question = _row_to_dict(row)
    review = _review_question(question)
    if review["status"] != "ready":
        raise ValueError("Câu hỏi chưa có đáp án hợp lệ để tải.")
    return _render_submission_csv(question), review["filename"]


def write_submission_csv(question: dict) -> None:
    """(Re)write, or remove, this question's submission/query-<N>-<type>.csv on
    disk so it always mirrors the currently saved answer(s) — matching the exact
    per-row format required by the organizers (data/thể lệ)."""
    path = _SUBMISSION_ROOT / _submission_filename(question)
    review = _review_question(question)
    if review["status"] != "ready":
        path.unlink(missing_ok=True)
        return
    _SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_render_submission_csv(question))


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
