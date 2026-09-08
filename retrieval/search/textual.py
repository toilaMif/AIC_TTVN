"""OCR and object-detection search."""

from dataclasses import dataclass

import psycopg

from retrieval.config import settings
from retrieval.search.global_visual import _frame_urls


@dataclass(frozen=True)
class TextualResult:
    keyframe_id: str
    video_id: str
    frame_idx: int
    pts_time: float
    score: float
    frame_url: str | None
    shot_id: str | None
    text: str


def search_ocr(query: str, top_k: int = 20, video_id: str | None = None) -> list[TextualResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 200:
        raise ValueError("top_k must be between 1 and 200")
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT keyframe_id, video_id, frame_idx, pts_time, shot_id, text,
                   ts_rank_cd(
                     to_tsvector('simple', COALESCE(search_text, text, '')),
                     plainto_tsquery('simple', %s)
                   ) AS score
            FROM ocr_records
            WHERE (to_tsvector('simple', COALESCE(search_text, text, ''))
                    @@ plainto_tsquery('simple', %s)
               OR text ILIKE %s
               OR shot_text ILIKE %s)
               {"AND video_id = %s" if video_id else ""}
            ORDER BY score DESC, pts_time
            LIMIT %s
            """,
            (query, query, f"%{query}%", f"%{query}%", *([video_id] if video_id else []), top_k),
        )
        rows = cur.fetchall()
    keyframe_ids = [str(row[0]) for row in rows]
    urls = _frame_urls(keyframe_ids)
    return [
        TextualResult(
            keyframe_id=str(keyframe_id),
            video_id=str(video_id),
            frame_idx=int(frame_idx),
            pts_time=float(pts_time),
            score=float(score or 0),
            frame_url=urls.get(str(keyframe_id)),
            shot_id=str(shot_id) if shot_id else None,
            text=str(text),
        )
        for keyframe_id, video_id, frame_idx, pts_time, shot_id, text, score in rows
    ]


def search_captions(query: str, top_k: int = 20, video_id: str | None = None) -> list[TextualResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 200:
        raise ValueError("top_k must be between 1 and 200")
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT keyframe_id, video_id, frame_idx, pts_time, shot_id,
                   COALESCE(NULLIF(caption_vi_detail, ''), search_text_vi) AS text,
                   ts_rank_cd(
                     to_tsvector('simple', COALESCE(search_text_vi, '')),
                     plainto_tsquery('simple', %s)
                   ) AS score
            FROM frame_captions
            WHERE to_tsvector('simple', COALESCE(search_text_vi, ''))
                    @@ plainto_tsquery('simple', %s)
               {"AND video_id = %s" if video_id else ""}
            ORDER BY score DESC, pts_time
            LIMIT %s
            """,
            (query, query, *([video_id] if video_id else []), top_k),
        )
        rows = cur.fetchall()
    keyframe_ids = [str(row[0]) for row in rows]
    urls = _frame_urls(keyframe_ids)
    return [
        TextualResult(
            keyframe_id=str(keyframe_id),
            video_id=str(video_id),
            frame_idx=int(frame_idx),
            pts_time=float(pts_time),
            score=float(score or 0),
            frame_url=urls.get(str(keyframe_id)),
            shot_id=str(shot_id) if shot_id else None,
            text=str(text),
        )
        for keyframe_id, video_id, frame_idx, pts_time, shot_id, text, score in rows
    ]


def search_objects(query: str, top_k: int = 20, video_id: str | None = None) -> list[TextualResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 200:
        raise ValueError("top_k must be between 1 and 200")
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.keyframe_id, d.video_id, d.frame_idx, d.pts_time, k.shot_id,
                   string_agg(DISTINCT d.class_name, ', ' ORDER BY d.class_name) AS labels,
                   max(d.confidence) AS score
            FROM object_detections d
            JOIN keyframes k ON k.keyframe_id=d.keyframe_id
            WHERE d.class_name ILIKE %s
               {"AND d.video_id = %s" if video_id else ""}
            GROUP BY d.keyframe_id, d.video_id, d.frame_idx, d.pts_time, k.shot_id
            ORDER BY score DESC, d.pts_time
            LIMIT %s
            """,
            (f"%{query}%", *([video_id] if video_id else []), top_k),
        )
        rows = cur.fetchall()
    keyframe_ids = [str(row[0]) for row in rows]
    urls = _frame_urls(keyframe_ids)
    return [
        TextualResult(
            keyframe_id=str(keyframe_id),
            video_id=str(video_id),
            frame_idx=int(frame_idx),
            pts_time=float(pts_time),
            score=float(score or 0),
            frame_url=urls.get(str(keyframe_id)),
            shot_id=str(shot_id) if shot_id else None,
            text=str(labels),
        )
        for keyframe_id, video_id, frame_idx, pts_time, shot_id, labels, score in rows
    ]
