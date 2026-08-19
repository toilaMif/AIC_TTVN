"""Timestamped Vietnamese transcript search."""

import re
from dataclasses import dataclass

import psycopg

from retrieval.config import settings
from retrieval.search.global_visual import _frame_urls


@dataclass(frozen=True)
class AsrResult:
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    score: float
    keyframe_id: str | None
    frame_idx: int | None
    pts_time: float | None
    frame_url: str | None


def search_asr(query: str, top_k: int = 20) -> list[AsrResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    token_query = " | ".join(tokens)
    if not token_query:
        raise ValueError("query must contain searchable words")
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT segment_id, video_id, start_time, end_time, text,
                   ts_rank_cd(to_tsvector('simple', text), to_tsquery('simple', %s)) AS score
            FROM asr_segments
            WHERE to_tsvector('simple', text) @@ to_tsquery('simple', %s)
               OR text ILIKE %s
            ORDER BY score DESC, start_time
            LIMIT %s
            """,
            (token_query, token_query, f"%{query}%", top_k),
        )
        segments = cur.fetchall()
        if not segments:
            return []
        results = []
        keyframe_ids = []
        for segment_id, video_id, start, end, text, score in segments:
            cur.execute(
                """
                SELECT keyframe_id, frame_idx, pts_time
                FROM keyframes
                WHERE video_id=%s AND pts_time BETWEEN %s AND %s
                ORDER BY ABS(pts_time - %s) LIMIT 1
                """,
                (video_id, start, end, start),
            )
            frame = cur.fetchone()
            keyframe_id = frame[0] if frame else None
            keyframe_ids.append(keyframe_id) if keyframe_id else None
            results.append((segment_id, video_id, start, end, text, float(score or 0), frame))
    urls = _frame_urls(keyframe_ids)
    return [
        AsrResult(
            segment_id=segment_id,
            video_id=video_id,
            start_time=float(start),
            end_time=float(end),
            text=text,
            score=score,
            keyframe_id=frame[0] if frame else None,
            frame_idx=int(frame[1]) if frame else None,
            pts_time=float(frame[2]) if frame else None,
            frame_url=urls.get(frame[0]) if frame else None,
        )
        for segment_id, video_id, start, end, text, score, frame in results
    ]
