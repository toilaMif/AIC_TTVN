"""Video-level topic/description search using enriched media-info metadata
(title, topic, key_points, subject_terms) rather than per-keyframe features."""

import re
from dataclasses import dataclass

import psycopg

from retrieval.config import settings
from retrieval.search.global_visual import _frame_urls

_SEARCH_EXPR = """
    coalesce(title,'') || ' ' ||
    coalesce(description,'') || ' ' ||
    coalesce(source_metadata->>'topic','') || ' ' ||
    coalesce((source_metadata->'key_points')::text,'') || ' ' ||
    coalesce((source_metadata->'subject_terms')::text,'')
"""


@dataclass(frozen=True)
class TopicResult:
    video_id: str
    title: str
    topic: str | None
    score: float
    keyframe_id: str | None
    frame_idx: int | None
    pts_time: float | None
    frame_url: str | None
    text: str


def search_topic(query: str, top_k: int = 20) -> list[TopicResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 200:
        raise ValueError("top_k must be between 1 and 200")
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    token_query = " | ".join(tokens)
    if not token_query:
        raise ValueError("query must contain searchable words")

    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT video_id, title, source_metadata->>'topic' AS topic,
                   ts_rank_cd(to_tsvector('simple', {_SEARCH_EXPR}), to_tsquery('simple', %s)) AS score
            FROM videos
            WHERE to_tsvector('simple', {_SEARCH_EXPR}) @@ to_tsquery('simple', %s)
               OR title ILIKE %s
               OR source_metadata->>'topic' ILIKE %s
            ORDER BY score DESC
            LIMIT %s
            """,
            (token_query, token_query, f"%{query}%", f"%{query}%", top_k),
        )
        rows = cur.fetchall()
        if not rows:
            return []

        results = []
        keyframe_ids = []
        for video_id, title, topic, score in rows:
            cur.execute(
                """
                SELECT keyframe_id, frame_idx, pts_time
                FROM keyframes WHERE video_id=%s
                ORDER BY pts_time LIMIT 1
                """,
                (video_id,),
            )
            frame = cur.fetchone()
            if frame:
                keyframe_ids.append(frame[0])
            results.append((video_id, title, topic, float(score or 0), frame))

    urls = _frame_urls(keyframe_ids)
    return [
        TopicResult(
            video_id=video_id,
            title=title,
            topic=topic,
            score=score,
            keyframe_id=frame[0] if frame else None,
            frame_idx=int(frame[1]) if frame else None,
            pts_time=float(frame[2]) if frame else None,
            frame_url=urls.get(frame[0]) if frame else None,
            text=topic or title or "",
        )
        for video_id, title, topic, score, frame in results
    ]
