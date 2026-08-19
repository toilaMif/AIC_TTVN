"""Import faster-whisper ASR artifacts into PostgreSQL."""

import json
from pathlib import Path

import pandas as pd
import psycopg

from retrieval.config import settings


def import_asr(root: Path, model_name: str = "faster-whisper/large-v3") -> dict[str, int | str]:
    path = root / "asr-segments.parquet"
    if not path.is_file():
        raise ValueError(f"Missing ASR artifact: {path}")
    frame = pd.read_parquet(path)
    required = {"segment_id", "video_id", "segment_index", "start_time", "end_time", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ASR artifact missing columns: {sorted(missing)}")
    if frame["segment_id"].duplicated().any():
        raise ValueError("Duplicate ASR segment_id")
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        videos = set()
        for row in frame.itertuples(index=False):
            videos.add(str(row.video_id))
        cur.execute("SELECT video_id FROM videos WHERE video_id = ANY(%s)", (list(videos),))
        absent = videos - {str(item[0]) for item in cur.fetchall()}
        if absent:
            raise ValueError(f"ASR references unknown videos: {sorted(absent)}")
        cur.execute("DELETE FROM asr_segments WHERE video_id = ANY(%s)", (list(videos),))
        for row in frame.itertuples(index=False):
            words = getattr(row, "words_json", None)
            if not isinstance(words, str) and words is not None:
                words = json.dumps(words, ensure_ascii=False)
            cur.execute(
                """
                INSERT INTO asr_segments
                (segment_id, video_id, segment_index, start_time, end_time, text,
                 language, avg_logprob, no_speech_prob, words_json, model_name)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    str(row.segment_id), str(row.video_id), int(row.segment_index),
                    float(row.start_time), float(row.end_time), str(row.text),
                    str(getattr(row, "language", "vi")),
                    getattr(row, "avg_logprob", None), getattr(row, "no_speech_prob", None),
                    words, model_name,
                ),
            )
    return {"videos": len(videos), "segments": len(frame), "model": model_name}
