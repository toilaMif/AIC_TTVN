"""Semantic (embedding-based) search over Vietnamese frame captions.

Complements retrieval.search.textual.search_captions, which does exact
keyword matching (Postgres full-text search). This module embeds
caption text with a Vietnamese sentence-embedding model and searches a
dedicated Milvus collection, catching paraphrases/synonyms that keyword
search misses.
"""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import psycopg

from retrieval.config import settings
from retrieval.indexes.milvus import CAPTION_TEXT_COLLECTION, stable_milvus_pk, upsert_visual_vectors
from retrieval.search.global_visual import _frame_urls

CAPTION_EMBEDDING_MODEL = "dangvantuan/vietnamese-embedding"
CAPTION_EMBEDDING_DIM = 768


@dataclass(frozen=True)
class SemanticCaptionResult:
    keyframe_id: str
    video_id: str
    frame_idx: int
    pts_time: float
    score: float
    frame_url: str | None
    shot_id: str | None
    text: str


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CAPTION_EMBEDDING_MODEL, device=device)
    # PhoBERT's position-embedding table tops out at 258 tokens; some
    # search_text_vi values (object/scene labels appended to the caption)
    # run longer than that and crash the embedding lookup unless truncation
    # is capped here (sentence-transformers won't infer this on its own for
    # this model).
    model.max_seq_length = 256
    return model


def encode_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_and_upsert_captions(frame: pd.DataFrame, video_id: str) -> int:
    """Embed every row's search_text_vi and upsert into the caption Milvus
    collection. Called right after a video's captions land in Postgres
    (retrieval/storage/caption_import.py) so new imports stay indexed
    automatically, and by the `embed-captions` CLI backfill for videos
    already imported before this feature existed."""
    texts = frame["search_text_vi"].fillna("").astype(str).tolist()
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        return 0
    vectors = encode_texts([t for _, t in non_empty])
    records = [
        {
            "pk": stable_milvus_pk(str(frame.iloc[i]["keyframe_id"])),
            "keyframe_id": str(frame.iloc[i]["keyframe_id"]),
            "video_id": str(video_id),
            "frame_idx": int(frame.iloc[i]["frame_idx"]),
            "embedding": vector,
        }
        for (i, _), vector in zip(non_empty, vectors)
    ]
    return upsert_visual_vectors(
        settings.milvus_uri, CAPTION_TEXT_COLLECTION, records, CAPTION_EMBEDDING_DIM
    )


def search_captions_semantic(
    query: str,
    top_k: int = 20,
    video_id: str | None = None,
    collection_name: str = CAPTION_TEXT_COLLECTION,
) -> list[SemanticCaptionResult]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1 or top_k > 200:
        raise ValueError("top_k must be between 1 and 200")
    from pymilvus import MilvusClient

    vector = encode_texts([query])[0]
    client = MilvusClient(uri=settings.milvus_uri)
    raw = client.search(
        collection_name=collection_name,
        data=[vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["keyframe_id", "video_id", "frame_idx"],
        search_params={"metric_type": "IP", "params": {}},
        filter=f'video_id == "{video_id.replace(chr(34), "")}"' if video_id else "",
    )[0]
    ids = [str(item["entity"]["keyframe_id"]) for item in raw]
    details: dict[str, tuple[float, str | None, str]] = {}
    if ids:
        connection_url = settings.database_url.replace("+psycopg", "")
        with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT keyframe_id, pts_time, shot_id,
                          COALESCE(NULLIF(caption_vi_detail, ''), search_text_vi)
                   FROM frame_captions WHERE keyframe_id = ANY(%s)""",
                (ids,),
            )
            details = {
                keyframe_id: (pts_time, shot_id, text)
                for keyframe_id, pts_time, shot_id, text in cur.fetchall()
            }
    urls = _frame_urls(ids)
    return [
        SemanticCaptionResult(
            keyframe_id=str(item["entity"]["keyframe_id"]),
            video_id=str(item["entity"]["video_id"]),
            frame_idx=int(item["entity"]["frame_idx"]),
            pts_time=float((details.get(str(item["entity"]["keyframe_id"])) or (0.0, None, ""))[0] or 0.0),
            score=float(item["distance"]),
            frame_url=urls.get(str(item["entity"]["keyframe_id"])),
            shot_id=(details.get(str(item["entity"]["keyframe_id"])) or (0.0, None, ""))[1],
            text=str((details.get(str(item["entity"]["keyframe_id"])) or (0.0, None, ""))[2] or ""),
        )
        for item in raw
    ]
