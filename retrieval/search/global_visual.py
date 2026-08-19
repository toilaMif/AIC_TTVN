"""Global OpenCLIP text-to-keyframe retrieval."""

from dataclasses import dataclass, replace
from functools import lru_cache

import psycopg
from minio import Minio

from retrieval.config import settings
from retrieval.indexes.milvus import SELF_AICV3_COLLECTION


@dataclass(frozen=True)
class SearchResult:
    keyframe_id: str
    video_id: str
    frame_idx: int
    pts_time: float | None
    score: float
    frame_url: str | None
    shot_id: str | None = None


@dataclass(frozen=True)
class LocalizedClause:
    text: str
    box: tuple[float, float, float, float]
    required: bool = True
    weight: float = 1.0


@lru_cache(maxsize=1)
def _load_model():
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer, preprocess, device


def encode_text(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("query must not be empty")
    import torch

    model, tokenizer, _, device = _load_model()
    tokens = tokenizer([query]).to(device)
    with torch.inference_mode():
        vector = model.encode_text(tokens)
        vector = torch.nn.functional.normalize(vector.float(), dim=-1)
    return vector[0].cpu().numpy().astype("float32").tolist()


def _frame_urls(keyframe_ids: list[str]) -> dict[str, str]:
    if not keyframe_ids:
        return {}
    minio = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=False,
    )
    connection_url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT keyframe_id, frame_object_key FROM keyframes WHERE keyframe_id = ANY(%s)",
            (keyframe_ids,),
        )
        rows = cur.fetchall()
    return {
        keyframe_id: minio.presigned_get_object("aic-frames", object_key)
        for keyframe_id, object_key in rows
        if object_key
    }


def search_text(
    query: str,
    top_k: int = 20,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> list[SearchResult]:
    if top_k < 1 or top_k > 1000:
        raise ValueError("top_k must be between 1 and 1000")
    from pymilvus import MilvusClient

    vector = encode_text(query)
    client = MilvusClient(uri=settings.milvus_uri)
    raw = client.search(
        collection_name=collection_name,
        data=[vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["keyframe_id", "video_id", "frame_idx"],
        search_params={"metric_type": "IP", "params": {}},
    )[0]
    ids = [str(item["entity"]["keyframe_id"]) for item in raw]
    timestamps: dict[str, tuple[float | None, str | None]] = {}
    if ids:
        connection_url = settings.database_url.replace("+psycopg", "")
        with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT keyframe_id, pts_time, shot_id FROM keyframes WHERE keyframe_id = ANY(%s)",
                (ids,),
            )
            timestamps = {
                keyframe_id: (pts_time, shot_id)
                for keyframe_id, pts_time, shot_id in cur.fetchall()
            }
    urls = _frame_urls(ids)
    return [
        SearchResult(
            keyframe_id=str(item["entity"]["keyframe_id"]),
            video_id=str(item["entity"]["video_id"]),
            frame_idx=int(item["entity"]["frame_idx"]),
            pts_time=(timestamps.get(str(item["entity"]["keyframe_id"])) or (None, None))[0],
            score=float(item["distance"]),
            frame_url=urls.get(str(item["entity"]["keyframe_id"])),
            shot_id=(timestamps.get(str(item["entity"]["keyframe_id"])) or (None, None))[1],
        )
        for item in raw
    ]


def diversify_results(
    results: list[SearchResult],
    top_k: int,
    max_results_per_shot: int = 1,
    min_temporal_gap_seconds: float = 3.0,
) -> list[SearchResult]:
    """Keep high-scoring results while preventing one shot dominating the page."""
    if top_k < 1 or max_results_per_shot < 1:
        raise ValueError("top_k and max_results_per_shot must be positive")
    if min_temporal_gap_seconds < 0:
        raise ValueError("min_temporal_gap_seconds must be non-negative")
    selected: list[SearchResult] = []
    shot_counts: dict[str, int] = {}
    for result in results:
        shot_key = result.shot_id or result.keyframe_id
        if shot_counts.get(shot_key, 0) >= max_results_per_shot:
            continue
        if result.pts_time is not None and any(
            chosen.video_id == result.video_id
            and chosen.pts_time is not None
            and abs(chosen.pts_time - result.pts_time) < min_temporal_gap_seconds
            for chosen in selected
        ):
            continue
        selected.append(result)
        shot_counts[shot_key] = shot_counts.get(shot_key, 0) + 1
        if len(selected) == top_k:
            break
    # Backfill by score if strict temporal diversity did not provide enough items.
    selected_ids = {item.keyframe_id for item in selected}
    for result in results:
        if len(selected) == top_k:
            break
        if result.keyframe_id in selected_ids:
            continue
        shot_key = result.shot_id or result.keyframe_id
        if shot_counts.get(shot_key, 0) >= max_results_per_shot:
            continue
        selected.append(result)
        selected_ids.add(result.keyframe_id)
        shot_counts[shot_key] = shot_counts.get(shot_key, 0) + 1
    return selected


def search_text_diversified(
    query: str,
    top_k: int = 20,
    candidate_k: int = 200,
    max_results_per_shot: int = 1,
    min_temporal_gap_seconds: float = 3.0,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> list[SearchResult]:
    if candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k")
    raw = search_text(query, top_k=candidate_k, collection_name=collection_name)
    return diversify_results(
        raw,
        top_k=top_k,
        max_results_per_shot=max_results_per_shot,
        min_temporal_gap_seconds=min_temporal_gap_seconds,
    )


def localized_search(
    global_query: str,
    clauses: list[LocalizedClause],
    candidate_k: int = 200,
    rerank_k: int = 50,
    top_k: int = 20,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> list[SearchResult]:
    """Rerank global candidates using OpenCLIP embeddings of normalized image regions."""
    if not clauses:
        raise ValueError("at least one localized clause is required")
    if rerank_k < 1 or candidate_k < rerank_k:
        raise ValueError("candidate_k must be >= rerank_k >= 1")
    from io import BytesIO

    import torch
    from PIL import Image

    raw = search_text(global_query, candidate_k, collection_name)
    candidates = raw[:rerank_k]
    if not candidates:
        return []
    model, tokenizer, preprocess, device = _load_model()
    texts = [global_query, *[clause.text for clause in clauses]]
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer(texts).to(device))
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    connection_url = settings.database_url.replace("+psycopg", "")
    ids = [candidate.keyframe_id for candidate in candidates]
    with psycopg.connect(connection_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT keyframe_id, frame_object_key FROM keyframes WHERE keyframe_id = ANY(%s)",
            (ids,),
        )
        object_keys = dict(cur.fetchall())
    minio = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=False,
    )
    scored: list[tuple[SearchResult, float]] = []
    global_scores = [candidate.score for candidate in candidates]
    g_min, g_max = min(global_scores), max(global_scores)
    for candidate in candidates:
        object_key = object_keys.get(candidate.keyframe_id)
        if not object_key:
            continue
        response = minio.get_object("aic-frames", object_key)
        try:
            image = Image.open(BytesIO(response.read())).convert("RGB")
        finally:
            response.close()
            response.release_conn()
        crops = []
        for clause in clauses:
            x1, y1, x2, y2 = clause.box
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"invalid normalized box: {clause.box}")
            crop = image.crop(
                (
                    int(x1 * image.width),
                    int(y1 * image.height),
                    int(x2 * image.width),
                    int(y2 * image.height),
                )
            )
            crops.append(preprocess(crop))
        with torch.inference_mode():
            region_features = model.encode_image(torch.stack(crops).to(device))
            region_features = torch.nn.functional.normalize(region_features.float(), dim=-1)
        region_scores = region_features @ text_features[1:].T
        total_weight = sum(clause.weight for clause in clauses)
        weighted = sum(
            clause.weight * float(region_scores[index, index])
            for index, clause in enumerate(clauses)
        ) / max(total_weight, 1e-9)
        normalized_global = (candidate.score - g_min) / max(g_max - g_min, 1e-6)
        final_score = 0.4 * normalized_global + 0.6 * weighted
        scored.append((replace(candidate, score=final_score), final_score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return diversify_results([item[0] for item in scored], top_k=top_k)
