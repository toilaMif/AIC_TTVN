"""Stage 08: low-latency hybrid query over a V2 retrieval pack.

Lexical retrieval and metadata filters run inside SQLite. Visual retrieval uses
the optional FAISS shot index and falls back to an exact memory-mapped NumPy
search. The modality rankings are fused with weighted RRF and collapsed by
shot for presentation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import PIPELINE_VERSION, fetch_snapshot_documents, l2_normalize, rrf_fuse, search_snapshot_fts  # noqa: E402

STAGE_VERSION = f"{PIPELINE_VERSION}-query"
LEVELS = ("event", "shot", "frame")
RRF_WEIGHTS = {"fts:event": 0.8, "fts:shot": 1.0, "fts:frame": 0.65, "ann:shot": 1.15, "ann:frame": 0.85}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--index-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "07-index-pack")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--levels", default="shot,event")
    parser.add_argument("--video-id")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--object", dest="object_label")
    parser.add_argument("--scene")
    parser.add_argument("--query-vector", type=Path)
    parser.add_argument("--visual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-text")
    parser.add_argument("--collapse-shot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_levels(value: str) -> tuple[str, ...]:
    levels = tuple(dict.fromkeys(part.strip().casefold() for part in str(value).split(",") if part.strip()))
    invalid = sorted(set(levels) - set(LEVELS))
    if invalid:
        raise ValueError(f"Unsupported levels {invalid}; choose from {LEVELS}")
    return levels or ("shot", "event")


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _hydrate(path: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    rows = fetch_snapshot_documents(path, list(ids))
    result = {}
    for row in rows:
        payload = row.get("payload")
        value: dict[str, Any] = {}
        if isinstance(payload, str) and payload:
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    value.update(decoded)
            except json.JSONDecodeError:
                pass
        for key, item in row.items():
            if key != "payload" and not _missing(item):
                value[key] = item
        value["doc_id"] = str(row["doc_id"])
        result[value["doc_id"]] = value
    return result


def _matches(row: dict[str, Any], options: argparse.Namespace, levels: Sequence[str]) -> bool:
    if str(row.get("level") or "").casefold() not in levels:
        return False
    if options.video_id and str(row.get("video_id") or "") != str(options.video_id):
        return False
    start = float(row.get("start_time", row.get("pts_time", 0.0)) or 0.0)
    end = float(row.get("end_time", start) or start)
    if options.start_time is not None and end < options.start_time:
        return False
    if options.end_time is not None and start > options.end_time:
        return False
    if options.object_label and str(options.object_label).casefold() not in str(row.get("object_labels_text") or "").casefold():
        return False
    if options.scene and str(options.scene).casefold() not in str(row.get("scene_labels_json") or "").casefold():
        return False
    return True


def _encode_openclip(text: str) -> tuple[np.ndarray | None, str]:
    try:
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            vector = model.encode_text(tokenizer([text]).to(device)).float().cpu().numpy()
        return l2_normalize(vector)[0], "openclip:ViT-B-32/laion2b_s34b_b79k"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}: {exc}"


def _faiss_search(root: Path, vector: np.ndarray, limit: int) -> tuple[list[str], dict[str, float]]:
    index_path, map_path = root / "shot-visual.faiss", root / "shot-visual.faiss.ids.parquet"
    if not index_path.exists() or not map_path.exists():
        return [], {}
    try:
        import faiss
        import pandas as pd

        index = faiss.read_index(str(index_path))
        query = np.ascontiguousarray(l2_normalize(np.asarray(vector, dtype="float32").reshape(1, -1)))
        if query.shape[1] != index.d:
            return [], {}
        scores, positions = index.search(query, min(max(1, limit), index.ntotal))
        mapping = pd.read_parquet(map_path).set_index("faiss_id", drop=False)
        ids, values = [], {}
        for position, score in zip(positions[0], scores[0]):
            if int(position) < 0 or int(position) not in mapping.index:
                continue
            row = mapping.loc[int(position)]
            doc_id = str(row.get("doc_id") or f"shot:{row.get('shot_id')}")
            ids.append(doc_id)
            values[doc_id] = float(score)
        return ids, values
    except Exception:
        return [], {}


def _numpy_search(root: Path, vector: np.ndarray, limit: int, levels: Sequence[str]) -> tuple[list[str], dict[str, float], str | None]:
    candidates = []
    if "shot" in levels:
        candidates.append(("shot", root / "shot-visual.npy", root / "shot-embedding-records.parquet", "shot_id"))
    if "frame" in levels:
        candidates.append(("frame", root / "frame-visual.npy", root / "frame-embedding-records.parquet", "keyframe_id"))
    for level, matrix_path, records_path, id_column in candidates:
        if not matrix_path.exists() or not records_path.exists():
            continue
        import pandas as pd

        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        query = l2_normalize(np.asarray(vector, dtype="float32").reshape(1, -1))[0]
        if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != len(query):
            continue
        scores = np.asarray(matrix @ query, dtype="float32")
        count = min(max(1, limit), len(scores))
        positions = np.argpartition(-scores, count - 1)[:count]
        positions = positions[np.argsort(-scores[positions], kind="stable")]
        records = pd.read_parquet(records_path)
        ids, values = [], {}
        for position in positions:
            row = records.iloc[int(position)]
            doc_id = str(row.get("doc_id") or f"{level}:{row.get(id_column)}")
            ids.append(doc_id)
            values[doc_id] = float(scores[int(position)])
        return ids, values, level
    return [], {}, None


def _visual_search(root: Path, vector: np.ndarray, limit: int, levels: Sequence[str]) -> tuple[list[str], dict[str, float], str | None, str]:
    if "shot" in levels:
        ids, scores = _faiss_search(root, vector, limit)
        if ids:
            return ids, scores, "shot", "faiss"
    ids, scores, level = _numpy_search(root, vector, limit, levels)
    return ids, scores, level, "numpy-memmap" if ids else "unavailable"


def _collapse(rows: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    seen, output = set(), []
    for row in rows:
        key = str(row.get("shot_id") or row.get("doc_id"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def run_query(options: argparse.Namespace) -> dict[str, Any]:
    query = options.query_file.read_text(encoding="utf-8").strip() if options.query_file else options.query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if options.top_k < 1 or options.candidate_k < options.top_k or options.rrf_k < 1:
        raise ValueError("require candidate-k >= top-k >= 1 and rrf-k >= 1")
    levels = parse_levels(options.levels)
    root = options.index_root.resolve()
    snapshot = root / "text-index.sqlite"
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing text-index.sqlite: {snapshot}")
    started = time.perf_counter()
    rankings: dict[str, Sequence[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    lexical_started = time.perf_counter()
    for level in levels:
        rows = search_snapshot_fts(
            snapshot, query, limit=options.candidate_k, levels=[level], video_id=options.video_id,
            min_time=options.start_time, max_time=options.end_time,
            facets=[options.scene] if options.scene else None,
            object_labels=[options.object_label] if options.object_label else None,
        )
        channel = f"fts:{level}"
        rankings[channel] = [str(row["doc_id"]) for row in rows]
        for row in rows:
            scores.setdefault(str(row["doc_id"]), {})["bm25_score"] = float(row["bm25_score"])
    lexical_ms = (time.perf_counter() - lexical_started) * 1000.0

    vector, vector_source = None, None
    if options.query_vector:
        vector, vector_source = np.load(options.query_vector, allow_pickle=False), str(options.query_vector)
    elif options.visual:
        vector, vector_source = _encode_openclip(options.visual_text or query)
    visual_ms_start = time.perf_counter()
    visual_level, visual_engine = None, "disabled" if vector is None else "unavailable"
    if vector is not None:
        ids, visual_scores, visual_level, visual_engine = _visual_search(root, vector, max(options.candidate_k * 4, 100), levels)
        hydrated = _hydrate(snapshot, ids)
        ids = [doc_id for doc_id in ids if doc_id in hydrated and _matches(hydrated[doc_id], options, levels)][:options.candidate_k]
        if ids and visual_level:
            rankings[f"ann:{visual_level}"] = ids
            for doc_id in ids:
                scores.setdefault(doc_id, {})["visual_score"] = float(visual_scores[doc_id])
    visual_ms = (time.perf_counter() - visual_ms_start) * 1000.0

    rankings = {name: values for name, values in rankings.items() if values}
    fused = rrf_fuse(rankings, k=options.rrf_k, limit=max(options.candidate_k, options.top_k), weights=RRF_WEIGHTS)
    hydrated = _hydrate(snapshot, [doc_id for doc_id, _ in fused])
    results = []
    for doc_id, score in fused:
        if doc_id not in hydrated:
            continue
        row = dict(hydrated[doc_id])
        row.update(scores.get(doc_id, {}))
        row["rrf_score"] = float(score)
        row["channel_ranks"] = {name: list(values).index(doc_id) + 1 for name, values in rankings.items() if doc_id in values}
        results.append(row)
    results = _collapse(results, options.collapse_shot)[:options.top_k]
    return {
        "schema_version": "aic-video-v2/1", "stage": "hybrid-query", "pipeline_version": STAGE_VERSION,
        "query": query,
        "filters": {"levels": list(levels), "video_id": options.video_id, "start_time": options.start_time, "end_time": options.end_time, "object": options.object_label, "scene": options.scene},
        "modalities": {"channels": {name: len(values) for name, values in rankings.items()}, "visual_level": visual_level, "visual_engine": visual_engine, "vector_source": vector_source},
        "latency_ms": {"total": (time.perf_counter() - started) * 1000.0, "lexical": lexical_ms, "visual": visual_ms},
        "results": results,
    }


def main() -> None:
    options = parse_args()
    report = run_query(options)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if options.json_output:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(output + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
