"""Stage 06: build one compact lexical/ANN-ready retrieval pack."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    atomic_write_json,
    bm25_stats,
    build_hybrid_snapshot,
    config_hash,
    l2_normalize,
    sha256_file,
    write_parquet_atomic,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--fusion-root", type=Path, default=None)
    parser.add_argument("--frame-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "07-index-pack")
    parser.add_argument("--encode-caption", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--milvus-index", choices=("HNSW", "IVF_FLAT", "AUTOINDEX"), default="HNSW")
    return parser.parse_args()


def collect_documents(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    roots = [root]
    if root == Path("/kaggle/input"):
        working_root = Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2"))
        if working_root not in roots:
            roots.append(working_root)
    paths = {
        path.resolve()
        for candidate_root in roots
        if candidate_root.exists()
        for path in candidate_root.rglob(filename)
        if path.is_file()
    }
    for path in sorted(paths, key=lambda value: value.as_posix().casefold()):
        try:
            frame = pd.read_parquet(path)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "doc_id" in merged.columns:
        merged = merged.drop_duplicates("doc_id", keep="last")
    return merged.reset_index(drop=True)


def add_weighted_text(documents: pd.DataFrame) -> pd.DataFrame:
    """Attach one compact, consistently boosted lexical field at every level."""
    if documents.empty:
        return documents.copy()
    result = documents.copy()
    result["search_text_weighted"] = result.apply(weighted_text, axis=1)
    return result


def weighted_text(row) -> str:
    # Field repetition approximates BM25 boosts while retaining a single compact
    # FTS5 body. Ticker text is intentionally repeated less than scene/OCR.
    return " ".join([
        str(getattr(row, "caption_vi_detail", "") or ""),
        str(getattr(row, "caption_vi_short", "") or ""),
        str(getattr(row, "asr_text", "") or ""),
        str(getattr(row, "ocr_scene_text", "") or "") + " " + str(getattr(row, "ocr_scene_text", "") or ""),
        str(getattr(row, "ocr_ticker_text", "") or ""),
        str(getattr(row, "object_labels_text", "") or "") + " " + str(getattr(row, "object_labels_text", "") or ""),
    ]).strip()


def encode_caption_vectors(documents: pd.DataFrame, output: Path) -> tuple[str | None, dict]:
    if documents.empty:
        return None, {"enabled": False, "reason": "no documents"}
    try:
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        vectors = []
        texts = documents["search_text_vi"].fillna("").astype(str).tolist()
        for start in range(0, len(texts), 128):
            tokens = tokenizer(texts[start : start + 128]).to(device)
            with torch.inference_mode():
                values = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
            vectors.append(values.cpu().numpy().astype("float32"))
        matrix = np.concatenate(vectors, axis=0)
        np.save(output / "caption-embeddings.npy", matrix)
        return "caption-embeddings.npy", {"enabled": True, "model": "ViT-B-32/laion2b_s34b_b79k", "shape": list(matrix.shape), "dtype": "float32"}
    except Exception as exc:
        return None, {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}


def consolidate_vector_level(
    root: Path,
    output: Path,
    *,
    vector_filename: str,
    records_filename: str,
    output_vector_filename: str,
    output_records_filename: str,
    id_column: str,
) -> dict:
    arrays = []
    records = []
    roots = [root]
    if root == Path("/kaggle/input"):
        working_root = Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2"))
        if working_root not in roots:
            roots.append(working_root)
    paths = {
        path.resolve()
        for candidate_root in roots
        if candidate_root.exists()
        for path in candidate_root.rglob(vector_filename)
        if path.is_file() and output.resolve() not in path.resolve().parents
    }
    for path in sorted(paths, key=lambda value: value.as_posix().casefold()):
        record_path = path.with_name(records_filename)
        if not record_path.exists():
            continue
        values = np.load(path, allow_pickle=False).astype("float32")
        frame = pd.read_parquet(record_path)
        if len(values) != len(frame) or id_column not in frame.columns or values.ndim != 2:
            continue
        arrays.append(values)
        records.append(frame)
    if not arrays:
        return {"available": False, "vectors": 0, "vector_file": output_vector_filename, "records_file": output_records_filename}
    dimensions = {int(array.shape[1]) for array in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"Mixed vector dimensions for {vector_filename}: {sorted(dimensions)}")
    matrix = l2_normalize(np.concatenate(arrays, axis=0))
    records_frame = pd.concat(records, ignore_index=True)
    records_frame["_source_row"] = np.arange(len(records_frame), dtype="int64")
    keep = ~records_frame[id_column].astype(str).duplicated(keep="last")
    duplicates_dropped = int((~keep).sum())
    records_frame = records_frame.loc[keep].copy()
    sort_columns = [column for column in ("video_id", "start_time", "pts_time", "frame_idx", id_column) if column in records_frame.columns]
    if sort_columns:
        records_frame = records_frame.sort_values(sort_columns, kind="stable")
    matrix = matrix[records_frame["_source_row"].to_numpy(dtype="int64")]
    records_frame = records_frame.drop(columns="_source_row").reset_index(drop=True)
    records_frame["embedding_row"] = np.arange(len(records_frame), dtype="int64")
    np.save(output / output_vector_filename, matrix.astype("float32"))
    write_parquet_atomic(records_frame, output / output_records_filename)
    return {
        "available": True,
        "vectors": len(matrix),
        "dimension": int(matrix.shape[1]),
        "dtype": "float32",
        "metric": "IP",
        "normalized": True,
        "source_files": len(arrays),
        "duplicates_dropped": duplicates_dropped,
        "id_column": id_column,
        "vector_file": output_vector_filename,
        "records_file": output_records_filename,
    }


def consolidate_vectors(root: Path, output: Path) -> dict:
    return consolidate_vector_level(
        root,
        output,
        vector_filename="shot-visual.npy",
        records_filename="shot-embedding-records.parquet",
        output_vector_filename="shot-visual.npy",
        output_records_filename="shot-embedding-records.parquet",
        id_column="shot_id",
    )


def consolidate_frame_vectors(root: Path, output: Path) -> dict:
    return consolidate_vector_level(
        root,
        output,
        vector_filename="visual.npy",
        records_filename="embedding-records.parquet",
        output_vector_filename="frame-visual.npy",
        output_records_filename="frame-embedding-records.parquet",
        id_column="keyframe_id",
    )


def ann_parameters(index_type: str, vectors: int) -> dict:
    count = max(1, int(vectors))
    if index_type == "HNSW":
        return {"index_type": "HNSW", "M": 16, "efConstruction": 200, "search_ef": 64}
    if index_type == "IVF_FLAT":
        nlist = max(1, min(4096, int(math.sqrt(count))))
        return {"index_type": "IVF_FLAT", "nlist": nlist, "nprobe": min(16, nlist)}
    return {"index_type": "AUTOINDEX"}


def build_faiss_shot_index(output: Path, index_type: str) -> dict:
    vector_path = output / "shot-visual.npy"
    records_path = output / "shot-embedding-records.parquet"
    index_path = output / "shot-visual.faiss"
    mapping_path = output / "shot-visual.faiss.ids.parquet"
    if not vector_path.exists() or not records_path.exists():
        return {"built": False, "reason": "shot vector payload unavailable"}
    if index_type == "AUTOINDEX":
        return {"built": False, "engine": "faiss", "requested_index_type": index_type, "reason": "AUTOINDEX is delegated to the downstream vector service"}
    try:
        import faiss

        vectors = np.ascontiguousarray(np.load(vector_path, allow_pickle=False).astype("float32"))
        records = pd.read_parquet(records_path)
        if vectors.ndim != 2 or not len(vectors) or len(vectors) != len(records):
            raise ValueError("FAISS shot vectors and records are not aligned")
        dimension = int(vectors.shape[1])
        resolved_type = index_type
        if resolved_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            parameters = {"M": 16, "efConstruction": 200, "search_ef": 64}
        else:
            nlist = max(1, min(4096, int(math.sqrt(len(vectors)))))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = min(16, nlist)
            parameters = {"nlist": nlist, "nprobe": int(index.nprobe)}
        index.add(vectors)
        temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
        faiss.write_index(index, str(temporary))
        temporary.replace(index_path)
        mapping = records.copy()
        mapping.insert(0, "faiss_id", np.arange(len(mapping), dtype="int64"))
        write_parquet_atomic(mapping, mapping_path)
        return {
            "built": True,
            "engine": "faiss",
            "requested_index_type": index_type,
            "index_type": resolved_type,
            "metric": "IP",
            "vectors": len(vectors),
            "dimension": dimension,
            "index_file": index_path.name,
            "mapping_file": mapping_path.name,
            "parameters": parameters,
        }
    except Exception as exc:
        index_path.unlink(missing_ok=True)
        mapping_path.unlink(missing_ok=True)
        return {"built": False, "engine": "faiss", "requested_index_type": index_type, "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    options = parse_args()
    started = time.perf_counter()
    fusion_root = options.fusion_root or options.input_root
    frame_root = options.frame_root or options.input_root
    options.output_root.mkdir(parents=True, exist_ok=True)
    config = {"stage": "index", "pipeline_version": STAGE_VERSION, "milvus_index": options.milvus_index, "encode_caption": options.encode_caption}
    cfg_hash = config_hash(config)

    frame_docs = collect_documents(fusion_root, "search-documents.parquet")
    shot_docs = collect_documents(fusion_root, "shot-docs.parquet")
    event_docs = collect_documents(fusion_root, "event-docs.parquet")
    if frame_docs.empty:
        raise FileNotFoundError("No search-documents.parquet found; run stage 05 first")
    frame_docs = add_weighted_text(frame_docs)
    shot_docs = add_weighted_text(shot_docs)
    event_docs = add_weighted_text(event_docs)
    for name, frame in (("frame-documents.parquet", frame_docs), ("shot-documents.parquet", shot_docs), ("event-documents.parquet", event_docs)):
        write_parquet_atomic(frame, options.output_root / name)
    all_documents = pd.concat([frame_docs, shot_docs, event_docs], ignore_index=True, sort=False)
    postings, stats = bm25_stats(all_documents, "search_text_weighted")
    write_parquet_atomic(postings, options.output_root / "sparse-postings.parquet")
    stats["field_weights"] = {"caption": 1.0, "asr": 1.0, "ocr_scene": 2.0, "ocr_ticker": 0.5, "objects": 2.0}
    atomic_write_json(options.output_root / "bm25-stats.json", stats)
    temporal_edges = collect_documents(fusion_root, "temporal-edges.parquet")
    hybrid_info = build_hybrid_snapshot(
        options.output_root / "text-index.sqlite",
        {"frame": frame_docs, "shot": shot_docs, "event": event_docs},
        text_column="search_text_weighted",
        temporal_edges=temporal_edges,
    )
    fts_enabled = bool(hybrid_info.get("fts5"))

    vector_info = consolidate_vectors(frame_root, options.output_root)
    frame_vector_info = consolidate_frame_vectors(frame_root, options.output_root)
    faiss_info = build_faiss_shot_index(options.output_root, options.milvus_index)
    caption_info = {"enabled": False}
    if options.encode_caption:
        filename, caption_info = encode_caption_vectors(frame_docs, options.output_root)
        if filename:
            frame_docs["caption_embedding_row"] = np.arange(len(frame_docs), dtype="int64")
            write_parquet_atomic(frame_docs, options.output_root / "frame-documents.parquet")

    dimension = int(vector_info.get("dimension") or frame_vector_info.get("dimension") or 512)
    ann = {
        "built": bool(faiss_info.get("built")),
        "state": "built" if faiss_info.get("built") else "payload-ready",
        "faiss": faiss_info,
        "metric_type": "IP",
        "normalization": "L2",
        "frame": {**ann_parameters(options.milvus_index, frame_vector_info.get("vectors", 0)), **frame_vector_info},
        "shot": {**ann_parameters(options.milvus_index, vector_info.get("vectors", 0)), **vector_info},
        "dimension": dimension,
        "two_stage": "shot ANN -> keyframe ANN/exact rerank",
        "loader_required": True,
    }
    atomic_write_json(options.output_root / "ann-config.json", ann)
    files = []
    for path in sorted(options.output_root.iterdir()):
        if path.is_file() and path.name not in {"index-manifest.json", "_SUCCESS.json"}:
            try:
                files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            except OSError:
                pass
    manifest = {
        "schema_version": "aic-video-v2/1", "stage": "index", "pipeline_version": STAGE_VERSION,
        "config_hash": cfg_hash, "documents": {"frame": len(frame_docs), "shot": len(shot_docs), "event": len(event_docs)},
        "sparse": {"postings": len(postings), "fts5": fts_enabled, "stats": "bm25-stats.json", "documents": len(all_documents)},
        "hybrid_snapshot": {
            **hybrid_info,
            "file": "text-index.sqlite",
            "query_api": "v2_runtime.search_hybrid_snapshot",
            "supports": ["fts5", "rrf", "level", "video", "time", "scene_facet", "object", "temporal_edges"],
        },
        "vectors": vector_info,
        "vector_levels": {"shot": vector_info, "frame": frame_vector_info},
        "caption_vectors": caption_info,
        "ann": ann,
        "build_seconds": time.perf_counter() - started, "files": files,
    }
    atomic_write_json(options.output_root / "index-manifest.json", manifest)
    write_success(options.output_root, {"stage": "index", "pipeline_version": STAGE_VERSION, "config_hash": cfg_hash, "frame_documents": len(frame_docs), "shot_documents": len(shot_docs), "event_documents": len(event_docs), "fts5": fts_enabled})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
