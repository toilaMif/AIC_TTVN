"""CPU-only smoke test for V2 contracts, temporal fusion, and index packing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    aggregate_embeddings,
    build_events,
    build_temporal_edges,
    l2_normalize,
    interval_join_asr,
    rrf_fuse,
    search_hybrid_snapshot,
    search_snapshot_fts,
)


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=ROOT.parents[2], check=True)


def run_json(script: str, *arguments: str) -> dict:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    completed = subprocess.run(command, cwd=ROOT.parents[2], capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{script} failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return json.loads(completed.stdout)


def main() -> None:
    video_id = "L99_V001"
    with tempfile.TemporaryDirectory(prefix="aic-v2-smoke-") as temporary:
        root = Path(temporary)
        input_root = root / "input"
        frames_root = input_root / "01-shot-keyframe" / video_id
        frames_root.mkdir(parents=True)
        shots = pd.DataFrame([
            {"shot_id": f"{video_id}:shot:0", "video_id": video_id, "shot_index": 0, "start_frame": 0, "end_frame": 100, "start_time": 0.0, "end_time": 4.0, "duration_sec": 4.0, "detector": "test"},
            {"shot_id": f"{video_id}:shot:1", "video_id": video_id, "shot_index": 1, "start_frame": 100, "end_frame": 200, "start_time": 4.0, "end_time": 8.0, "duration_sec": 4.0, "detector": "test"},
        ])
        keyframes = pd.DataFrame([
            {"keyframe_id": f"{video_id}:kf:0", "video_id": video_id, "shot_id": f"{video_id}:shot:0", "ordinal": 1, "frame_idx": 50, "pts_time": 2.0, "fps": 25.0, "frame_path": f"frames/{video_id}/0.webp", "pipeline_version": "aicv4-v2-keyframes"},
            {"keyframe_id": f"{video_id}:kf:1", "video_id": video_id, "shot_id": f"{video_id}:shot:1", "ordinal": 2, "frame_idx": 150, "pts_time": 6.0, "fps": 25.0, "frame_path": f"frames/{video_id}/1.webp", "pipeline_version": "aicv4-v2-keyframes"},
        ])
        shots.to_parquet(frames_root / "shots.parquet", index=False)
        keyframes.to_parquet(frames_root / "keyframes.parquet", index=False)
        vectors = l2_normalize(np.asarray([[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]], dtype="float32"))
        records = keyframes[["keyframe_id", "video_id", "shot_id", "frame_idx", "pts_time", "frame_path"]].copy()
        records["embedding_row"] = range(len(records))
        np.save(frames_root / "visual.npy", vectors)
        records.to_parquet(frames_root / "embedding-records.parquet", index=False)
        shot_records, shot_vectors = aggregate_embeddings(records, vectors)
        shot_records["embedding_row"] = range(len(shot_records))
        np.save(frames_root / "shot-visual.npy", shot_vectors)
        shot_records.to_parquet(frames_root / "shot-embedding-records.parquet", index=False)

        ocr_root = input_root / "02-ocr" / video_id
        ocr_root.mkdir(parents=True)
        pd.DataFrame([
            {"keyframe_id": f"{video_id}:kf:0", "video_id": video_id, "frame_idx": 50, "pts_time": 2.0, "shot_id": f"{video_id}:shot:0", "text": "Chợ Bến Thành", "search_text": "Chợ Bến Thành", "pipeline_version": "aicv4-v2-ocr"},
            {"keyframe_id": f"{video_id}:kf:1", "video_id": video_id, "frame_idx": 150, "pts_time": 6.0, "shot_id": f"{video_id}:shot:1", "text": "Trời mưa", "search_text": "Trời mưa", "pipeline_version": "aicv4-v2-ocr"},
        ]).to_parquet(ocr_root / "ocr.parquet", index=False)

        asr_root = input_root / "03-asr" / video_id
        asr_root.mkdir(parents=True)
        pd.DataFrame([
            {"segment_id": "seg-0", "video_id": video_id, "segment_index": 0, "start_time": 1.0, "end_time": 3.0, "text": "chúng ta đang ở chợ", "pipeline_version": "aicv4-v2-asr"},
            {"segment_id": "seg-1", "video_id": video_id, "segment_index": 1, "start_time": 5.0, "end_time": 7.0, "text": "bên ngoài đang mưa", "pipeline_version": "aicv4-v2-asr"},
        ]).to_parquet(asr_root / "asr-segments.parquet", index=False)

        object_root = input_root / "04-objects" / video_id
        object_root.mkdir(parents=True)
        pd.DataFrame([
            {"detection_id": "det-0", "keyframe_id": f"{video_id}:kf:0", "video_id": video_id, "frame_idx": 50, "pts_time": 2.0, "model": "test", "class_id": 0, "class_name": "person", "confidence": 0.9, "x1_norm": 0.1, "y1_norm": 0.1, "x2_norm": 0.5, "y2_norm": 0.8, "bbox_area_ratio": 0.28, "pipeline_version": "aicv4-v2-objects"},
            {"detection_id": "det-1", "keyframe_id": f"{video_id}:kf:1", "video_id": video_id, "frame_idx": 150, "pts_time": 6.0, "model": "test", "class_id": 2, "class_name": "car", "confidence": 0.8, "x1_norm": 0.2, "y1_norm": 0.3, "x2_norm": 0.8, "y2_norm": 0.9, "bbox_area_ratio": 0.36, "pipeline_version": "aicv4-v2-objects"},
        ]).to_parquet(object_root / "objects.parquet", index=False)

        edges = build_temporal_edges(shots)
        assert len(edges) == 2 and set(edges.relation) == {"NEXT", "AFTER"}
        assert len(build_events(shots, max_duration_sec=30.0)) == 1
        assert rrf_fuse({"a": ["x", "y"], "b": ["y", "z"]})[0][0] == "y"
        joined = interval_join_asr(
            keyframes[["video_id", "pts_time"]],
            pd.DataFrame([
                {"video_id": video_id, "start_time": 1.0, "end_time": 3.0, "text": "một đoạn"},
                {"video_id": video_id, "start_time": 5.0, "end_time": 7.0, "text": "đoạn hai"},
            ]),
            max_chars=8,
        )
        assert joined.iloc[0] == "một đoạn" and len(joined.iloc[1]) <= 8

        fusion_root = root / "fusion"
        run("05_context_fusion.py", "--input-root", str(input_root), "--output-root", str(fusion_root), "--no-enable-vlm")
        assert (fusion_root / video_id / "search-documents.parquet").exists()
        index_root = root / "index"
        run("06_index_pack.py", "--input-root", str(input_root), "--fusion-root", str(fusion_root), "--frame-root", str(input_root), "--output-root", str(index_root))
        manifest = json.loads((index_root / "index-manifest.json").read_text(encoding="utf-8"))
        assert manifest["documents"] == {"frame": 2, "shot": 2, "event": 1}
        assert manifest["vectors"]["vectors"] == 2
        assert manifest["vector_levels"]["frame"]["vectors"] == 2
        assert manifest["sparse"]["postings"] > 0
        snapshot = index_root / "text-index.sqlite"
        lexical = search_snapshot_fts(snapshot, "chợ Bến Thành", levels=["frame", "shot", "event"], limit=10)
        assert lexical and lexical[0]["video_id"] == video_id
        filtered = search_hybrid_snapshot(snapshot, "mưa", levels=("event", "shot", "frame"), object_labels=["car"], limit=5)
        assert filtered and filtered[0]["video_id"] == video_id and filtered[0]["rrf_score"] > 0
        ann_fused = search_hybrid_snapshot(
            snapshot,
            "chợ",
            levels=("shot", "frame"),
            external_rankings={"ann:shot": ["shot:L99_V001:missing", "frame:L99_V001:kf:0"]},
            limit=5,
        )
        assert ann_fused and "channel_ranks" in ann_fused[0]

        query_report = run_json(
            "08_hybrid_query.py", "mua", "--index-root", str(index_root),
            "--levels", "shot,event", "--top-k", "2", "--object", "car", "--no-visual",
        )
        assert query_report["results"] and query_report["results"][0]["video_id"] == video_id
        assert query_report["modalities"]["channels"]["fts:shot"] > 0
        query_vector = root / "query-vector.npy"
        np.save(query_vector, vectors[0])
        visual_report = run_json(
            "08_hybrid_query.py", "mua", "--index-root", str(index_root),
            "--levels", "shot", "--top-k", "1", "--query-vector", str(query_vector),
        )
        assert visual_report["results"] and visual_report["modalities"]["visual_engine"] == "numpy-memmap"

        query_zip = root / "queries.zip"
        with zipfile.ZipFile(query_zip, "w") as archive:
            archive.writestr("query-p1-1-kis.txt", "người ở chợ Bến Thành")
            archive.writestr("query-p1-2-kis.txt", "ô tô ngoài trời mưa")
        eval_root = root / "eval"
        run(
            "07_eval_benchmark.py",
            "--index-root", str(index_root),
            "--query-zip", str(query_zip),
            "--output-root", str(eval_root),
            "--warmup", "1",
            "--repeat", "2",
        )
        report = json.loads((eval_root / "eval-report.json").read_text(encoding="utf-8"))
        assert report["queries"] == 2
        assert report["labels"]["provided"] is False
        assert report["metrics"] is None
        assert report["latency_ms"]["p50"] is not None
        assert report["latency_ms"]["p95"] is not None
        assert report["latency_ms"]["p99"] is not None
        assert report["latency_by_query_type_ms"]["kis"]["samples"] == 4
        assert report["index_size_bytes"] > 0

        labels_path = root / "labels.jsonl"
        labels_path.write_text(
            "\n".join([
                json.dumps({
                    "query_id": "query-p1-1-kis",
                    "relevant_doc_ids": [
                        {"id": "event:L99_V001:self:aicv4-v2:event:000000", "relevance": 2},
                        "frame:L99_V001:kf:0",
                    ],
                    "relevant_shot_ids": ["L99_V001:shot:0"],
                    "relevant_video_ids": [video_id],
                }, ensure_ascii=False),
                json.dumps({
                    "query_id": "query-p1-2-kis",
                    "relevant": [
                        {"level": "doc_id", "id": "frame:L99_V001:kf:1", "relevance": 1},
                        {"level": "shot_id", "id": "L99_V001:shot:1", "relevance": 1},
                        {"level": "video_id", "id": video_id, "relevance": 1},
                    ],
                }, ensure_ascii=False),
            ]) + "\n",
            encoding="utf-8",
        )
        labeled_eval_root = root / "eval-labeled"
        run(
            "07_eval_benchmark.py",
            "--index-root", str(index_root),
            "--query-zip", str(query_zip),
            "--labels", str(labels_path),
            "--output-root", str(labeled_eval_root),
            "--ks", "1,5,10",
            "--warmup", "0",
            "--repeat", "1",
        )
        labeled_report = json.loads((labeled_eval_root / "eval-report.json").read_text(encoding="utf-8"))
        assert labeled_report["labels"]["provided"] is True
        assert labeled_report["quality"]["matched_queries"] == 2
        assert labeled_report["metrics"]["doc_id"]["queries_labeled"] == 2
        assert labeled_report["metrics"]["shot_id"]["queries_labeled"] == 2
        assert labeled_report["metrics"]["video_id"]["queries_labeled"] == 2
        assert labeled_report["metrics"]["doc_id"]["recall@1"] is not None
        assert labeled_report["metrics"]["doc_id"]["mrr"] is not None
        assert labeled_report["metrics"]["doc_id"]["ndcg@10"] is not None
        assert set(labeled_report["quality"]["by_query_type"]) == {"kis"}

        for notebook in ROOT.rglob("*.ipynb"):
            data = json.loads(notebook.read_text(encoding="utf-8"))
            assert data["nbformat"] == 4 and data["cells"]
            assert all(cell.get("outputs", []) == [] for cell in data["cells"] if cell["cell_type"] == "code")
        print("AIC V2 CPU smoke test: PASS")


if __name__ == "__main__":
    main()
