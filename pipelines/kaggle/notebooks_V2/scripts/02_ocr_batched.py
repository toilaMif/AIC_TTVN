"""Stage 02: OCR with a cached path map and append-only checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    append_jsonl,
    atomic_write_json,
    config_hash,
    discover_video_parquets,
    join_unique,
    read_jsonl_latest,
    read_video_rows,
    resolve_manifest_path,
    success_matches,
    write_parquet_atomic,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-ocr"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "03-ocr")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--make-compat-zips", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def payload(page: object) -> dict:
    value = getattr(page, "json", page)
    if callable(value):
        value = value()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        value = value["res"]
    return value if isinstance(value, dict) else {}


def values(data: dict, names: tuple[str, ...], default):
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def box_points(box) -> list[list[float]]:
    array = np.asarray(box, dtype="float32")
    if array.shape == (4,):
        x1, y1, x2, y2 = array.tolist()
        return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
    return [[float(x), float(y)] for x, y in array.reshape(-1, 2)[:4]]


def classify(points: list[list[float]], width: int, height: int, text: str) -> str:
    if not points:
        return "scene"
    x = sum(point[0] for point in points) / max(len(points), 1) / max(width, 1)
    y = sum(point[1] for point in points) / max(len(points), 1) / max(height, 1)
    if y > 0.80:
        return "ticker"
    if y < 0.14 and (x < 0.20 or x > 0.80) and len(text) <= 16:
        return "logo"
    return "scene"


def read_image(path: Path, max_side: int = 1600) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        image = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract(ocr, image: np.ndarray, minimum_confidence: float) -> list[dict]:
    records: list[dict] = []
    try:
        pages = ocr.predict(image)
    except Exception:
        pages = ocr.predict([image])
    for page in pages or []:
        data = payload(page)
        boxes = values(data, ("rec_polys", "dt_polys", "rec_boxes"), [])
        texts = values(data, ("rec_texts", "rec_text"), [""] * len(boxes))
        scores = values(data, ("rec_scores", "rec_score"), [1.0] * len(boxes))
        for box, text, score in zip(boxes, texts, scores):
            text = normalize(text)
            confidence = float(score or 0.0)
            if not text or confidence < minimum_confidence:
                continue
            points = box_points(box)
            records.append({
                "text": text,
                "confidence": confidence,
                "bbox": points,
                "text_type": classify(points, image.shape[1], image.shape[0], text),
            })
    unique: dict[str, dict] = {}
    for item in records:
        key = re.sub(r"[^a-z0-9]+", "", item["text"].casefold())
        if key and (key not in unique or item["confidence"] > unique[key]["confidence"]):
            unique[key] = item
    return list(unique.values())


def main() -> None:
    config = vars(args())
    options = argparse.Namespace(**config)
    config.pop("input_root", None)
    config.pop("output_root", None)
    cfg_hash = config_hash({"stage": "ocr", "pipeline_version": STAGE_VERSION, **config})
    options.output_root.mkdir(parents=True, exist_ok=True)
    manifests = discover_video_parquets(options.input_root, "keyframes.parquet", {"keyframe_id", "video_id", "frame_idx", "pts_time", "frame_path"})
    selected = list(manifests.items())[: options.max_videos] if options.max_videos is not None else list(manifests.items())
    if not selected:
        raise FileNotFoundError("No V2 keyframes.parquet found")

    ocr = None
    ocr_init_error = None
    try:
        from paddleocr import PaddleOCR

        try:
            ocr = PaddleOCR(lang="vi", ocr_version="PP-OCRv5", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device="gpu:0")
        except TypeError:
            ocr = PaddleOCR(lang="vi", use_angle_cls=False, use_gpu=True)
    except Exception as exc:
        # PaddleOCR/PaddlePaddle wheels can be incompatible with Kaggle's
        # preloaded CUDA runtime. Preserve the output contract and continue
        # with empty OCR evidence so later stages remain reproducible.
        ocr_init_error = f"{type(exc).__name__}: {exc}"
        print(f"PaddleOCR unavailable; writing empty OCR records: {ocr_init_error}")
    failures: list[dict[str, str]] = []
    completed = 0
    for video_id, manifest_path in selected:
        output = options.output_root / video_id
        if success_matches(output, stage="ocr", version=STAGE_VERSION, config={"stage": "ocr", "pipeline_version": STAGE_VERSION, **config}):
            completed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "ocr-checkpoint.jsonl"
        try:
            keyframes = read_video_rows(manifest_path, video_id).sort_values("frame_idx").reset_index(drop=True)
            if options.max_frames is not None:
                keyframes = keyframes.head(max(0, options.max_frames))
            latest = read_jsonl_latest(checkpoint, "keyframe_id")
            rows: dict[str, dict] = {key: value for key, value in latest.items() if value.get("pipeline_version") == STAGE_VERSION}
            pending = []
            for row in keyframes.itertuples(index=False):
                keyframe_id = str(row.keyframe_id)
                if keyframe_id not in rows or rows[keyframe_id].get("ocr_error"):
                    pending.append(row)
            for start in range(0, len(pending), max(1, options.batch_size)):
                shard = []
                for row in pending[start : start + options.batch_size]:
                    error = None
                    detections: list[dict] = []
                    if ocr is not None:
                        try:
                            path = resolve_manifest_path(str(row.frame_path), manifest_path.parent)
                            detections = extract(ocr, read_image(path), options.min_confidence)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                    scene = [item["text"] for item in detections if item["text_type"] == "scene"]
                    ticker = [item["text"] for item in detections if item["text_type"] == "ticker"]
                    text = join_unique(scene + ticker)
                    item = {
                        "keyframe_id": str(row.keyframe_id), "video_id": str(row.video_id), "frame_idx": int(row.frame_idx),
                        "pts_time": float(row.pts_time), "shot_id": str(getattr(row, "shot_id", "")), "text": text,
                        "raw_text": text, "normalized_text": text.casefold(), "search_text": " ".join(scene),
                        "shot_text": "", "shot_search_text": "", "detections": detections,
                        "detections_json": json.dumps(detections, ensure_ascii=False), "merged_detections_json": "[]",
                        "ocr_error": error, "pipeline_version": STAGE_VERSION,
                    }
                    rows[str(row.keyframe_id)] = item
                    shard.append(item)
                append_jsonl(checkpoint, shard)
                print(f"{video_id}: OCR {min(start + options.batch_size, len(pending))}/{len(pending)}")
            ordered = sorted((rows[str(row.keyframe_id)] for row in keyframes.itertuples(index=False) if str(row.keyframe_id) in rows), key=lambda item: item["frame_idx"])
            # Stable shot text is computed once, then copied as lower-priority evidence.
            by_shot: dict[str, list[dict]] = {}
            for item in ordered:
                by_shot.setdefault(str(item.get("shot_id", "")), []).append(item)
            for group in by_shot.values():
                shot_text = join_unique(item.get("search_text", item.get("text", "")) for item in group)
                merged = [{"text": text, "temporal_support": sum(text.casefold() in item.get("text", "").casefold() for item in group)} for text in shot_text.split()]
                for item in group:
                    item["shot_text"] = shot_text
                    item["shot_search_text"] = shot_text.casefold()
                    item["merged_detections_json"] = json.dumps(merged, ensure_ascii=False)
            frame = pd.DataFrame(ordered)
            write_parquet_atomic(frame, output / "ocr.parquet")
            frame.to_json(output / "ocr.jsonl", orient="records", lines=True, force_ascii=False)
            errors = int(frame["ocr_error"].notna().sum()) if "ocr_error" in frame else 0
            if errors / max(len(frame), 1) > 0.10:
                raise RuntimeError(f"OCR failed for {errors}/{len(frame)} keyframes")
            summary = {
                "schema_version": "aic-video-v2/1", "stage": "ocr", "pipeline_version": STAGE_VERSION,
                "config_hash": cfg_hash, "video_id": video_id, "keyframes": len(frame),
                "keyframes_with_text": int((frame["text"].fillna("").str.len() > 0).sum()), "errors": errors,
                "engine": "PaddleOCR PP-OCRv5" if ocr is not None else "disabled-empty-fallback",
                "init_error": ocr_init_error,
                "checkpoint": "append-only-jsonl",
            }
            atomic_write_json(output / "ocr-summary.json", summary)
            write_success(output, {"stage": "ocr", "pipeline_version": STAGE_VERSION, "config_hash": cfg_hash, "video_id": video_id, "keyframes": len(frame)})
            if options.make_compat_zips:
                archive_path = options.output_root / f"03-ocr-optimized-{video_id}.zip"
                temporary = archive_path.with_suffix(".tmp.zip")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
                    for name in ("ocr.parquet", "ocr.jsonl", "ocr-summary.json", "ocr-checkpoint.jsonl", "_SUCCESS.json"):
                        path = output / name
                        if path.is_file():
                            archive.write(path, name)
                temporary.replace(archive_path)
            completed += 1
        except Exception as exc:
            failures.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
            (output / "_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            gc.collect()
    atomic_write_json(options.output_root / "batch-summary.json", {"stage": "ocr", "pipeline_version": STAGE_VERSION, "completed": completed, "failed": len(failures), "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} OCR videos failed")


if __name__ == "__main__":
    main()
