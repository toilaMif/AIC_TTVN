"""Stage 04: batched fixed-vocabulary object detection with frame resume."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    append_jsonl,
    atomic_write_json,
    config_hash,
    discover_video_parquets,
    read_jsonl_latest,
    read_video_rows,
    resolve_manifest_path,
    stable_id,
    success_matches,
    write_parquet_atomic,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-objects"
OBJECT_COLUMNS = [
    "detection_id", "keyframe_id", "video_id", "frame_idx", "pts_time", "model",
    "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
    "x1_norm", "y1_norm", "x2_norm", "y2_norm", "bbox_area_ratio", "pipeline_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "05-object-detection")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--model", default=os.getenv("AIC_V2_OBJECT_MODEL", "yolo26m.pt"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--make-compat-zips", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for object detection")
    config = {
        "stage": "objects", "pipeline_version": STAGE_VERSION, "model": options.model,
        "image_size": options.image_size, "confidence": options.confidence, "iou": options.iou,
        "make_compat_zips": options.make_compat_zips,
        "max_frames": options.max_frames,
    }
    cfg_hash = config_hash(config)
    options.output_root.mkdir(parents=True, exist_ok=True)
    manifests = discover_video_parquets(options.input_root, "keyframes.parquet", {"keyframe_id", "video_id", "frame_idx", "pts_time", "frame_path"})
    selected = list(manifests.items())[: options.max_videos] if options.max_videos is not None else list(manifests.items())
    if not selected:
        raise FileNotFoundError("No keyframes.parquet found")

    from ultralytics import YOLO

    model = YOLO(options.model)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch_size = 32 if vram_gb >= 14 else 16 if vram_gb >= 10 else 8
    failures: list[dict[str, str]] = []
    completed = 0
    for video_id, manifest_path in selected:
        output = options.output_root / video_id
        if success_matches(output, stage="objects", version=STAGE_VERSION, config=config):
            completed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "objects-checkpoint.jsonl"
        try:
            keyframes = read_video_rows(manifest_path, video_id).sort_values("frame_idx").reset_index(drop=True)
            if options.max_frames is not None:
                keyframes = keyframes.head(max(0, options.max_frames))
            latest = read_jsonl_latest(checkpoint, "keyframe_id")
            pending_rows = [row for row in keyframes.itertuples(index=False) if str(row.keyframe_id) not in latest]
            for start in range(0, len(pending_rows), batch_size):
                rows = pending_rows[start : start + batch_size]
                paths = [str(resolve_manifest_path(str(row.frame_path), manifest_path.parent)) for row in rows]
                try:
                    results = list(model.predict(
                        source=paths, stream=True, batch=batch_size, imgsz=options.image_size,
                        conf=options.confidence, iou=options.iou, device=0, half=True,
                        verbose=False, save=False, max_det=300,
                    ))
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).casefold() or len(rows) == 1:
                        raise
                    torch.cuda.empty_cache()
                    results = [
                        model.predict(source=path, imgsz=options.image_size, conf=options.confidence, iou=options.iou, device=0, half=True, verbose=False, save=False, max_det=300)[0]
                        for path in paths
                    ]
                shard = []
                for row, result in zip(rows, results):
                    detections: list[dict] = []
                    boxes = getattr(result, "boxes", None)
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        xyxyn = boxes.xyxyn.detach().cpu().numpy()
                        confidences = boxes.conf.detach().cpu().numpy()
                        classes = boxes.cls.detach().cpu().numpy().astype(int)
                        names = result.names
                        for index, (absolute, normalized, confidence, class_id) in enumerate(zip(xyxy, xyxyn, confidences, classes)):
                            x1, y1, x2, y2 = [float(value) for value in absolute]
                            nx1, ny1, nx2, ny2 = [float(value) for value in normalized]
                            class_name = str(names[int(class_id)])
                            detection_id = f"{row.keyframe_id}:det:{stable_id(row.keyframe_id, class_id, index, round(nx1, 4), round(ny1, 4))}"
                            detections.append({
                                "detection_id": detection_id, "keyframe_id": str(row.keyframe_id), "video_id": str(row.video_id),
                                "frame_idx": int(row.frame_idx), "pts_time": float(row.pts_time), "model": options.model,
                                "class_id": int(class_id), "class_name": class_name, "confidence": float(confidence),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "x1_norm": nx1, "y1_norm": ny1, "x2_norm": nx2, "y2_norm": ny2,
                                "bbox_area_ratio": max(0.0, (nx2 - nx1) * (ny2 - ny1)), "pipeline_version": STAGE_VERSION,
                            })
                    item = {"keyframe_id": str(row.keyframe_id), "video_id": str(row.video_id), "frame_idx": int(row.frame_idx), "pts_time": float(row.pts_time), "detections": detections, "pipeline_version": STAGE_VERSION}
                    latest[str(row.keyframe_id)] = item
                    shard.append(item)
                append_jsonl(checkpoint, shard)
                print(f"{video_id}: objects {min(start + batch_size, len(pending_rows))}/{len(pending_rows)}")

            ordered_frames = [latest.get(str(row.keyframe_id), {"keyframe_id": str(row.keyframe_id), "video_id": video_id, "frame_idx": int(row.frame_idx), "pts_time": float(row.pts_time), "detections": []}) for row in keyframes.itertuples(index=False)]
            detections = [item for frame in ordered_frames for item in frame.get("detections", [])]
            objects = pd.DataFrame(detections, columns=OBJECT_COLUMNS)
            write_parquet_atomic(objects, output / "objects.parquet")
            objects.to_json(output / "objects.jsonl", orient="records", lines=True, force_ascii=False)
            summaries = []
            for frame in ordered_frames:
                counts = Counter(item["class_name"] for item in frame.get("detections", []))
                summaries.append({
                    "keyframe_id": frame["keyframe_id"], "video_id": frame["video_id"], "frame_idx": frame["frame_idx"],
                    "pts_time": frame["pts_time"], "object_labels_text": " ".join(sorted(counts)),
                    "object_counts_json": json.dumps(dict(counts), ensure_ascii=False), "detection_count": sum(counts.values()),
                })
            summary_frame = pd.DataFrame(summaries)
            write_parquet_atomic(summary_frame, output / "frame-object-summary.parquet")
            pd.DataFrame([{"class_name": key, "count": value} for key, value in Counter(objects.get("class_name", pd.Series(dtype=str))).most_common()]).to_csv(output / "detected-classes.csv", index=False)
            summary = {
                "schema_version": "aic-video-v2/1", "stage": "objects", "pipeline_version": STAGE_VERSION,
                "config_hash": cfg_hash, "video_id": video_id, "model": options.model, "keyframes": len(keyframes),
                "detections": len(objects), "frames_without_detection": int((summary_frame.detection_count == 0).sum()),
                "batch_size": batch_size, "image_size": options.image_size,
            }
            atomic_write_json(output / "object-summary.json", summary)
            write_success(output, {"stage": "objects", "pipeline_version": STAGE_VERSION, "config_hash": cfg_hash, "video_id": video_id, "keyframes": len(keyframes), "detections": len(objects)})
            if options.make_compat_zips:
                archive_path = options.output_root / f"05-object-detection-{video_id}.zip"
                temporary = archive_path.with_suffix(".tmp.zip")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
                    for name in ("objects.parquet", "objects.jsonl", "frame-object-summary.parquet", "object-summary.json", "detected-classes.csv", "_SUCCESS.json"):
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
            torch.cuda.empty_cache()
    atomic_write_json(options.output_root / "batch-summary.json", {"stage": "objects", "pipeline_version": STAGE_VERSION, "completed": completed, "failed": len(failures), "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} object videos failed")


if __name__ == "__main__":
    main()
