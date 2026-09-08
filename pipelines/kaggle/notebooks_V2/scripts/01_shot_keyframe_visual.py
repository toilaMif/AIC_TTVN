"""Stage 01: shots, adaptive keyframes, and retrieval vectors in one GPU pass.

The V1 pipeline discarded the OpenCLIP vectors used for keyframe selection and
then encoded the selected WebP files again. This stage keeps the selected
vectors and writes the complete legacy-compatible stage 01 + stage 02 contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    aggregate_embeddings,
    atomic_write_json,
    config_hash,
    l2_normalize,
    make_keyframe_id,
    make_shot_id,
    success_matches,
    validate_frame_contract,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-keyframes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "01-shot-keyframe")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--transnet-root", type=Path, default=Path("/kaggle/working/TransNetV2"))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--make-compat-zips", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def find_manifest(input_root: Path, explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    roots = [input_root]
    working_root = Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2"))
    if working_root not in roots:
        roots.append(working_root)
    candidates = sorted(
        {path.resolve() for root in roots if root.exists() for path in root.rglob("video-manifest.parquet")},
        key=lambda path: path.as_posix().casefold(),
    )
    if not candidates:
        raise FileNotFoundError("Attach the output of stage 00 (video-manifest.parquet)")
    return candidates[0]


def resolve_video(row, input_root: Path) -> Path:
    original = Path(str(row.path))
    if original.exists():
        return original
    matches = sorted(input_root.rglob(f"{row.video_id}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Video not found for {row.video_id}")
    return matches[0]


def perceptual_hash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    dct = cv2.dct(resized)[:8, :8].ravel()[1:]
    median = float(np.median(dct))
    return sum(int(value > median) << index for index, value in enumerate(dct))


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def frame_quality(frame: np.ndarray) -> tuple[float, dict[str, float | int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luma = float(gray.mean())
    black_ratio = float((gray < 16).mean())
    white_ratio = float((gray > 245).mean())
    exposure = max(0.0, 1.0 - abs(luma - 127.5) / 127.5)
    sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(1500.0))
    score = 0.60 * sharpness_score + 0.25 * exposure + 0.15 * (1.0 - max(black_ratio, white_ratio))
    return score, {
        "sharpness": sharpness,
        "mean_luma": luma,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "phash": perceptual_hash(frame),
    }


def keyframe_budget(duration: float, motion_score: float) -> int:
    if duration <= 4.0:
        budget = 1
    elif duration <= 12.0:
        budget = 2
    else:
        budget = 3
    if motion_score >= 0.12 and duration >= 2.0:
        budget += 1
    return min(4, budget)


def farthest_point_select(records: list[dict], budget: int) -> list[int]:
    if len(records) <= budget:
        return list(range(len(records)))
    features = np.asarray([row["feature"] for row in records], dtype="float32")
    qualities = np.asarray([row["quality"] for row in records], dtype="float32")
    frames = np.asarray([row["frame_idx"] for row in records], dtype="float32")
    centrality = (features @ features.T).mean(axis=1)
    selected = [int(np.argmax(0.65 * qualities + 0.35 * centrality))]
    frame_span = max(float(frames.max() - frames.min()), 1.0)
    while len(selected) < budget:
        semantic_distance = 1.0 - np.max(features @ features[selected].T, axis=1)
        temporal_distance = np.min(np.abs(frames[:, None] - frames[selected][None, :]), axis=1) / frame_span
        score = 0.65 * semantic_distance + 0.20 * temporal_distance + 0.15 * qualities
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return sorted(selected, key=lambda index: records[index]["frame_idx"])


def zip_selected(destination: Path, root: Path, names: list[str], include_frames: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in names:
            path = root / name
            if path.is_file():
                archive.write(path, name)
        if include_frames:
            for path in sorted((root / "frames").rglob("*.webp")):
                archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(destination)


def archive_legacy_layout(archive_path: Path, root: Path, names: list[str], include_frames: bool = False) -> None:
    """Create a ZIP whose top-level entries match the V1 importer contract."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in names:
            path = root / name
            if path.is_file():
                archive.write(path, name)
        if include_frames:
            for path in sorted((root / "frames").rglob("*.webp")):
                archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive_path)


def ensure_compat_archives(batch_root: Path, output: Path, video_id: str) -> None:
    archive_legacy_layout(
        batch_root / "01-shot-keyframes" / f"01-shot-keyframe-{video_id}.zip",
        output,
        ["summary.json", "frame-mapping-report.json", "shots.parquet", "keyframes.parquet", "_SUCCESS.json"],
        include_frames=True,
    )
    archive_legacy_layout(
        batch_root / "02-visual-embeddings" / f"02-visual-embedding-{video_id}.zip",
        output,
        ["embedding-records.parquet", "visual.npy", "visual-summary.json", "_SUCCESS.json"],
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for stage 01")

    import av
    import open_clip

    manifest_path = find_manifest(args.input_root, args.manifest)
    manifest = pd.read_parquet(manifest_path).sort_values("video_id").reset_index(drop=True)
    if args.max_videos is not None:
        manifest = manifest.head(max(0, args.max_videos))
    config = {
        "stage": "shot-keyframe-visual",
        "pipeline_version": STAGE_VERSION,
        "model": "ViT-B-32/laion2b_s34b_b79k",
        "candidate_stride_sec": 1.25,
        "max_candidates_per_shot": 24,
        "min_shot_sec": 0.60,
        "duplicate_hash_distance": 6,
        "webp_quality": 90,
    }
    cfg_hash = config_hash(config)
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    transnet = None
    transnet_error = None
    inference_root = args.transnet_root / "inference"
    if inference_root.exists():
        try:
            sys.path.insert(0, str(inference_root))
            from transnetv2 import TransNetV2

            transnet = TransNetV2()
        except Exception as exc:
            transnet_error = f"{type(exc).__name__}: {exc}"
    else:
        transnet_error = f"inference directory missing: {inference_root}"
    if transnet is None:
        print(f"TransNetV2 unavailable ({transnet_error}); using PySceneDetect ContentDetector")
    print("GPU:", torch.cuda.get_device_name(0), "videos:", len(manifest))

    def encode_batch(frames: list[np.ndarray]) -> np.ndarray:
        tensors = [preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames]
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            values = model.encode_image(batch)
            values = torch.nn.functional.normalize(values.float(), dim=-1)
        return values.cpu().numpy().astype("float32")

    completed = failed = 0
    failures: list[dict[str, str]] = []
    for number, item in enumerate(manifest.itertuples(index=False), start=1):
        video_id = str(item.video_id)
        output = args.output_root / video_id
        frame_root = output / "frames" / video_id
        if success_matches(output, stage="shot-keyframe-visual", version=STAGE_VERSION, config=config):
            if args.make_compat_zips:
                ensure_compat_archives(args.output_root.parent, output, video_id)
            print(f"[{number}/{len(manifest)}] skip {video_id}")
            completed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        frame_root.mkdir(parents=True, exist_ok=True)
        try:
            video_path = resolve_video(item, args.input_root)
            fps = float(item.fps)
            estimated_frame_count = max(int(item.frame_count or 0), int(round(float(item.duration_sec) * fps)))
            if transnet is not None:
                _, predictions, _ = transnet.predict_video(str(video_path))
                raw = transnet.predictions_to_scenes(predictions)
                scenes = [(int(start), int(end) + 1) for start, end in raw]
                detector_name = "transnetv2"
            else:
                from scenedetect import ContentDetector, SceneManager, open_video

                scene_video = open_video(str(video_path))
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=max(1, round(0.60 * fps))))
                scene_manager.detect_scenes(scene_video, show_progress=False)
                scenes = [
                    (int(start.get_frames()), int(end.get_frames()))
                    for start, end in scene_manager.get_scene_list(start_in_scene=True)
                ]
                detector_name = "pyscenedetect-content"
            min_frames = max(1, round(0.60 * fps))
            merged: list[tuple[int, int]] = []
            for start, end in scenes or [(0, estimated_frame_count)]:
                if merged and end - start < min_frames:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            scenes = merged

            shot_rows: list[dict] = []
            wanted: dict[int, tuple[int, int]] = {}
            for shot_index, (start, end) in enumerate(scenes):
                duration = max((end - start) / fps, 1.0 / fps)
                count = min(24, max(1, math.ceil(duration / 1.25)))
                indices = np.unique(np.linspace(start, max(start, end - 1), count, dtype=int))
                shot_id = make_shot_id(video_id, shot_index, STAGE_VERSION)
                shot_rows.append({
                    "shot_id": shot_id, "video_id": video_id, "shot_index": shot_index,
                    "start_frame": start, "end_frame": end, "start_time": start / fps,
                    "end_time": end / fps, "duration_sec": duration, "detector": detector_name,
                })
                for frame_idx in indices:
                    window_id = int((int(frame_idx) - start) / max(1, fps * 5.0))
                    wanted[int(frame_idx)] = (shot_index, window_id)

            accepted: list[list[dict]] = [[] for _ in scenes]
            hashes: list[list[int]] = [[] for _ in scenes]
            best: list[dict | None] = [None for _ in scenes]
            thumbs: list[list[np.ndarray]] = [[] for _ in scenes]
            encode_records: list[dict] = []
            encode_frames: list[np.ndarray] = []

            def flush_encode() -> None:
                if not encode_frames:
                    return
                features = encode_batch(encode_frames)
                for record, feature in zip(encode_records, features):
                    record["feature"] = feature
                encode_records.clear()
                encode_frames.clear()

            decoded_count = 0
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                for frame_idx, decoded in enumerate(container.decode(stream)):
                    decoded_count = frame_idx + 1
                    target = wanted.get(frame_idx)
                    if target is None:
                        continue
                    shot_index, window_id = target
                    frame = decoded.to_ndarray(format="bgr24")
                    thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                    thumbs[shot_index].append(thumb)
                    quality, metrics = frame_quality(frame)
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok:
                        continue
                    record = {
                        "frame_idx": frame_idx,
                        "pts": decoded.pts,
                        "pts_time": float(decoded.pts * decoded.time_base) if decoded.pts is not None and decoded.time_base is not None else frame_idx / fps,
                        "time_base": str(decoded.time_base or item.time_base),
                        "window_id": window_id,
                        "quality": quality,
                        "metrics": metrics,
                        "encoded": encoded,
                        "quality_fallback": False,
                    }
                    if best[shot_index] is None or quality > best[shot_index]["quality"]:
                        best[shot_index] = record
                    eligible = metrics["black_ratio"] <= 0.60 and metrics["white_ratio"] <= 0.60 and metrics["sharpness"] >= 25.0
                    duplicate = any(hamming_distance(int(metrics["phash"]), value) < 6 for value in hashes[shot_index])
                    if not eligible or duplicate:
                        continue
                    hashes[shot_index].append(int(metrics["phash"]))
                    accepted[shot_index].append(record)
                    encode_records.append(record)
                    encode_frames.append(frame)
                    if len(encode_frames) >= 128:
                        flush_encode()
            flush_encode()

            # Encode one quality fallback only for shots whose filtered pool is empty.
            for shot_index in range(len(scenes)):
                if accepted[shot_index]:
                    continue
                fallback = best[shot_index]
                if fallback is None:
                    raise RuntimeError(f"No decoded candidate for shot {shot_index}")
                fallback["quality_fallback"] = True
                frame = cv2.imdecode(fallback["encoded"], cv2.IMREAD_COLOR)
                accepted[shot_index] = [fallback]
                encode_records.append(fallback)
                encode_frames.append(frame)
            flush_encode()

            keyframe_rows: list[dict] = []
            vector_by_keyframe: dict[str, np.ndarray] = {}
            ordinal = 0
            for shot_index, candidates in enumerate(accepted):
                differences = [float(np.mean(cv2.absdiff(left, right))) / 255.0 for left, right in zip(thumbs[shot_index], thumbs[shot_index][1:])]
                motion_score = float(np.mean(differences)) if differences else 0.0
                budget = keyframe_budget(float(shot_rows[shot_index]["duration_sec"]), motion_score)
                shot_rows[shot_index]["motion_score"] = motion_score
                shot_rows[shot_index]["keyframe_budget"] = budget
                for local_index in farthest_point_select(candidates, budget):
                    row = candidates[local_index]
                    ordinal += 1
                    shot_id = str(shot_rows[shot_index]["shot_id"])
                    keyframe_id = make_keyframe_id(video_id, shot_id, int(row["frame_idx"]), STAGE_VERSION)
                    frame_name = f"{video_id}_{keyframe_id.rsplit(':', 1)[-1]}.webp"
                    frame_path = frame_root / frame_name
                    frame = cv2.imdecode(row["encoded"], cv2.IMREAD_COLOR)
                    if frame is None or not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_WEBP_QUALITY, 90]):
                        raise RuntimeError(f"Cannot write {frame_path}")
                    metrics = row["metrics"]
                    keyframe_rows.append({
                        "keyframe_id": keyframe_id, "video_id": video_id, "shot_id": shot_id,
                        "ordinal": ordinal, "frame_idx": int(row["frame_idx"]), "pts": row["pts"],
                        "pts_time": float(row["pts_time"]), "time_base": row["time_base"], "fps": fps,
                        "quality_score": float(row["quality"]), "sharpness": float(metrics["sharpness"]),
                        "mean_luma": float(metrics["mean_luma"]), "black_ratio": float(metrics["black_ratio"]),
                        "white_ratio": float(metrics["white_ratio"]), "window_id": str(row["window_id"]),
                        "motion_score": motion_score, "quality_fallback": bool(row["quality_fallback"]),
                        "frame_path": f"frames/{video_id}/{frame_name}", "source": "self_extracted",
                        "pipeline_version": STAGE_VERSION, "run_id": f"{STAGE_VERSION}-{video_id.lower()}",
                        "extractor_version": "transnetv2-pyav-phash-openclip-farthest-v2",
                    })
                    vector_by_keyframe[keyframe_id] = np.asarray(row["feature"], dtype="float32")

            shots_df = pd.DataFrame(shot_rows)
            keyframes_df = pd.DataFrame(keyframe_rows).sort_values(["frame_idx", "keyframe_id"]).reset_index(drop=True)
            visual = l2_normalize(np.asarray(
                [vector_by_keyframe[str(keyframe_id)] for keyframe_id in keyframes_df["keyframe_id"]],
                dtype="float32",
            ))
            if len(keyframes_df) != len(visual):
                raise RuntimeError("Keyframe/vector count mismatch")
            records = keyframes_df[["keyframe_id", "video_id", "shot_id", "frame_idx", "pts_time", "frame_path"]].copy()
            records["embedding_row"] = np.arange(len(records), dtype="int64")
            records["embedding_model"] = "ViT-B-32/laion2b_s34b_b79k"
            records["embedding_dim"] = int(visual.shape[1])
            shot_records, shot_vectors = aggregate_embeddings(records, visual, "shot_id")
            shot_records["embedding_row"] = np.arange(len(shot_records), dtype="int64")

            shots_df.to_parquet(output / "shots.parquet", index=False)
            keyframes_df.to_parquet(output / "keyframes.parquet", index=False)
            records.to_parquet(output / "embedding-records.parquet", index=False)
            shot_records.to_parquet(output / "shot-embedding-records.parquet", index=False)
            np.save(output / "visual.npy", visual.astype("float32"))
            np.save(output / "visual.f16.npy", visual.astype("float16"))
            np.save(output / "shot-visual.npy", shot_vectors.astype("float32"))
            atomic_write_json(output / "frame-mapping-report.json", [])
            coverage = (float(keyframes_df.pts_time.max()) - float(keyframes_df.pts_time.min())) / max(float(item.duration_sec), 1e-6)
            quality_report = {
                "passed": bool(keyframes_df.keyframe_id.is_unique and set(shots_df.shot_id).issubset(set(keyframes_df.shot_id))),
                "unique_keyframe_ids": bool(keyframes_df.keyframe_id.is_unique),
                "shots_without_keyframe": int((~shots_df.shot_id.isin(keyframes_df.shot_id)).sum()),
                "temporal_coverage_ratio": coverage,
                "keyframes_per_minute": float(len(keyframes_df) / max(float(item.duration_sec) / 60.0, 1e-6)),
                "quality_fallback_count": int(keyframes_df.quality_fallback.sum()),
                "decoded_frame_count": decoded_count,
                "manifest_frame_count": int(item.frame_count or 0),
            }
            summary = {
                "schema_version": "aic-video-v2/1", "stage": "shot-keyframe-visual",
                "pipeline_version": STAGE_VERSION, "video_id": video_id, "source": "self_extracted",
                "run_id": f"{STAGE_VERSION}-{video_id.lower()}", "video_path": str(video_path),
                "fps": fps, "frame_count": decoded_count, "shots": len(shots_df), "keyframes": len(keyframes_df),
                "shot_detector": "TransNetV2" if transnet is not None else "PySceneDetect ContentDetector",
                "keyframe_selector": "quality+pHash+motion+OpenCLIP+farthest-point",
                "embedding_model": "ViT-B-32/laion2b_s34b_b79k", "embedding_dim": int(visual.shape[1]),
                "metric": "IP with L2-normalized vectors", "config_hash": cfg_hash, "quality_report": quality_report,
            }
            atomic_write_json(output / "summary.json", summary)
            atomic_write_json(output / "visual-summary.json", {
                "schema_version": "aic-video-v2/1", "stage": "visual-embedding", "pipeline_version": STAGE_VERSION,
                "video_id": video_id, "keyframes": len(records), "shots": len(shot_records),
                "embedding_shape": list(visual.shape), "model": "ViT-B-32/laion2b_s34b_b79k",
                "metric": "IP with L2-normalized vectors", "source": "reused-keyframe-selection-pass",
            })
            validate_frame_contract(output)
            write_success(output, {
                "stage": "shot-keyframe-visual", "pipeline_version": STAGE_VERSION,
                "config_hash": cfg_hash, "video_id": video_id, "shots": len(shots_df), "keyframes": len(keyframes_df),
            })
            if args.make_compat_zips:
                ensure_compat_archives(args.output_root.parent, output, video_id)
            completed += 1
            print(f"[{number}/{len(manifest)}] {video_id}: {len(shots_df)} shots, {len(keyframes_df)} keyframes")
        except Exception as exc:
            failed += 1
            (output / "_SUCCESS.json").unlink(missing_ok=True)
            (output / "_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            failures.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {video_id}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    atomic_write_json(args.output_root / "batch-summary.json", {
        "stage": "shot-keyframe-visual", "pipeline_version": STAGE_VERSION,
        "config_hash": cfg_hash, "videos": len(manifest), "completed": completed, "failed": failed, "failures": failures,
    })
    if failed:
        raise SystemExit(f"{failed} videos failed; rerun after inspecting _ERROR.txt")


if __name__ == "__main__":
    main()
