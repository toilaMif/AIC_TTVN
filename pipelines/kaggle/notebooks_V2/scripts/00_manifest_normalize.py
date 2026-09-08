"""Stage 00: deterministic video manifest and cached 16 kHz mono audio.

This stage is deliberately CPU-safe. It is useful to run once and attach as a
small Kaggle dataset for all later notebooks.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    VideoInfo,
    atomic_write_json,
    config_hash,
    discover_videos,
    extract_audio,
    manifest_dataframe,
    probe_video,
    sha256_file,
    write_success,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "00-manifest")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--extract-audio", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "manifest",
        "input_root": str(args.input_root.resolve()),
        "extract_audio": bool(args.extract_audio),
        "audio_sample_rate": 16_000,
        "pipeline_version": PIPELINE_VERSION,
        "max_videos": args.max_videos,
    }
    from v2_runtime import success_matches
    if success_matches(output_root, stage="manifest", version=PIPELINE_VERSION, config=config):
        print(f"Manifest already exists: {output_root}")
        return

    paths = discover_videos(args.input_root)
    if args.max_videos is not None:
        paths = paths[: max(0, args.max_videos)]
    if not paths:
        raise FileNotFoundError(f"No MP4 files found under {args.input_root}")

    infos: list[VideoInfo] = []
    failures: list[dict[str, str]] = []
    audio_root = output_root / "audio"
    for index, path in enumerate(paths, start=1):
        try:
            info = probe_video(path)
            audio_path = None
            if args.extract_audio:
                destination = audio_root / f"{info.video_id}.wav"
                extract_audio(path, destination)
                # Store a portable path. The absolute `/kaggle/working` path is
                # invalid after the manifest is attached as a Kaggle dataset.
                audio_path = str(destination.relative_to(output_root)).replace("\\", "/")
            info = VideoInfo(**{**info.__dict__, "audio_path": audio_path})
            infos.append(info)
            print(f"[{index}/{len(paths)}] {info.video_id}: {info.duration_sec:.1f}s, {info.fps:.3f} fps")
        except Exception as exc:  # keep other videos resumable
            failures.append({"video_id": path.stem, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            (output_root / f"{path.stem}._ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")

    if not infos:
        raise RuntimeError(f"All videos failed: {failures[:2]}")
    frame = manifest_dataframe(infos)
    frame["manifest_key"] = frame.apply(lambda row: f"{row.video_id}:{row.content_sha256[:16]}", axis=1)
    frame["audio_sha256"] = frame["audio_path"].map(
        lambda value: sha256_file(output_root / str(value))
        if value and (output_root / str(value)).exists()
        else None
    )
    frame.to_parquet(output_root / "video-manifest.parquet", index=False)
    frame.to_json(output_root / "video-manifest.jsonl", orient="records", lines=True, force_ascii=False)
    atomic_write_json(output_root / "run.json", {
        "schema_version": SCHEMA_VERSION,
        "stage": "manifest",
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash(config),
        "input_root": str(args.input_root.resolve()),
        "videos_found": len(paths),
        "videos_ok": len(frame),
        "videos_failed": len(failures),
        "failures": failures,
    })
    write_success(output_root, {
        "stage": "manifest",
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash(config),
        "videos": len(frame),
        "manifest": "video-manifest.parquet",
    })
    print(f"Wrote {len(frame)} videos to {output_root}")
    if failures:
        print(f"Warnings: {len(failures)} videos failed; inspect *_ERROR.txt")


if __name__ == "__main__":
    main()
