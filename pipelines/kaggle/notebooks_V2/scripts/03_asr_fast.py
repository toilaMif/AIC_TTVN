"""Stage 03: cached-audio Vietnamese ASR with segment-level resume."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    append_jsonl,
    atomic_write_json,
    config_hash,
    discover_video_parquets,
    extract_audio,
    read_jsonl_latest,
    read_video_rows,
    success_matches,
    stable_id,
    write_parquet_atomic,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-asr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "04-asr")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--model", default=os.getenv("AIC_V2_ASR_MODEL", "large-v3"))
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--make-compat-zips", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    config = {"stage": "asr", "pipeline_version": STAGE_VERSION, "model": options.model, "beam_size": options.beam_size, "compute_type": "int8_float16", "sample_rate": 16_000, "make_compat_zips": options.make_compat_zips}
    cfg_hash = config_hash(config)
    options.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = options.manifest
    if manifest_path is None:
        candidates = sorted(options.input_root.rglob("video-manifest.parquet"))
        if not candidates:
            raise FileNotFoundError("Attach stage 00 video-manifest.parquet")
        manifest_path = candidates[0]
    manifest = pd.read_parquet(manifest_path).sort_values("video_id").reset_index(drop=True)
    if options.max_videos is not None:
        manifest = manifest.head(max(0, options.max_videos))

    from faster_whisper import WhisperModel

    model = WhisperModel(options.model, device="cuda", compute_type="int8_float16")
    failures: list[dict[str, str]] = []
    completed = 0
    for item in manifest.itertuples(index=False):
        video_id = str(item.video_id)
        output = options.output_root / video_id
        if success_matches(output, stage="asr", version=STAGE_VERSION, config=config):
            completed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        try:
            audio_value = getattr(item, "audio_path", None)
            audio = None
            if audio_value:
                candidate = Path(str(audio_value))
                if not candidate.is_absolute():
                    candidate = manifest_path.parent / candidate
                if candidate.exists():
                    audio = candidate
            if audio is None:
                source = Path(str(item.path))
                if not source.exists():
                    matches = sorted(options.input_root.rglob(f"{video_id}.mp4"))
                    source = matches[0] if matches else source
                audio = extract_audio(source, output / "audio" / f"{video_id}.wav")
            checkpoint = output / "asr-checkpoint.jsonl"
            latest = read_jsonl_latest(checkpoint, "segment_id")
            segments, info = model.transcribe(
                str(audio), language="vi", beam_size=options.beam_size, vad_filter=True,
                word_timestamps=True, condition_on_previous_text=False,
            )
            rows = []
            for index, segment in enumerate(segments):
                start = float(segment.start)
                end = float(segment.end)
                segment_id = f"{video_id}:asr:{stable_id(video_id, round(start, 3), round(end, 3), segment.text)}"
                item_row = {
                    "segment_id": segment_id, "video_id": video_id, "segment_index": index,
                    "start_time": start, "end_time": end, "text": str(segment.text).strip(),
                    "language": "vi", "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                    "words_json": json.dumps([
                        {"start": float(word.start), "end": float(word.end), "word": word.word, "probability": float(getattr(word, "probability", 0.0) or 0.0)}
                        for word in (getattr(segment, "words", None) or [])
                    ], ensure_ascii=False),
                    "model_name": options.model, "pipeline_version": STAGE_VERSION,
                }
                if segment_id not in latest:
                    rows.append(item_row)
                latest[segment_id] = item_row
            if rows:
                append_jsonl(checkpoint, rows)
            ordered = sorted(latest.values(), key=lambda row: (row["start_time"], row["segment_index"]))
            frame = pd.DataFrame(ordered)
            write_parquet_atomic(frame, output / "asr-segments.parquet")
            frame.to_json(output / "asr-segments.jsonl", orient="records", lines=True, force_ascii=False)
            (output / "transcript.txt").write_text(" ".join(str(row["text"]) for row in ordered).strip(), encoding="utf-8")
            summary = {
                "schema_version": "aic-video-v2/1", "stage": "asr", "pipeline_version": STAGE_VERSION,
                "config_hash": cfg_hash, "video_id": video_id, "segments": len(frame), "model": options.model,
                "language": "vi", "audio_sample_rate": 16_000, "resume_key": "segment_id",
            }
            atomic_write_json(output / "asr-summary.json", summary)
            write_success(output, {"stage": "asr", "pipeline_version": STAGE_VERSION, "config_hash": cfg_hash, "video_id": video_id, "segments": len(frame)})
            if options.make_compat_zips:
                archive_path = options.output_root / f"04-asr-vietnamese-{video_id}.zip"
                temporary = archive_path.with_suffix(".tmp.zip")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
                    for name in ("asr-segments.parquet", "asr-segments.jsonl", "transcript.txt", "asr-summary.json", "_SUCCESS.json"):
                        path = output / name
                        if path.is_file():
                            archive.write(path, name)
                temporary.replace(archive_path)
            completed += 1
            print(f"{video_id}: {len(frame)} ASR segments")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
            (output / "_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            gc.collect()
    atomic_write_json(options.output_root / "batch-summary.json", {"stage": "asr", "pipeline_version": STAGE_VERSION, "completed": completed, "failed": len(failures), "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} ASR videos failed")


if __name__ == "__main__":
    main()
