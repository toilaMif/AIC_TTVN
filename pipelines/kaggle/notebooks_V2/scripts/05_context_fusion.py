"""Stage 05: timestamp-aware frame -> shot -> event search documents.

The default path is evidence-first and fast. An optional Qwen2.5-VL hook can
caption only shot representatives/evidence changes; propagated frame records
never copy a detailed caption blindly and carry an explicit provenance flag.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    atomic_write_json,
    build_events,
    build_temporal_edges,
    config_hash,
    discover_video_parquets,
    interval_join_asr,
    join_unique,
    normalize_text,
    read_video_rows,
    success_matches,
    write_parquet_atomic,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-fusion"
SCENE_CUES = {
    "indoor": ("trong nhà", "phòng", "bệnh viện", "văn phòng", "studio"),
    "outdoor": ("ngoài trời", "đường phố", "cánh đồng", "bờ sông", "biển", "công viên"),
    "night": ("ban đêm", "ban tối", "trời tối"),
    "daytime": ("ban ngày", "buổi sáng", "buổi trưa", "trời sáng"),
    "rain": ("trời mưa", "đang mưa", "mưa lớn"),
    "sunny": ("trời nắng", "ánh nắng", "trời sáng"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(os.getenv("AIC_V2_INPUT_ROOT", "/kaggle/input")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "06-frame-understanding")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--enable-vlm", action=argparse.BooleanOptionalAction, default=os.getenv("AIC_V2_ENABLE_VLM", "0") == "1")
    parser.add_argument("--event-window-sec", type=float, default=30.0)
    parser.add_argument("--make-compat-zips", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_counts(value) -> dict[str, int]:
    if isinstance(value, dict):
        decoded = value
    elif isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    if not isinstance(decoded, dict):
        return {}
    output: dict[str, int] = {}
    for key, count in decoded.items():
        try:
            output[str(key)] = max(0, int(count))
        except (TypeError, ValueError):
            continue
    return output


def aggregate_counts(values) -> dict[str, int]:
    """Use temporal max counts so repeated keyframes do not double-count."""
    output: dict[str, int] = {}
    for value in values:
        for label, count in parse_counts(value).items():
            output[label] = max(output.get(label, 0), count)
    return output


def labels_from_text(value: str) -> list[str]:
    text = normalize_text(value).casefold()
    labels = []
    for label, cues in SCENE_CUES.items():
        if any(cue.casefold() in text for cue in cues):
            labels.append(label)
    return labels


def evidence_caption(ocr: str, asr: str, objects: list[str], scene_labels: list[str]) -> str:
    parts = []
    if objects:
        parts.append("có " + ", ".join(objects[:8]))
    if scene_labels:
        parts.append("bối cảnh " + ", ".join(scene_labels))
    if ocr:
        parts.append("chữ nhìn thấy: " + ocr[:240])
    if asr:
        parts.append("lời thoại: " + asr[:240])
    return join_unique(parts, limit=4)


class OptionalVLM:
    """Small lazy adapter; no Qwen dependency is imported unless enabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.model = None
        self.processor = None
        self.device = "cuda"
        self.error: str | None = None
        if not enabled:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            name = os.getenv("AIC_V2_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.processor = AutoProcessor.from_pretrained(name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
            self.model.eval()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.enabled = False

    def caption(self, image_path: Path, evidence: str, memory: str) -> tuple[str, str]:
        if not self.enabled or self.model is None or self.processor is None:
            return "", ""
        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            prompt = (
                "Return JSON only with keys caption_vi_short, entities_vi, actions_vi, "
                "environment, time_of_day, weather, confidence. Vietnamese video search. "
                f"Evidence: {evidence[:1200]} Previous memory: {memory[:600]}"
            )
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=192, do_sample=False, use_cache=True)
            decoded = self.processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
            start, end = decoded.find("{"), decoded.rfind("}")
            if start >= 0 and end > start:
                result = json.loads(decoded[start : end + 1])
                short = normalize_text(result.get("caption_vi_short", ""))
                memory_out = join_unique([short, *parse_list(result.get("entities_vi", [])), *parse_list(result.get("actions_vi", []))], limit=20)
                return short, memory_out
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return "", ""


def main() -> None:
    options = parse_args()
    config = {"stage": "fusion", "pipeline_version": STAGE_VERSION, "enable_vlm": options.enable_vlm, "event_window_sec": options.event_window_sec, "make_compat_zips": options.make_compat_zips, "max_shots": options.max_shots}
    cfg_hash = config_hash(config)
    options.output_root.mkdir(parents=True, exist_ok=True)
    frame_manifests = discover_video_parquets(options.input_root, "keyframes.parquet", {"keyframe_id", "video_id", "frame_idx", "pts_time", "frame_path"})
    selected = list(frame_manifests.items())[: options.max_videos] if options.max_videos is not None else list(frame_manifests.items())
    ocr_manifests = discover_video_parquets(options.input_root, "ocr.parquet", {"keyframe_id", "video_id"})
    object_manifests = discover_video_parquets(options.input_root, "objects.parquet", {"keyframe_id", "video_id"})
    asr_manifests = discover_video_parquets(options.input_root, "asr-segments.parquet", {"segment_id", "video_id", "start_time", "end_time", "text"})
    vlm = OptionalVLM(options.enable_vlm)
    failures: list[dict[str, str]] = []
    completed = 0
    for video_id, keyframe_path in selected:
        output = options.output_root / video_id
        if success_matches(output, stage="fusion", version=STAGE_VERSION, config=config):
            completed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        try:
            keyframes = read_video_rows(keyframe_path, video_id).sort_values("frame_idx").reset_index(drop=True)
            shots_path = keyframe_path.parent / "shots.parquet"
            if not shots_path.exists():
                candidates = sorted(keyframe_path.parent.rglob("shots.parquet"))
                shots_path = candidates[0] if candidates else shots_path
            shots = read_video_rows(shots_path, video_id) if shots_path.exists() else keyframes[["shot_id", "video_id"]].drop_duplicates().assign(shot_index=lambda frame: range(len(frame)), start_frame=0, end_frame=0, start_time=keyframes.pts_time.min(), end_time=keyframes.pts_time.max(), duration_sec=0.0)
            if options.max_shots is not None:
                allowed = set(shots.sort_values("shot_index").head(max(0, options.max_shots)).shot_id.astype(str))
                keyframes = keyframes[keyframes.shot_id.astype(str).isin(allowed)].copy()
                shots = shots[shots.shot_id.astype(str).isin(allowed)].copy()

            ocr = read_video_rows(ocr_manifests[video_id], video_id) if video_id in ocr_manifests else pd.DataFrame()
            objects = read_video_rows(object_manifests[video_id], video_id) if video_id in object_manifests else pd.DataFrame()
            asr = read_video_rows(asr_manifests[video_id], video_id) if video_id in asr_manifests else pd.DataFrame()
            ocr_by_id = {str(row.keyframe_id): row for row in ocr.itertuples(index=False)}
            objects_by_id: defaultdict[str, list] = defaultdict(list)
            for row in objects.itertuples(index=False):
                objects_by_id[str(row.keyframe_id)].append(row)
            asr_text = interval_join_asr(keyframes, asr)
            frame_rows: list[dict] = []
            temporal_memory = ""
            vlm_signatures: dict[str, tuple] = {}
            vlm_calls_by_shot: defaultdict[str, int] = defaultdict(int)
            selection_rows: list[dict] = []
            for position, row in enumerate(keyframes.itertuples(index=False)):
                keyframe_id = str(row.keyframe_id)
                ocr_row = ocr_by_id.get(keyframe_id)
                scene_text = normalize_text(getattr(ocr_row, "search_text", "") if ocr_row else getattr(ocr_row, "text", "") if ocr_row else "")
                ticker_text = normalize_text(getattr(ocr_row, "text", "") if ocr_row else "")
                detections = objects_by_id.get(keyframe_id, [])
                labels = []
                counts: dict[str, int] = {}
                for detection in detections:
                    label = normalize_text(getattr(detection, "class_name", ""))
                    if label:
                        labels.append(label)
                        counts[label] = counts.get(label, 0) + 1
                speech = normalize_text(asr_text.iloc[position]) if position < len(asr_text) else ""
                evidence = " ".join([scene_text, speech, " ".join(labels)])
                scene_labels = labels_from_text(" ".join([scene_text, speech]))
                shot_id = str(getattr(row, "shot_id", ""))
                direct = False
                caption = evidence_caption(scene_text, speech, list(dict.fromkeys(labels)), scene_labels)
                signature = (
                    tuple(sorted(set(labels))),
                    tuple(sorted(set(re.findall(r"[\w]+", scene_text.casefold())))),
                    tuple(sorted(set(re.findall(r"[\w]+", speech.casefold()))))[:24],
                )
                signature_changed = vlm_signatures.get(shot_id) != signature
                should_call_vlm = options.enable_vlm and (
                    vlm_calls_by_shot[shot_id] == 0
                    or (signature_changed and vlm_calls_by_shot[shot_id] < 2)
                )
                if should_call_vlm:
                    image_path = keyframe_path.parent / str(row.frame_path)
                    if not image_path.exists():
                        candidates = sorted(keyframe_path.parent.rglob(Path(str(row.frame_path)).name))
                        image_path = candidates[0] if candidates else image_path
                    vlm_caption, memory = vlm.caption(image_path, evidence, temporal_memory) if image_path.exists() else ("", "")
                    vlm_signatures[shot_id] = signature
                    vlm_calls_by_shot[shot_id] += 1
                    selection_rows.append({
                        "keyframe_id": keyframe_id,
                        "video_id": video_id,
                        "shot_id": shot_id,
                        "frame_idx": int(row.frame_idx),
                        "reason": "shot_representative" if vlm_calls_by_shot[shot_id] == 1 else "evidence_change",
                        "signature_changed": signature_changed,
                    })
                    if vlm_caption:
                        caption = vlm_caption
                        temporal_memory = join_unique([temporal_memory, memory], limit=40)
                        direct = True
                stable_entities = join_unique([" ".join(dict.fromkeys(labels)), *scene_labels], limit=24)
                search_text = join_unique([caption, stable_entities, scene_text, ticker_text, speech], limit=80)
                frame_rows.append({
                    "doc_id": f"frame:{keyframe_id}", "level": "frame", "keyframe_id": keyframe_id,
                    "video_id": str(row.video_id), "shot_id": shot_id, "frame_idx": int(row.frame_idx), "pts_time": float(row.pts_time),
                    "start_time": float(row.pts_time), "end_time": float(row.pts_time), "frame_path": str(row.frame_path),
                    "caption_vi_short": caption, "caption_vi_detail": caption, "search_text_vi": search_text,
                    "search_text_en": "", "asr_text": speech, "ocr_text": ticker_text, "ocr_scene_text": scene_text,
                    "ocr_ticker_text": ticker_text if ticker_text != scene_text else "", "ocr_ticker_weight": 0.25,
                    "object_labels_text": " ".join(dict.fromkeys(labels)), "object_counts_json": json.dumps(counts, ensure_ascii=False),
                    "scene_labels_json": json.dumps(sorted(set(scene_labels)), ensure_ascii=False),
                    "description_source": "vlm" if direct else "evidence", "propagation_confidence": 1.0 if direct else 0.75,
                    "vlm_source_keyframe_id": keyframe_id if direct else None, "confidence": 0.85 if direct else 0.55,
                    "pipeline_version": STAGE_VERSION,
                })
            frame_docs = pd.DataFrame(frame_rows)
            shot_rows: list[dict] = []
            for shot_id, group in frame_docs.sort_values(["shot_id", "confidence"], ascending=[True, False]).groupby("shot_id", sort=False):
                representative = group.iloc[0]
                shot_rows.append({
                    "doc_id": f"shot:{shot_id}", "level": "shot", "video_id": video_id, "shot_id": str(shot_id),
                    "keyframe_id": str(representative.keyframe_id), "frame_idx": int(representative.frame_idx),
                    "pts_time": float(group.pts_time.min()), "start_time": float(group.pts_time.min()), "end_time": float(group.pts_time.max()),
                    "caption_vi_short": join_unique(group.caption_vi_short.tolist(), limit=2),
                    "caption_vi_detail": join_unique(group.caption_vi_detail.tolist(), limit=4),
                    "search_text_vi": join_unique(group.search_text_vi.tolist(), limit=100), "search_text_en": "",
                    "asr_text": join_unique(group.asr_text.tolist(), limit=40), "ocr_text": join_unique(group.ocr_text.tolist(), limit=40),
                    "ocr_scene_text": join_unique(group.ocr_scene_text.tolist(), limit=40), "ocr_ticker_text": join_unique(group.ocr_ticker_text.tolist(), limit=20),
                    "ocr_ticker_weight": 0.25, "object_labels_text": join_unique(group.object_labels_text.tolist(), limit=30),
                    "object_counts_json": json.dumps(aggregate_counts(group.object_counts_json), ensure_ascii=False), "scene_labels_json": json.dumps(sorted({label for value in group.scene_labels_json for label in parse_list(value)}), ensure_ascii=False),
                    "description_source": str(representative.description_source), "propagation_confidence": float(group.propagation_confidence.mean()),
                    "vlm_source_keyframe_id": representative.vlm_source_keyframe_id, "confidence": float(group.confidence.max()),
                    "pipeline_version": STAGE_VERSION,
                })
            shot_docs = pd.DataFrame(shot_rows)
            event_meta = build_events(shots, max_duration_sec=options.event_window_sec)
            event_rows = []
            for event in event_meta.itertuples(index=False):
                group = shot_docs[shot_docs.shot_id.astype(str).isin(parse_list(event.shot_ids_json))]
                event_rows.append({
                    "doc_id": f"event:{event.event_id}", "level": "event", "event_id": event.event_id, "video_id": video_id,
                    "shot_id": event.start_shot_id, "keyframe_id": group.iloc[0].keyframe_id if not group.empty else None,
                    "frame_idx": int(group.iloc[0].frame_idx) if not group.empty else 0, "pts_time": float(event.start_time),
                    "start_time": float(event.start_time), "end_time": float(event.end_time),
                    "caption_vi_short": join_unique(group.caption_vi_short.tolist(), limit=4), "caption_vi_detail": join_unique(group.caption_vi_detail.tolist(), limit=8),
                    "search_text_vi": join_unique(group.search_text_vi.tolist(), limit=150), "search_text_en": "",
                    "asr_text": join_unique(group.asr_text.tolist(), limit=80), "ocr_text": join_unique(group.ocr_text.tolist(), limit=80),
                    "ocr_scene_text": join_unique(group.ocr_scene_text.tolist(), limit=80), "ocr_ticker_text": join_unique(group.ocr_ticker_text.tolist(), limit=30),
                    "ocr_ticker_weight": 0.25, "object_labels_text": join_unique(group.object_labels_text.tolist(), limit=50),
                    "object_counts_json": json.dumps(aggregate_counts(group.object_counts_json), ensure_ascii=False), "scene_labels_json": json.dumps(sorted({label for value in group.scene_labels_json for label in parse_list(value)}), ensure_ascii=False),
                    "description_source": "temporal-aggregate", "propagation_confidence": float(group.propagation_confidence.mean()) if not group.empty else 0.0,
                    "vlm_source_keyframe_id": None, "confidence": float(group.confidence.max()) if not group.empty else 0.0,
                    "shot_ids_json": event.shot_ids_json, "pipeline_version": STAGE_VERSION,
                })
            event_docs = pd.DataFrame(event_rows)
            edges = build_temporal_edges(shots)
            vlm_selection = pd.DataFrame(selection_rows, columns=["keyframe_id", "video_id", "shot_id", "frame_idx", "reason", "signature_changed"])
            legacy_understanding = frame_docs.rename(columns={
                "caption_vi_short": "caption_vi",
            }).copy()
            legacy_shots = shot_docs.rename(columns={
                "caption_vi_short": "caption_vi",
            }).copy()
            for name, frame in (
                ("frame-docs.parquet", frame_docs),
                ("shot-docs.parquet", shot_docs),
                ("event-docs.parquet", event_docs),
                ("search-documents.parquet", frame_docs),
                ("temporal-edges.parquet", edges),
                ("vlm-selection.parquet", vlm_selection),
                ("frame-understanding.parquet", legacy_understanding),
                ("shot-scenes.parquet", legacy_shots),
            ):
                write_parquet_atomic(frame, output / name)
            frame_docs.to_json(output / "search-documents.jsonl", orient="records", lines=True, force_ascii=False)
            legacy_understanding.to_json(output / "frame-understanding.jsonl", orient="records", lines=True, force_ascii=False)
            facets = defaultdict(int)
            for value in frame_docs.scene_labels_json:
                for label in parse_list(value):
                    facets[label] += 1
            atomic_write_json(output / "scene-facets.json", {"all_option": {"value": "all", "label": "All scenes"}, "options": [{"value": key, "label": key, "count": count} for key, count in sorted(facets.items())]})
            summary = {
                "schema_version": "aic-video-v2/1", "stage": "fusion", "pipeline_version": STAGE_VERSION,
                "config_hash": cfg_hash, "video_id": video_id, "frame_docs": len(frame_docs), "shot_docs": len(shot_docs),
                "event_docs": len(event_docs), "temporal_edges": len(edges), "vlm_enabled": options.enable_vlm,
                "vlm_loaded": bool(vlm.enabled), "vlm_error": vlm.error, "join_policy": "keyframe_id+timestamp",
            }
            atomic_write_json(output / "fusion-summary.json", summary)
            atomic_write_json(output / "frame-understanding-summary.json", summary)
            write_success(output, {"stage": "fusion", "pipeline_version": STAGE_VERSION, "config_hash": cfg_hash, "video_id": video_id, "frame_docs": len(frame_docs), "shot_docs": len(shot_docs), "event_docs": len(event_docs)})
            if options.make_compat_zips:
                archive_path = options.output_root / f"06-frame-understanding-{video_id}.zip"
                temporary = archive_path.with_suffix(".tmp.zip")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
                    for name in ("frame-docs.parquet", "shot-docs.parquet", "event-docs.parquet", "frame-understanding.parquet", "frame-understanding.jsonl", "search-documents.parquet", "search-documents.jsonl", "shot-scenes.parquet", "scene-facets.json", "vlm-selection.parquet", "temporal-edges.parquet", "fusion-summary.json", "frame-understanding-summary.json", "_SUCCESS.json"):
                        path = output / name
                        if path.is_file():
                            archive.write(path, name)
                temporary.replace(archive_path)
            completed += 1
            print(f"{video_id}: frame={len(frame_docs)} shot={len(shot_docs)} event={len(event_docs)}")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
            (output / "_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            gc.collect()
    atomic_write_json(options.output_root / "batch-summary.json", {"stage": "fusion", "pipeline_version": STAGE_VERSION, "completed": completed, "failed": len(failures), "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} fusion videos failed")


if __name__ == "__main__":
    main()
