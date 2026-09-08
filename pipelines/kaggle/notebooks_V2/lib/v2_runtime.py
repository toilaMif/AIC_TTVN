"""Shared, dependency-light runtime helpers for the AIC Kaggle V2 pipeline.

The notebooks deliberately keep model-specific code in their own cells, while
this module owns the parts that must be identical across stages: deterministic
IDs, manifests, atomic checkpoints, path resolution, temporal joins, validation,
and retrieval-pack construction.  It is safe to import on a CPU-only machine;
heavy GPU libraries are imported lazily by the notebooks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "aic-video-v2/1"
PIPELINE_VERSION = "aicv4-v2"
DEFAULT_ROOT = Path("/kaggle/working/aic-v2")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))[:16]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append a checkpoint shard without rewriting prior records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def read_jsonl_latest(path: Path, key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(key) is not None:
                latest[str(row[key])] = row
    return latest


def write_success(root: Path, payload: Mapping[str, Any]) -> None:
    result = {"schema_version": SCHEMA_VERSION, "success": True, **dict(payload)}
    atomic_write_json(root / "_SUCCESS.json", result)
    (root / "_ERROR.txt").unlink(missing_ok=True)


def success_matches(root: Path, *, stage: str, version: str = PIPELINE_VERSION, config: Mapping[str, Any] | None = None) -> bool:
    payload = read_json(root / "_SUCCESS.json", {})
    if not payload.get("success") or payload.get("stage") != stage:
        return False
    if payload.get("pipeline_version") != version:
        return False
    if config is not None and payload.get("config_hash") != config_hash(config):
        return False
    return True


def stable_id(*parts: object) -> str:
    """Stable compact ID; unlike row order it survives input reordering."""
    value = "|".join(str(part) for part in parts)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def make_shot_id(video_id: str, shot_index: int, version: str = PIPELINE_VERSION) -> str:
    # PostgreSQL/Milvus keep IDs in VARCHAR(64). A full model/config version in
    # the identifier would overflow for the longest (32-char) video IDs; the
    # complete version remains in every artifact's `pipeline_version` field.
    return f"{video_id}:self:v2:shot:{shot_index:06d}"


def make_keyframe_id(video_id: str, shot_id: str, frame_idx: int, version: str = PIPELINE_VERSION) -> str:
    # Include the source frame, not just an ordinal, so reruns with a changed
    # candidate budget cannot silently point old IDs at new images.
    return f"{video_id}:self:v2:kf:{stable_id(video_id, shot_id, frame_idx)}"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokenize(value: Any) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(normalize_text(value)) if token]


def join_unique(parts: Iterable[Any], limit: int | None = None) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for part in parts:
        text = normalize_text(part)
        marker = text.casefold()
        if text and marker not in seen:
            output.append(text)
            seen.add(marker)
            if limit is not None and len(output) >= limit:
                break
    return " ".join(output)


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    time_base: str
    audio_path: str | None = None
    content_sha256: str | None = None


def discover_videos(input_root: Path) -> list[Path]:
    # Sort by relative path and deduplicate stems deterministically.  The
    # manifest is the only source used by later stages, so they never rglob.
    paths = sorted((path for path in input_root.rglob("*.mp4") if path.is_file()), key=lambda p: p.as_posix().casefold())
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        video_id = path.stem.casefold()
        if video_id in seen:
            continue
        seen.add(video_id)
        result.append(path)
    return result


def probe_video(path: Path) -> VideoInfo:
    """Probe with PyAV when available, then OpenCV as a conservative fallback."""
    fps = 0.0
    frame_count = 0
    duration = 0.0
    width = height = 0
    time_base = "1/1"
    try:
        import av  # type: ignore

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0.0)
            frame_count = int(stream.frames or 0)
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            width, height = int(stream.width or 0), int(stream.height or 0)
            time_base = str(stream.time_base or time_base)
    except Exception:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps
    if fps <= 0 or duration <= 0:
        raise ValueError(f"Unable to read video timing: {path}")
    return VideoInfo(
        video_id=path.stem,
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        width=width,
        height=height,
        time_base=time_base,
        content_sha256=sha256_file(path),
    )


def extract_audio(video: Path, destination: Path, sample_rate: int = 16_000) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 44:
        return destination
    # Keep a real audio suffix so ffmpeg can select the muxer on Kaggle images
    # that do not honor `-f` inference for a `.tmp` filename.
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.wav")
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(destination)
    return destination


def manifest_dataframe(infos: Sequence[VideoInfo]) -> pd.DataFrame:
    return pd.DataFrame([asdict(info) for info in infos]).sort_values("video_id").reset_index(drop=True)


def resolve_manifest_path(value: str | Path, manifest_root: Path, path_map: Mapping[str, str] | None = None) -> Path:
    raw = str(value)
    if path_map and raw in path_map:
        return Path(path_map[raw])
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (manifest_root, manifest_root.parent, Path("/kaggle/input"), Path("/kaggle/working")):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(raw)


def discover_video_parquets(
    input_root: Path,
    filename: str,
    required_columns: set[str],
    preferred_pipeline_prefix: str = PIPELINE_VERSION,
) -> dict[str, Path]:
    """Choose one manifest per video by schema and pipeline version.

    Only the identifying columns are read. This prevents the V1 pattern of
    loading every candidate Parquet in full merely to choose an input.
    """
    choices: defaultdict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    roots = [input_root]
    # In a single Kaggle session the previous stage is commonly in /working;
    # when outputs are attached as datasets it remains under /input. Supporting
    # both keeps the notebook order ergonomic without falling back to basename
    # matching.
    if input_root == Path("/kaggle/input"):
        working_root = Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2"))
        if working_root not in roots:
            roots.append(working_root)
    columns = sorted(required_columns | {"video_id", "pipeline_version"})
    paths = {
        path.resolve()
        for root in roots
        if root.exists()
        for path in root.rglob(filename)
        if path.is_file()
    }
    for path in sorted(paths, key=lambda value: value.as_posix().casefold()):
        try:
            import pyarrow.parquet as pq

            names = set(pq.ParquetFile(path).schema.names)
            if not required_columns.issubset(names) or "video_id" not in names:
                continue
            selected = [name for name in columns if name in names]
            frame = pd.read_parquet(path, columns=selected)
        except Exception:
            continue
        for video_id in frame["video_id"].dropna().astype(str).unique():
            subset = frame[frame["video_id"].astype(str) == video_id]
            versions = subset.get("pipeline_version", pd.Series(dtype=str)).dropna().astype(str)
            preferred = int(any(value.startswith(preferred_pipeline_prefix) for value in versions))
            # Prefer V2, then a path containing the video ID, then lexical order.
            score = 10 * preferred + int(video_id.casefold() in path.as_posix().casefold())
            choices[video_id].append((score, path.as_posix().casefold(), path))
    return {
        video_id: sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]
        for video_id, candidates in sorted(choices.items())
    }


def read_video_rows(path: Path, video_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "video_id" in frame.columns:
        frame = frame[frame["video_id"].astype(str) == str(video_id)].copy()
    return frame.reset_index(drop=True)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def aggregate_embeddings(records: pd.DataFrame, vectors: np.ndarray, group_column: str = "shot_id") -> tuple[pd.DataFrame, np.ndarray]:
    if len(records) != len(vectors):
        raise ValueError("embedding rows do not align with records")
    groups: list[dict[str, Any]] = []
    output: list[np.ndarray] = []
    for group_id, indexes in records.groupby(group_column, sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype="int64")
        vector = l2_normalize(np.asarray(vectors[positions].mean(axis=0, keepdims=True)))[0]
        first = records.iloc[int(positions[0])]
        groups.append({
            group_column: str(group_id),
            "video_id": str(first.video_id),
            "start_time": float(records.iloc[positions].pts_time.min()),
            "end_time": float(records.iloc[positions].pts_time.max()),
            "keyframe_count": int(len(positions)),
        })
        output.append(vector)
    return pd.DataFrame(groups), np.asarray(output, dtype="float32")


def interval_join_asr(keyframes: pd.DataFrame, asr: pd.DataFrame, max_chars: int = 1200) -> pd.Series:
    """Join ASR intervals with a per-video sweep.

    Segment boundaries are expanded by exactly +/-1.5 seconds, matching the
    previous behavior. The active interval set is advanced once as keyframes
    move forward; work is O((K + S) log S + M), where M is the number of texts
    actually emitted (rather than scanning every segment for every keyframe).
    """
    if keyframes.empty or asr.empty or max_chars <= 0:
        return pd.Series([""] * len(keyframes), index=keyframes.index, dtype="object")

    output = [""] * len(keyframes)
    keyframe_work = keyframes.assign(
        join_position=np.arange(len(keyframes), dtype="int64"),
        join_video_id=keyframes["video_id"].astype(str),
    )
    asr_work = asr.assign(join_video_id=asr["video_id"].astype(str))
    keyframe_groups = keyframe_work.groupby("join_video_id", sort=False)
    asr_groups = asr_work.groupby("join_video_id", sort=False)
    for video_value, keyframe_group in keyframe_groups:
        try:
            segment_group = asr_groups.get_group(video_value)
        except KeyError:
            continue
        segments: list[tuple[float, float, int, str]] = []
        for segment_index, row in enumerate(segment_group.itertuples(index=False)):
            try:
                start = float(getattr(row, "start_time"))
                end = float(getattr(row, "end_time"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            if end < start:
                start, end = end, start
            text = normalize_text(getattr(row, "text", ""))
            if text:
                segments.append((start - 1.5, end + 1.5, segment_index, text))
        segments.sort(key=lambda item: (item[0], item[1], item[2]))
        ordered_keyframes = sorted(
            keyframe_group.itertuples(index=False),
            key=lambda row: float(getattr(row, "pts_time", 0.0) or 0.0),
        )
        next_segment = 0
        expiry: list[tuple[float, int]] = []
        active_keys: dict[int, str] = {}
        # Keep one current (earliest) active segment per normalized text in a
        # global heap. Per-text heaps plus lazy deletion make updates O(log S),
        # while query assembly visits unique texts rather than every segment.
        members: defaultdict[str, dict[int, tuple[float, str]]] = defaultdict(dict)
        member_heaps: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
        first_heap: list[tuple[float, str, int]] = []
        current_heads: dict[str, tuple[float, int]] = {}

        def clean_text_key(key: str) -> tuple[float, int] | None:
            heap = member_heaps.get(key)
            bucket = members.get(key)
            if not heap or not bucket:
                current_heads.pop(key, None)
                return None
            while heap and heap[0][1] not in bucket:
                heapq.heappop(heap)
            if not heap:
                member_heaps.pop(key, None)
                members.pop(key, None)
                current_heads.pop(key, None)
                return None
            head = heap[0]
            if current_heads.get(key) != head:
                current_heads[key] = head
                heapq.heappush(first_heap, (head[0], key, head[1]))
            return head

        def current_first() -> tuple[float, str, int] | None:
            while first_heap:
                start, key, segment_id = first_heap[0]
                current = clean_text_key(key)
                if current is None or current != (start, segment_id):
                    heapq.heappop(first_heap)
                    continue
                if segment_id not in members[key]:
                    heapq.heappop(first_heap)
                    continue
                return first_heap[0]
            return None

        for row in ordered_keyframes:
            timestamp = float(getattr(row, "pts_time", 0.0) or 0.0)
            while next_segment < len(segments) and segments[next_segment][0] <= timestamp:
                expanded_start, expanded_end, segment_index, text = segments[next_segment]
                key = text.casefold()
                active_keys[segment_index] = key
                members[key][segment_index] = (expanded_start, text)
                heapq.heappush(member_heaps[key], (expanded_start, segment_index))
                clean_text_key(key)
                heapq.heappush(expiry, (expanded_end, segment_index))
                next_segment += 1
            while expiry and expiry[0][0] < timestamp:
                _, segment_index = heapq.heappop(expiry)
                key = active_keys.pop(segment_index, None)
                if key is not None:
                    members[key].pop(segment_index, None)
                    clean_text_key(key)

            # Pop valid per-text heads while building the bounded result, then
            # restore them for the next keyframe. Stale heap entries are
            # discarded by `current_first`.
            held: list[tuple[float, str, int]] = []
            seen: set[str] = set()
            parts: list[str] = []
            remaining = max(0, int(max_chars))
            while remaining > 0:
                candidate = current_first()
                if candidate is None:
                    break
                start, key, segment_id = heapq.heappop(first_heap)
                if current_heads.get(key) != (start, segment_id) or segment_id not in members.get(key, {}):
                    continue
                held.append((start, key, segment_id))
                if key in seen:
                    continue
                seen.add(key)
                text = members[key][segment_id][1]
                if parts:
                    remaining -= 1
                    if remaining <= 0:
                        break
                if len(text) > remaining:
                    parts.append(text[:remaining].rstrip())
                    remaining = 0
                    break
                parts.append(text)
                remaining -= len(text)
            for start, key, segment_id in held:
                if current_heads.get(key) == (start, segment_id) and segment_id in members.get(key, {}):
                    heapq.heappush(first_heap, (start, key, segment_id))
            joined = " ".join(parts)
            output[int(getattr(row, "join_position"))] = joined
    return pd.Series(output, index=keyframes.index, dtype="object")


def build_temporal_edges(shots: pd.DataFrame, max_gap_sec: float = 3.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video_id, group in shots.sort_values(["video_id", "shot_index"]).groupby("video_id", sort=False):
        ordered = list(group.itertuples(index=False))
        for index, row in enumerate(ordered):
            for next_index in (index - 1, index + 1):
                if not (0 <= next_index < len(ordered)):
                    continue
                other = ordered[next_index]
                if next_index > index:
                    gap = max(0.0, float(other.start_time) - float(row.end_time))
                    relation = "NEXT"
                else:
                    gap = max(0.0, float(row.start_time) - float(other.end_time))
                    relation = "AFTER"
                if gap > max_gap_sec:
                    continue
                rows.append({
                    "edge_id": stable_id(row.shot_id, other.shot_id, relation),
                    "video_id": str(video_id),
                    "source_shot_id": str(row.shot_id),
                    "target_shot_id": str(other.shot_id),
                    "relation": relation,
                    "gap_sec": gap,
                })
    return pd.DataFrame(rows, columns=["edge_id", "video_id", "source_shot_id", "target_shot_id", "relation", "gap_sec"])


def build_events(shots: pd.DataFrame, max_duration_sec: float = 30.0) -> pd.DataFrame:
    """Create deterministic short event windows from adjacent shots."""
    rows: list[dict[str, Any]] = []
    for video_id, group in shots.sort_values(["video_id", "shot_index"]).groupby("video_id", sort=False):
        ordered = list(group.itertuples(index=False))
        event_index = 0
        start = 0
        while start < len(ordered):
            end = start
            while end + 1 < len(ordered) and float(ordered[end + 1].end_time) - float(ordered[start].start_time) <= max_duration_sec:
                end += 1
            first, last = ordered[start], ordered[end]
            event_id = f"{video_id}:self:{PIPELINE_VERSION}:event:{event_index:06d}"
            rows.append({
                "event_id": event_id,
                "video_id": str(video_id),
                "start_shot_id": str(first.shot_id),
                "end_shot_id": str(last.shot_id),
                "start_time": float(first.start_time),
                "end_time": float(last.end_time),
                "shot_ids_json": json.dumps([str(item.shot_id) for item in ordered[start : end + 1]], ensure_ascii=False),
                "shot_count": end - start + 1,
            })
            event_index += 1
            start = end + 1
    return pd.DataFrame(rows)


def bm25_stats(documents: pd.DataFrame, text_column: str = "search_text_vi") -> tuple[pd.DataFrame, dict[str, Any]]:
    postings: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []
    for row in documents.itertuples(index=False):
        tokens = tokenize(getattr(row, text_column, ""))
        counts = Counter(tokens)
        lengths.append(len(tokens))
        for token in counts:
            document_frequency[token] += 1
        for token, frequency in counts.items():
            postings.append({"token": token, "doc_id": str(row.doc_id), "tf": int(frequency)})
    average_length = float(np.mean(lengths)) if lengths else 0.0
    stats = {
        "documents": int(len(documents)),
        "average_document_length": average_length,
        "document_frequency": dict(document_frequency),
        "k1": 1.2,
        "b": 0.75,
        "field": text_column,
    }
    return pd.DataFrame(postings, columns=["token", "doc_id", "tf"]), stats


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            return {str(key): item for key, item in decoded.items()} if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_hybrid_snapshot(
    path: Path,
    documents_by_level: Mapping[str, pd.DataFrame],
    *,
    text_column: str = "search_text_weighted",
    temporal_edges: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build one portable SQLite snapshot for lexical and metadata retrieval.

    `documents_fts(doc_id, body)` is retained for compatibility with the V2
    benchmark. The normalized `documents` table and secondary tables make the
    same file directly queryable by level, video, time range, facets, objects,
    and temporal shot edges without scanning Parquet at request time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.sqlite")
    temporary.unlink(missing_ok=True)
    level_counts: Counter[str] = Counter()
    document_rows: list[tuple[Any, ...]] = []
    fts_rows: list[tuple[str, str]] = []
    facet_rows: list[tuple[str, str, str]] = []
    object_rows: list[tuple[str, str, int]] = []
    seen_doc_ids: set[str] = set()

    for declared_level, documents in documents_by_level.items():
        if documents is None or documents.empty:
            continue
        for row in documents.itertuples(index=False):
            payload = {column: _json_value(getattr(row, column, None)) for column in documents.columns}
            doc_id = str(payload.get("doc_id") or "")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            level = str(payload.get("level") or declared_level)
            body = normalize_text(payload.get(text_column) or payload.get("search_text_vi"))

            def text_or_none(name: str) -> str | None:
                value = payload.get(name)
                return None if _is_missing(value) else str(value)

            def float_or_none(name: str) -> float | None:
                value = payload.get(name)
                return None if _is_missing(value) else float(value)

            def int_or_none(name: str) -> int | None:
                value = payload.get(name)
                return None if _is_missing(value) else int(value)

            document_rows.append((
                doc_id,
                level,
                text_or_none("video_id"),
                text_or_none("shot_id"),
                text_or_none("keyframe_id"),
                text_or_none("event_id"),
                int_or_none("frame_idx"),
                float_or_none("pts_time"),
                float_or_none("start_time"),
                float_or_none("end_time"),
                body,
                canonical_json(payload),
            ))
            fts_rows.append((doc_id, body))
            level_counts[level] += 1
            for facet in _json_list(payload.get("scene_labels_json")):
                value = normalize_text(facet)
                if value:
                    facet_rows.append((doc_id, "scene", value))
            for class_name, count in _json_object(payload.get("object_counts_json")).items():
                label = normalize_text(class_name)
                if label:
                    try:
                        object_rows.append((doc_id, label, int(count)))
                    except (TypeError, ValueError):
                        continue

    edge_rows: list[tuple[Any, ...]] = []
    if temporal_edges is not None and not temporal_edges.empty:
        for row in temporal_edges.itertuples(index=False):
            edge_rows.append((
                str(getattr(row, "edge_id")),
                str(getattr(row, "video_id")),
                str(getattr(row, "source_shot_id")),
                str(getattr(row, "target_shot_id")),
                str(getattr(row, "relation")),
                float(getattr(row, "gap_sec", 0.0) or 0.0),
            ))

    connection: sqlite3.Connection | None = None
    fts5_enabled = False
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                video_id TEXT,
                shot_id TEXT,
                keyframe_id TEXT,
                event_id TEXT,
                frame_idx INTEGER,
                pts_time REAL,
                start_time REAL,
                end_time REAL,
                search_text TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE document_facets (
                doc_id TEXT NOT NULL,
                facet TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (doc_id, facet, value)
            ) WITHOUT ROWID;
            CREATE TABLE document_objects (
                doc_id TEXT NOT NULL,
                class_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (doc_id, class_name)
            ) WITHOUT ROWID;
            CREATE TABLE temporal_edges (
                edge_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                source_shot_id TEXT NOT NULL,
                target_shot_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                gap_sec REAL NOT NULL
            );
            CREATE TABLE snapshot_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE INDEX idx_documents_level_video_time ON documents(level, video_id, start_time, end_time);
            CREATE INDEX idx_documents_video_time ON documents(video_id, start_time, end_time);
            CREATE INDEX idx_documents_shot ON documents(shot_id);
            CREATE INDEX idx_documents_keyframe ON documents(keyframe_id);
            CREATE INDEX idx_documents_event ON documents(event_id);
            CREATE INDEX idx_facets_lookup ON document_facets(facet, value, doc_id);
            CREATE INDEX idx_objects_lookup ON document_objects(class_name, count, doc_id);
            CREATE INDEX idx_temporal_source ON temporal_edges(source_shot_id, relation, gap_sec);
            CREATE INDEX idx_temporal_target ON temporal_edges(target_shot_id, relation, gap_sec);
            """
        )
        connection.executemany(
            """INSERT INTO documents(
                doc_id,level,video_id,shot_id,keyframe_id,event_id,frame_idx,pts_time,
                start_time,end_time,search_text,payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            document_rows,
        )
        connection.executemany("INSERT OR IGNORE INTO document_facets(doc_id,facet,value) VALUES (?,?,?)", facet_rows)
        connection.executemany("INSERT OR REPLACE INTO document_objects(doc_id,class_name,count) VALUES (?,?,?)", object_rows)
        connection.executemany(
            "INSERT OR REPLACE INTO temporal_edges(edge_id,video_id,source_shot_id,target_shot_id,relation,gap_sec) VALUES (?,?,?,?,?,?)",
            edge_rows,
        )
        try:
            connection.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(doc_id UNINDEXED, body)")
            connection.executemany("INSERT INTO documents_fts(doc_id,body) VALUES (?,?)", fts_rows)
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
            fts5_enabled = True
        except sqlite3.OperationalError:
            fts5_enabled = False
        meta_rows = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "documents": len(document_rows),
            "levels": dict(sorted(level_counts.items())),
            "fts5": fts5_enabled,
            "temporal_edges": len(edge_rows),
        }
        connection.executemany(
            "INSERT INTO snapshot_meta(key,value) VALUES (?,?)",
            [(key, canonical_json(value)) for key, value in meta_rows.items()],
        )
        connection.execute("ANALYZE")
        connection.commit()
        connection.close()
        connection = None
        temporary.replace(path)
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise

    return {
        "available": True,
        "path": path.name,
        "fts5": fts5_enabled,
        "documents": len(document_rows),
        "levels": dict(sorted(level_counts.items())),
        "facets": len(facet_rows),
        "object_rows": len(object_rows),
        "temporal_edges": len(edge_rows),
        "tables": ["documents", "documents_fts", "document_facets", "document_objects", "temporal_edges", "snapshot_meta"]
        if fts5_enabled
        else ["documents", "document_facets", "document_objects", "temporal_edges", "snapshot_meta"],
        "query_helper": "v2_runtime.search_hybrid_snapshot",
    }


def build_fts5(path: Path, documents: pd.DataFrame, text_column: str = "search_text_vi") -> bool:
    """Compatibility wrapper for callers that only need one FTS5 document set."""
    level = "frame"
    if not documents.empty and "level" in documents.columns:
        values = documents["level"].dropna().astype(str).unique().tolist()
        if len(values) == 1:
            level = values[0]
    result = build_hybrid_snapshot(path, {level: documents}, text_column=text_column)
    return bool(result["fts5"])


def search_snapshot_fts(
    path: Path,
    query: str,
    *,
    limit: int = 100,
    levels: Sequence[str] | None = None,
    video_id: str | None = None,
    min_time: float | None = None,
    max_time: float | None = None,
    facets: Sequence[str] | None = None,
    object_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Search a hybrid snapshot with indexed metadata filters."""
    terms = [token for token in tokenize(query) if token]
    if not terms or limit < 1:
        return []
    expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
    where = ["documents_fts MATCH ?"]
    parameters: list[Any] = [expression]
    if levels:
        values = [str(value) for value in levels]
        where.append(f"d.level IN ({','.join('?' for _ in values)})")
        parameters.extend(values)
    if video_id is not None:
        where.append("d.video_id = ?")
        parameters.append(str(video_id))
    if min_time is not None:
        where.append("COALESCE(d.end_time, d.start_time, d.pts_time) >= ?")
        parameters.append(float(min_time))
    if max_time is not None:
        where.append("COALESCE(d.start_time, d.pts_time, d.end_time) <= ?")
        parameters.append(float(max_time))
    for value in facets or ():
        where.append("EXISTS (SELECT 1 FROM document_facets f WHERE f.doc_id=d.doc_id AND f.facet='scene' AND f.value=?)")
        parameters.append(str(value))
    for value in object_labels or ():
        where.append("EXISTS (SELECT 1 FROM document_objects o WHERE o.doc_id=d.doc_id AND o.class_name=? AND o.count>0)")
        parameters.append(str(value))
    parameters.append(int(limit))
    statement = f"""
        SELECT d.doc_id,d.level,d.video_id,d.shot_id,d.keyframe_id,d.event_id,
               d.frame_idx,d.pts_time,d.start_time,d.end_time,bm25(documents_fts) AS bm25_score
        FROM documents_fts
        JOIN documents d ON d.doc_id=documents_fts.doc_id
        WHERE {' AND '.join(where)}
        ORDER BY bm25_score,d.doc_id
        LIMIT ?
    """
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(statement, parameters).fetchall()]
    except sqlite3.OperationalError as exc:
        if "documents_fts" in str(exc):
            return []
        raise
    finally:
        connection.close()


def fetch_snapshot_documents(path: Path, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not doc_ids:
        return []
    unique_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))
    placeholders = ",".join("?" for _ in unique_ids)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"SELECT doc_id,level,video_id,shot_id,keyframe_id,event_id,frame_idx,pts_time,start_time,end_time,payload FROM documents WHERE doc_id IN ({placeholders})",
            unique_ids,
        ).fetchall()
    finally:
        connection.close()
    by_id = {str(row["doc_id"]): dict(row) for row in rows}
    return [by_id[doc_id] for doc_id in unique_ids if doc_id in by_id]


def search_hybrid_snapshot(
    path: Path,
    query: str,
    *,
    limit: int = 20,
    candidate_limit: int = 200,
    levels: Sequence[str] = ("event", "shot", "frame"),
    external_rankings: Mapping[str, Sequence[str]] | None = None,
    ranking_weights: Mapping[str, float] | None = None,
    video_id: str | None = None,
    min_time: float | None = None,
    max_time: float | None = None,
    facets: Sequence[str] | None = None,
    object_labels: Sequence[str] | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Run level-aware FTS and fuse it with optional ANN rankings using RRF."""
    rankings: dict[str, Sequence[str]] = dict(external_rankings or {})
    lexical_rows: dict[str, dict[str, Any]] = {}
    for level in levels:
        rows = search_snapshot_fts(
            path,
            query,
            limit=candidate_limit,
            levels=[level],
            video_id=video_id,
            min_time=min_time,
            max_time=max_time,
            facets=facets,
            object_labels=object_labels,
        )
        channel = f"fts:{level}"
        rankings[channel] = [str(row["doc_id"]) for row in rows]
        lexical_rows.update({str(row["doc_id"]): row for row in rows})
    rankings = {name: values for name, values in rankings.items() if values}
    fused = rrf_fuse(rankings, k=rrf_k, limit=limit, weights=ranking_weights)
    details = {str(row["doc_id"]): row for row in fetch_snapshot_documents(path, [doc_id for doc_id, _ in fused])}
    results: list[dict[str, Any]] = []
    for doc_id, score in fused:
        item = {**details.get(doc_id, lexical_rows.get(doc_id, {"doc_id": doc_id}))}
        item["rrf_score"] = float(score)
        item["channel_ranks"] = {
            name: list(values).index(doc_id) + 1
            for name, values in rankings.items()
            if doc_id in values
        }
        results.append(item)
    return results


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    k: int = 60,
    limit: int = 100,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for name, ranking in rankings.items():
        weight = float((weights or {}).get(name, 1.0))
        for rank, doc_id in enumerate(ranking, start=1):
            scores[str(doc_id)] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def validate_frame_contract(root: Path) -> dict[str, Any]:
    required = ["summary.json", "frame-mapping-report.json", "shots.parquet", "keyframes.parquet", "visual.npy", "embedding-records.parquet"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing V2 frame artifacts: {missing}")
    summary = read_json(root / "summary.json", {})
    shots = pd.read_parquet(root / "shots.parquet")
    keyframes = pd.read_parquet(root / "keyframes.parquet")
    records = pd.read_parquet(root / "embedding-records.parquet")
    vectors = np.load(root / "visual.npy", allow_pickle=False)
    if not summary.get("quality_report", {}).get("passed"):
        raise ValueError("quality_report.passed must be true")
    if len(keyframes) != len(records) or len(records) != len(vectors):
        raise ValueError("keyframes, embedding records and vectors are misaligned")
    if records["embedding_row"].tolist() != list(range(len(records))):
        raise ValueError("embedding_row must be contiguous")
    if keyframes["keyframe_id"].tolist() != records["keyframe_id"].tolist():
        raise ValueError("embedding record order differs from keyframes")
    if not keyframes["keyframe_id"].is_unique or not shots["shot_id"].is_unique:
        raise ValueError("duplicate keyframe or shot IDs")
    if not set(shots["shot_id"]).issubset(set(keyframes["shot_id"])):
        raise ValueError("shot without a keyframe")
    norms = np.linalg.norm(vectors.astype("float32"), axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("visual vectors must be finite and L2 normalized")
    return {"video_id": summary.get("video_id"), "shots": len(shots), "keyframes": len(keyframes), "dimension": int(vectors.shape[1])}


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def timed_call(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started
