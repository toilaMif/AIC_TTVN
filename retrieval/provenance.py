"""Helpers for stable dataset and artifact-batch provenance."""

from __future__ import annotations

import re
from pathlib import Path

VIDEO_GROUP_RE = re.compile(r"^(L\d+)_V\d+$", re.IGNORECASE)
VIDEO_IN_ARTIFACT_RE = re.compile(r"(L\d+_V\d+)$", re.IGNORECASE)


def dataset_group(video_id: str) -> str:
    """Return the logical source group (for example ``L26``)."""
    match = VIDEO_GROUP_RE.fullmatch(video_id.strip())
    return match.group(1).upper() if match else video_id.strip().split("_", 1)[0].upper()


def artifact_batch_for_video(video_id: str, artifact_root: Path) -> str | None:
    """Find the artifact batch containing a video by inspecting stage-01 names."""
    wanted = video_id.upper()
    if not artifact_root.is_dir():
        return None
    for batch_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        shot_dir = batch_dir / "01-shot-keyframes"
        if not shot_dir.is_dir():
            continue
        for archive in shot_dir.glob("*.zip"):
            match = VIDEO_IN_ARTIFACT_RE.search(archive.stem)
            if match and match.group(1).upper() == wanted:
                return batch_dir.name
    return None


def artifact_batch_from_root(root: Path) -> str | None:
    """Return the batch name for a ``<artifact-root>/<batch>/<stage>`` path."""
    parts = list(root.resolve().parts)
    for marker in ("01-shot-keyframes", "02-visual-embeddings", "03-ocr", "04-asr", "05-object-detection", "06-frame-understanding"):
        if marker in parts:
            index = parts.index(marker)
            return parts[index - 1] if index else None
    return None
