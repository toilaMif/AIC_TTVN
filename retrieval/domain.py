"""Stable domain identifiers shared by ingest, search and interaction layers."""

from dataclasses import dataclass


def keyframe_id(video_id: str, ordinal: int) -> str:
    """Return the legacy BTC keyframe ID used by audited mapping imports."""
    return f"{video_id}:{ordinal:06d}"


def self_keyframe_id(video_id: str, ordinal: int, version: str = "aicv2") -> str:
    """Return a stable ID for a self-extracted keyframe version."""
    return f"{video_id}:self:{version}:kf:{ordinal:06d}"


def self_shot_id(video_id: str, shot_index: int, version: str = "aicv2") -> str:
    """Return a stable ID for a self-detected shot version."""
    return f"{video_id}:self:{version}:shot:{shot_index:06d}"


@dataclass(frozen=True)
class TemporalLocation:
    video_id: str
    keyframe_id: str
    frame_idx: int
    pts_time: float
    shot_id: str | None = None
    window_id: str | None = None


@dataclass(frozen=True)
class Candidate:
    location: TemporalLocation
    score: float
    source: str
