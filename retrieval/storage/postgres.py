"""SQLAlchemy models for source metadata and feature lifecycle."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    youtube_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    watch_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    duration_expected_sec: Mapped[float | None] = mapped_column(Float)
    availability_status: Mapped[str] = mapped_column(String(32), default="unchecked")
    source_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    dataset_group: Mapped[str | None] = mapped_column(String(32), index=True)
    artifact_batch: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)


class Keyframe(Base):
    __tablename__ = "keyframes"
    __table_args__ = (UniqueConstraint("video_id", "source", "pipeline_version", "n"),)

    keyframe_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    n: Mapped[int] = mapped_column(Integer)
    frame_idx: Mapped[int] = mapped_column(BigInteger)
    pts_time: Mapped[float] = mapped_column(Float)
    fps: Mapped[float] = mapped_column(Float)
    shot_id: Mapped[str | None] = mapped_column(String(64))
    window_id: Mapped[str | None] = mapped_column(String(64))
    frame_object_key: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="btc")
    pipeline_version: Mapped[str] = mapped_column(String(32), default="btc")
    run_id: Mapped[str | None] = mapped_column(ForeignKey("ingest_runs.run_id"))
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_fallback: Mapped[bool | None] = mapped_column(Boolean)
    dataset_group: Mapped[str | None] = mapped_column(String(32), index=True)
    artifact_batch: Mapped[str | None] = mapped_column(String(64), index=True)


class Shot(Base):
    __tablename__ = "shots"

    shot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    shot_index: Mapped[int] = mapped_column(Integer)
    start_frame: Mapped[int] = mapped_column(BigInteger)
    end_frame: Mapped[int] = mapped_column(BigInteger)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    duration_sec: Mapped[float] = mapped_column(Float)
    detector: Mapped[str] = mapped_column(String(64))
    pipeline_version: Mapped[str] = mapped_column(String(32))
    run_id: Mapped[str] = mapped_column(ForeignKey("ingest_runs.run_id"))


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    video_count: Mapped[int] = mapped_column(Integer, default=0)
    keyframe_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dataset_group: Mapped[str | None] = mapped_column(String(32))
    artifact_batch: Mapped[str | None] = mapped_column(String(64))


class FeatureJob(Base):
    __tablename__ = "feature_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    embedding_version: Mapped[str] = mapped_column(String(128), unique=True)
    dimension: Mapped[int] = mapped_column(Integer)
    dtype: Mapped[str] = mapped_column(String(16))
    expected_count: Mapped[int] = mapped_column(Integer)
    completed_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    dataset_group: Mapped[str | None] = mapped_column(String(32))
    artifact_batch: Mapped[str | None] = mapped_column(String(64))


class AsrSegment(Base):
    __tablename__ = "asr_segments"

    segment_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    segment_index: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(16), default="vi")
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    words_json: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dataset_group: Mapped[str | None] = mapped_column(String(32), index=True)
    artifact_batch: Mapped[str | None] = mapped_column(String(64))


class OcrRecord(Base):
    __tablename__ = "ocr_records"

    keyframe_id: Mapped[str] = mapped_column(ForeignKey("keyframes.keyframe_id"), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    frame_idx: Mapped[int] = mapped_column(BigInteger)
    pts_time: Mapped[float] = mapped_column(Float)
    shot_id: Mapped[str | None] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    normalized_text: Mapped[str | None] = mapped_column(Text)
    search_text: Mapped[str | None] = mapped_column(Text)
    shot_text: Mapped[str | None] = mapped_column(Text)
    shot_search_text: Mapped[str | None] = mapped_column(Text)
    ocr_error: Mapped[str | None] = mapped_column(Text)
    pipeline_version: Mapped[str] = mapped_column(String(64))
    detections_json: Mapped[str | None] = mapped_column(Text)
    merged_detections_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dataset_group: Mapped[str | None] = mapped_column(String(32), index=True)
    artifact_batch: Mapped[str | None] = mapped_column(String(64))


class ObjectDetection(Base):
    __tablename__ = "object_detections"

    detection_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    keyframe_id: Mapped[str] = mapped_column(ForeignKey("keyframes.keyframe_id"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    frame_idx: Mapped[int] = mapped_column(BigInteger)
    pts_time: Mapped[float] = mapped_column(Float)
    model: Mapped[str] = mapped_column(String(128))
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(128), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1_norm: Mapped[float] = mapped_column(Float)
    y1_norm: Mapped[float] = mapped_column(Float)
    x2_norm: Mapped[float] = mapped_column(Float)
    y2_norm: Mapped[float] = mapped_column(Float)
    bbox_area_ratio: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dataset_group: Mapped[str | None] = mapped_column(String(32), index=True)
    artifact_batch: Mapped[str | None] = mapped_column(String(64))


class FeatureRecord(Base):
    __tablename__ = "feature_records"
    __table_args__ = (
        UniqueConstraint("keyframe_id", "embedding_version"),
        UniqueConstraint("milvus_collection", "milvus_pk"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("feature_jobs.job_id"))
    keyframe_id: Mapped[str] = mapped_column(ForeignKey("keyframes.keyframe_id"))
    embedding_version: Mapped[str] = mapped_column(String(128))
    milvus_collection: Mapped[str] = mapped_column(String(128))
    milvus_pk: Mapped[int] = mapped_column(BigInteger)
    import_status: Mapped[str] = mapped_column(String(32), default="pending")
