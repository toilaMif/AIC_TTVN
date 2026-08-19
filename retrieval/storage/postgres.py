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
