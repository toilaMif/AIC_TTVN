from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psycopg
from retrieval.config import settings

from retrieval.search.asr import search_asr
from retrieval.search.global_visual import (
    LocalizedClause,
    localized_search,
    search_text_diversified,
)

app = FastAPI(title="AIC-TTVN API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/assets/LogoAIC_TTVN_1.png", include_in_schema=False)
def logo() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parents[2] / "LogoAIC_TTVN_1.png")


class VisualSearchItem(BaseModel):
    keyframe_id: str
    video_id: str
    frame_idx: int
    pts_time: float | None
    score: float
    frame_url: str | None
    shot_id: str | None


class LocalizedClauseRequest(BaseModel):
    text: str = Field(min_length=1)
    box: tuple[float, float, float, float]
    required: bool = True
    weight: float = 1.0


class LocalizedSearchRequest(BaseModel):
    global_query: str = Field(min_length=1)
    clauses: list[LocalizedClauseRequest] = Field(min_length=1)
    candidate_k: int = Field(default=200, ge=1, le=500)
    rerank_k: int = Field(default=50, ge=1, le=200)
    top_k: int = Field(default=20, ge=1, le=100)


class AsrSearchItem(BaseModel):
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    score: float
    keyframe_id: str | None
    frame_idx: int | None
    pts_time: float | None
    frame_url: str | None

class VideoPlayback(BaseModel):
    video_id: str
    watch_url: str

@app.get('/videos/{video_id}', response_model=VideoPlayback)
def video_playback(video_id: str) -> VideoPlayback:
    with psycopg.connect(settings.database_url.replace('+psycopg', '')) as conn, conn.cursor() as cur:
        cur.execute('SELECT watch_url FROM videos WHERE video_id=%s', (video_id,))
        row = cur.fetchone()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail='Video not found')
    return VideoPlayback(video_id=video_id, watch_url=str(row[0]))


@app.post("/search/localized", response_model=list[VisualSearchItem])
def localized_visual_search(request: LocalizedSearchRequest) -> list[VisualSearchItem]:
    clauses = [
        LocalizedClause(item.text, item.box, item.required, item.weight) for item in request.clauses
    ]
    results = localized_search(
        request.global_query, clauses, request.candidate_k, request.rerank_k, request.top_k
    )
    return [VisualSearchItem(**result.__dict__) for result in results]


@app.get("/search/visual", response_model=list[VisualSearchItem])
def visual_search(
    query: str = Query(min_length=1),
    top_k: int = Query(default=20, ge=1, le=100),
) -> list[VisualSearchItem]:
    return [
        VisualSearchItem(**result.__dict__)
        for result in search_text_diversified(query, top_k=top_k, candidate_k=200)
    ]


@app.get("/search/asr", response_model=list[AsrSearchItem])
def asr_search(
    query: str = Query(min_length=1), top_k: int = Query(default=20, ge=1, le=100)
) -> list[AsrSearchItem]:
    return [AsrSearchItem(**result.__dict__) for result in search_asr(query, top_k)]


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REACT_ROOT = _PROJECT_ROOT / "apps" / "frontend" / "dist"
_LEGACY_WEB_ROOT = _PROJECT_ROOT / "apps" / "web"
_WEB_ROOT = _REACT_ROOT if _REACT_ROOT.is_dir() else _LEGACY_WEB_ROOT
if _WEB_ROOT.is_dir():
    app.mount("/ui", StaticFiles(directory=_WEB_ROOT, html=True), name="ui")
