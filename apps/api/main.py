import csv
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psycopg
from retrieval.config import settings

from apps.api.auth import create_token, get_username
from apps.api.questions import (
    delete_question_set,
    import_question_set,
    list_questions,
    parse_question_zip,
    select_question,
    set_answer_text,
    set_question_done,
    toggle_answer_frame,
)

from retrieval.search.asr import search_asr
from retrieval.search.global_visual import (
    LocalizedClause,
    localized_search,
    search_text_diversified,
)
from retrieval.search.textual import search_objects, search_ocr
from retrieval.search.video_topic import search_topic

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAP_KEYFRAMES_ROOT = _PROJECT_ROOT / "data" / "source" / "aic2026" / "map-keyframes"

app = FastAPI(title="AIC-TTVN API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    username = get_username(credentials.credentials) if credentials and credentials.scheme.lower() == "bearer" else None
    if username is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return username


def require_admin(username: str = Depends(require_auth)) -> str:
    if username not in settings.admin_accounts:
        raise HTTPException(status_code=403, detail="Chỉ admin mới được thực hiện thao tác này")
    return username


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


@app.post("/auth/login")
def login(request: LoginRequest) -> dict[str, str]:
    expected_password = settings.accounts.get(request.username)
    if expected_password is None or not secrets.compare_digest(request.password, expected_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": create_token(request.username)}


@app.get("/auth/me")
def auth_me(username: str = Depends(require_auth)) -> dict:
    return {"username": username, "is_admin": username in settings.admin_accounts}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search-options")
def search_options(_: None = Depends(require_auth)) -> dict[str, list[str]]:
    """Return selectable artifact batches/directories for the UI."""
    root = Path(str(settings.aic_kaggle_artifact_root))
    batches = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    directories = ["all"] + batches
    return {"batches": ["all"] + batches, "directories": directories}


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


class TopicSearchItem(BaseModel):
    video_id: str
    title: str
    topic: str | None
    score: float
    keyframe_id: str | None
    frame_idx: int | None
    pts_time: float | None
    frame_url: str | None
    text: str


class TextualSearchItem(BaseModel):
    keyframe_id: str
    video_id: str
    frame_idx: int
    pts_time: float
    score: float
    frame_url: str | None
    shot_id: str | None
    text: str


class VideoPlayback(BaseModel):
    video_id: str
    watch_url: str


class VideoSearchItem(BaseModel):
    video_id: str
    title: str | None
    watch_url: str
    thumbnail_url: str | None


@app.get("/videos", response_model=list[VideoSearchItem])
def video_search(
    query: str = Query(min_length=1), limit: int = Query(default=8, ge=1, le=20),
    _: None = Depends(require_auth),
) -> list[VideoSearchItem]:
    term = query.strip()
    if not term:
        return []
    contains = f"%{term}%"
    prefix = f"{term}%"
    with (
        psycopg.connect(settings.database_url.replace("+psycopg", "")) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            """
            SELECT video_id, title, watch_url, source_metadata->>'thumbnail_url'
            FROM videos
            WHERE video_id ILIKE %s OR COALESCE(title, '') ILIKE %s
            ORDER BY
                CASE
                    WHEN LOWER(video_id) = LOWER(%s) THEN 0
                    WHEN video_id ILIKE %s THEN 1
                    WHEN COALESCE(title, '') ILIKE %s THEN 2
                    ELSE 3
                END,
                video_id
            LIMIT %s
            """,
            (contains, contains, term, prefix, prefix, limit),
        )
        rows = cur.fetchall()
    return [
        VideoSearchItem(
            video_id=str(video_id),
            title=str(title) if title else None,
            watch_url=str(watch_url),
            thumbnail_url=str(thumbnail_url) if thumbnail_url else None,
        )
        for video_id, title, watch_url, thumbnail_url in rows
    ]


def _matches_scope(video_id: str, batch: str, directory: str) -> bool:
    scopes = [value.strip().lower() for value in (batch, directory) if value and value.lower() != "all"]
    return not scopes or any(video_id.lower().startswith(scope.replace("_", "")) or video_id.lower().startswith(scope) for scope in scopes)


def _scope(results, batch: str, directory: str):
    return [item for item in results if _matches_scope(item.video_id, batch, directory)]


@app.get("/videos/{video_id}", response_model=VideoPlayback)
def video_playback(video_id: str, _: None = Depends(require_auth)) -> VideoPlayback:
    with (
        psycopg.connect(settings.database_url.replace("+psycopg", "")) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT watch_url FROM videos WHERE video_id=%s", (video_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoPlayback(video_id=video_id, watch_url=str(row[0]))


class KeyframeMapRow(BaseModel):
    n: int
    pts_time: float
    fps: float
    frame_idx: int


@app.get('/keyframes/{video_id}', response_model=list[KeyframeMapRow])
def keyframe_map(video_id: str, _: None = Depends(require_auth)) -> list[KeyframeMapRow]:
    path = _MAP_KEYFRAMES_ROOT / f"{video_id}.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail='Keyframe map not found')
    with path.open(newline='', encoding='utf-8') as handle:
        return [
            KeyframeMapRow(
                n=int(row['n']),
                pts_time=float(row['pts_time']),
                fps=float(row['fps']),
                frame_idx=int(row['frame_idx']),
            )
            for row in csv.DictReader(handle)
        ]


@app.post("/search/localized", response_model=list[VisualSearchItem])
def localized_visual_search(request: LocalizedSearchRequest, _: None = Depends(require_auth)) -> list[VisualSearchItem]:
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
    top_k: int = Query(default=20, ge=1, le=200),
    batch: str = Query(default="all"), directory: str = Query(default="all"),
    video_id: str | None = Query(default=None),
    _: None = Depends(require_auth),
) -> list[VisualSearchItem]:
    return [
        VisualSearchItem(**result.__dict__)
        for result in _scope(search_text_diversified(query, top_k=min(200, top_k * 5), candidate_k=200, video_id=video_id), batch, directory)[:top_k]
    ]


@app.get("/search/asr", response_model=list[AsrSearchItem])
def asr_search(
    query: str = Query(min_length=1), top_k: int = Query(default=20, ge=1, le=200), batch: str = Query(default="all"), directory: str = Query(default="all"),
    video_id: str | None = Query(default=None),
    _: None = Depends(require_auth),
) -> list[AsrSearchItem]:
    return [AsrSearchItem(**result.__dict__) for result in _scope(search_asr(query, min(200, top_k * 5), video_id=video_id), batch, directory)[:top_k]]


@app.get("/search/ocr", response_model=list[TextualSearchItem])
def ocr_search(
    query: str = Query(min_length=1), top_k: int = Query(default=20, ge=1, le=200), batch: str = Query(default="all"), directory: str = Query(default="all"),
    video_id: str | None = Query(default=None),
    _: None = Depends(require_auth),
) -> list[TextualSearchItem]:
    return [TextualSearchItem(**result.__dict__) for result in _scope(search_ocr(query, min(200, top_k * 5), video_id=video_id), batch, directory)[:top_k]]


@app.get("/search/object", response_model=list[TextualSearchItem])
def object_search(
    query: str = Query(min_length=1), top_k: int = Query(default=20, ge=1, le=200), batch: str = Query(default="all"), directory: str = Query(default="all"),
    video_id: str | None = Query(default=None),
    _: None = Depends(require_auth),
) -> list[TextualSearchItem]:
    return [TextualSearchItem(**result.__dict__) for result in _scope(search_objects(query, min(200, top_k * 5), video_id=video_id), batch, directory)[:top_k]]


@app.get("/search/topic", response_model=list[TopicSearchItem])
def topic_search(
    query: str = Query(min_length=1), top_k: int = Query(default=20, ge=1, le=200), batch: str = Query(default="all"), directory: str = Query(default="all"),
    _: None = Depends(require_auth),
) -> list[TopicSearchItem]:
    return [TopicSearchItem(**result.__dict__) for result in _scope(search_topic(query, min(200, top_k * 5)), batch, directory)[:top_k]]


class QuestionDoneRequest(BaseModel):
    done: bool


class AnswerFrameRequest(BaseModel):
    video_id: str = Field(min_length=1)
    frame_idx: int
    pts_time: float


class AnswerTextRequest(BaseModel):
    text: str = Field(default="", max_length=100)


@app.post("/questions/import")
async def questions_import(
    file: UploadFile = File(...),
    _: str = Depends(require_admin),
) -> dict:
    data = await file.read()
    try:
        parsed = parse_question_zip(data, file.filename or "bo-de.zip")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được file zip: {exc}") from exc
    if not parsed.questions:
        raise HTTPException(status_code=400, detail="Không tìm thấy câu hỏi hợp lệ trong file zip.")
    return import_question_set(parsed, file.filename or "bo-de.zip")


@app.get("/questions")
def questions_list(
    set_id: str | None = Query(default=None),
    _: None = Depends(require_auth),
) -> list[dict]:
    return list_questions(set_id)


@app.delete("/questions/{set_id}")
def questions_delete_set(
    set_id: str,
    _: str = Depends(require_admin),
) -> dict:
    if not delete_question_set(set_id):
        raise HTTPException(status_code=404, detail="Question set not found")
    return {"deleted": set_id}


@app.post("/questions/{question_id}/done")
def questions_set_done(
    question_id: str,
    request: QuestionDoneRequest,
    username: str = Depends(require_auth),
) -> dict:
    updated = set_question_done(question_id, request.done, username)
    if updated is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated


@app.post("/questions/{question_id}/select")
def questions_select(
    question_id: str,
    username: str = Depends(require_auth),
) -> dict:
    updated = select_question(question_id, username)
    if updated is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated


@app.post("/questions/{question_id}/answer/frame")
def questions_answer_frame(
    question_id: str,
    request: AnswerFrameRequest,
    _: str = Depends(require_auth),
) -> dict:
    try:
        updated = toggle_answer_frame(question_id, request.video_id, request.frame_idx, request.pts_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated


@app.post("/questions/{question_id}/answer/text")
def questions_answer_text(
    question_id: str,
    request: AnswerTextRequest,
    _: str = Depends(require_auth),
) -> dict:
    updated = set_answer_text(question_id, request.text)
    if updated is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated


_REACT_ROOT = _PROJECT_ROOT / "apps" / "frontend" / "dist"
_LEGACY_WEB_ROOT = _PROJECT_ROOT / "apps" / "web"
_WEB_ROOT = _REACT_ROOT if _REACT_ROOT.is_dir() else _LEGACY_WEB_ROOT
if _WEB_ROOT.is_dir():
    app.mount("/ui", StaticFiles(directory=_WEB_ROOT, html=True), name="ui")
