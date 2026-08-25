from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    # Clear stale Kaggle execution state so the next run starts cleanly.
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def cell_source(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index].get("source", []))


def set_cell(notebook: dict, index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source


def replace_once(notebook: dict, index: int, old: str, new: str) -> None:
    source = cell_source(notebook, index)
    if old not in source:
        raise RuntimeError(f"Could not find replacement text in cell {index}: {old!r}")
    set_cell(notebook, index, source.replace(old, new, 1))


def update_ocr() -> None:
    path = ROOT / "pipelines/kaggle/notebooks/03-ocr/03-ocr-single-video-optimized.ipynb"
    notebook = load_notebook(path)
    set_cell(
        notebook,
        0,
        """# 03 - OCR single video (optimized PP-OCRv5, full-video default)

This stage reads keyframes from notebook 01 and writes one OCR record per
`keyframe_id`. The production default processes every keyframe in the selected
video. Set `OCR_DEBUG_MAX_FRAMES` to a positive number only for a smoke test.

OCR keeps scene text and ticker text typed separately. Stage 06 uses scene text
normally and keeps ticker text as lower-weight evidence.

If an earlier run raised `libtorch_cuda.so: undefined symbol: ncclCommShrink`,
stop the Kaggle session and start a fresh GPU session before running this
notebook from the first cell.
""",
    )
    source = cell_source(notebook, 2)
    source = source.replace(
        'PIPELINE_VERSION = "aicv3-ocr-optimized-v5-nccl-safe"',
        'PIPELINE_VERSION = "aicv3-ocr-optimized-v6-full-video"',
    )
    source = source.replace(
        "OCR_DEBUG_MAX_FRAMES = 64  # Set to None for the complete video.",
        "OCR_DEBUG_MAX_FRAMES = None  # Set a positive number for a smoke test.",
    )
    set_cell(notebook, 2, source)
    save_notebook(path, notebook)


def update_objects() -> None:
    path = ROOT / "pipelines/kaggle/notebooks/05-object-detection/05-object-detection-single-video-optimized.ipynb"
    notebook = load_notebook(path)
    set_cell(
        notebook,
        0,
        """# 05 - Object detection (single video, fixed COCO vocabulary)

This stage uses the fixed-vocabulary `yolo26m.pt` model. It supplies stable
basic objects such as people, vehicles, dogs, horses, boats and common food.
Long-tail species and produce are handled by the VLM in stage 06, rather than
by prompt-free open-vocabulary detections.

The old `yoloe-26m-seg-pf.pt` prompt-free output is intentionally not accepted
as stage 06 input because it produced many unrelated labels. A future YOLOE
pass must use an explicit, audited vocabulary and remain a separate artifact.

The production default processes every keyframe in the selected video. Set
`MAX_FRAMES` to a positive number only for a smoke test. Preview, annotated
exports and archives are disabled by default to reduce runtime and I/O.
""",
    )
    source = cell_source(notebook, 2)
    replacements = {
        'MODEL_NAME = "yoloe-26m-seg-pf.pt"': 'MODEL_NAME = "yolo26m.pt"',
        '# Faster open-vocabulary options (lower recall):\n# MODEL_NAME = "yoloe-26s-seg-pf.pt"\n# MODEL_NAME = "yoloe-26n-seg-pf.pt"\n# Fixed-vocabulary alternatives when the 80 COCO classes are sufficient:\n# MODEL_NAME = "yolo26s.pt"  # fastest\n# MODEL_NAME = "yolo26m.pt"': '# Faster fixed-vocabulary alternative (lower recall):\n# MODEL_NAME = "yolo26s.pt"',
        'CONFIDENCE_THRESHOLD = 0.20': 'CONFIDENCE_THRESHOLD = 0.25',
        'MAX_FRAMES = 64  # Smoke test default; set to None after visual verification.': 'MAX_FRAMES = None  # Full selected video; set a positive number for smoke test.',
        'PREVIEW_COUNT = 8': 'PREVIEW_COUNT = 0',
    }
    for old, new in replacements.items():
        # Keep the updater idempotent: a second invocation should leave the
        # already-updated notebook unchanged.
        if old in source:
            source = source.replace(old, new, 1)
    source = source.replace(
        'PIPELINE_VERSION = f"aicv3-object-{Path(MODEL_NAME).stem}-fast"',
        'PIPELINE_VERSION = f"aicv3-object-{Path(MODEL_NAME).stem}-coco-full-video"',
    )
    set_cell(notebook, 2, source)
    save_notebook(path, notebook)


FRAME_INPUT_CELL = r'''def choose_video_parquet(
    filename: str,
    required_columns: set[str],
    preferred_terms: tuple[str, ...] = (),
    require_trusted_coco: bool = False,
) -> tuple[Path, pd.DataFrame] | tuple[None, pd.DataFrame]:
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    valid = []
    for path in candidates:
        try:
            dataframe = pd.read_parquet(path)
        except Exception:
            continue
        if not required_columns.issubset(dataframe.columns):
            continue
        if "video_id" in dataframe.columns:
            video_rows = dataframe[
                dataframe["video_id"].astype(str) == VIDEO_ID
            ].copy()
            if video_rows.empty:
                continue
        else:
            video_rows = dataframe.copy()
        if require_trusted_coco:
            model_values = {
                Path(str(value)).stem.casefold()
                for value in video_rows.get("model", pd.Series(dtype=str)).dropna()
            }
            path_text = path.as_posix().casefold()
            trusted = bool(model_values & TRUSTED_COCO_MODEL_STEMS)
            trusted = trusted or any(
                f"{stem}.pt" in path_text for stem in TRUSTED_COCO_MODEL_STEMS
            )
            if not trusted:
                continue
        path_text = path.as_posix().casefold()
        score = 5 * int(VIDEO_ID.casefold() in path_text)
        score += sum(int(term.casefold() in path_text) for term in preferred_terms)
        valid.append((score, path, video_rows))
    if not valid:
        if require_trusted_coco:
            return None, pd.DataFrame()
        raise FileNotFoundError(
            f"No usable {filename} for {VIDEO_ID} under /kaggle/input"
        )
    valid.sort(key=lambda item: (-item[0], item[1].as_posix()))
    return valid[0][1], valid[0][2]


KEYFRAMES_PATH, keyframes_df = choose_video_parquet(
    "keyframes.parquet",
    {"keyframe_id", "video_id", "frame_idx", "pts_time", "frame_path"},
    ("shot", "keyframe"),
)
OCR_PATH, ocr_df = choose_video_parquet(
    "ocr.parquet",
    {"keyframe_id", "video_id"},
    ("ocr", "optimized"),
)
OBJECTS_PATH, objects_df = choose_video_parquet(
    "objects.parquet",
    {"keyframe_id", "video_id", "class_name", "confidence"},
    ("object", "coco", "yolo26", "optimized"),
    require_trusted_coco=True,
)

if OBJECTS_PATH is None:
    print(
        "No trusted YOLO26 COCO objects.parquet found. "
        "Continuing with VLM + OCR only."
    )
    objects_df = pd.DataFrame(
        columns=["keyframe_id", "video_id", "class_name", "confidence"]
    )

INPUT_ROOT = KEYFRAMES_PATH.parent
keyframes_df = keyframes_df.sort_values("frame_idx").reset_index(drop=True)
objects_df = objects_df.reset_index(drop=True)
ocr_df = ocr_df.reset_index(drop=True)

if "shot_id" not in keyframes_df.columns:
    keyframes_df["shot_id"] = (
        keyframes_df["video_id"].astype(str)
        + ":synthetic-shot:"
        + keyframes_df.index.astype(str)
    )
if "quality_score" not in keyframes_df.columns:
    keyframes_df["quality_score"] = 0.0
if "text" not in ocr_df.columns:
    ocr_df["text"] = ""
if "shot_text" not in ocr_df.columns:
    ocr_df["shot_text"] = ocr_df["text"]

all_shot_ids = keyframes_df["shot_id"].astype(str).drop_duplicates().tolist()
if MAX_SHOTS is not None and MAX_SHOTS < len(all_shot_ids):
    sampled_indices = np.unique(
        np.linspace(0, len(all_shot_ids) - 1, int(MAX_SHOTS), dtype=int)
    )
    selected_shots = {all_shot_ids[index] for index in sampled_indices}
    keyframes_df = keyframes_df[
        keyframes_df["shot_id"].astype(str).isin(selected_shots)
    ].copy()
    selected_keyframes = set(keyframes_df["keyframe_id"].astype(str))
    objects_df = objects_df[
        objects_df["keyframe_id"].astype(str).isin(selected_keyframes)
    ].copy()
    ocr_df = ocr_df[
        ocr_df["keyframe_id"].astype(str).isin(selected_keyframes)
    ].copy()

keyframes_df = keyframes_df.sort_values("frame_idx").reset_index(drop=True)
print("Keyframes:", KEYFRAMES_PATH, len(keyframes_df))
print("OCR:", OCR_PATH, len(ocr_df))
print("Objects:", OBJECTS_PATH or "none", len(objects_df))
print("Shots:", keyframes_df["shot_id"].nunique())
'''


FRAME_EVIDENCE_CELL = r'''def normalize_text(value) -> str:
    value = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def normalized_tokens(value) -> set[str]:
    text = unicodedata.normalize("NFD", normalize_text(value).casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return set(re.findall(r"[a-z0-9]+", text))


def jaccard_distance(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / max(len(left | right), 1)


def safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def parse_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def object_prominence(row) -> float:
    confidence = safe_float(getattr(row, "confidence", 0.0))
    area = max(0.0, safe_float(getattr(row, "bbox_area_ratio", 0.0)))
    if all(hasattr(row, name) for name in ("x1_norm", "y1_norm", "x2_norm", "y2_norm")):
        center_x = (safe_float(row.x1_norm) + safe_float(row.x2_norm)) / 2
        center_y = (safe_float(row.y1_norm) + safe_float(row.y2_norm)) / 2
        distance = min(1.0, math.hypot(center_x - 0.5, center_y - 0.5) / math.hypot(0.5, 0.5))
    else:
        distance = 0.5
    return confidence * (0.45 + 0.55 * math.sqrt(min(area, 1.0))) * (1.0 - 0.35 * distance)


# Only fixed COCO names from YOLO26 are allowed into the VLM hints and search index.
objects_df["class_name"] = objects_df.get("class_name", "").map(normalize_text)
objects_df["confidence"] = pd.to_numeric(objects_df.get("confidence", 0.0), errors="coerce").fillna(0.0)
objects_df = objects_df[
    (objects_df["confidence"] >= OBJECT_MIN_CONFIDENCE)
    & objects_df["class_name"].str.casefold().isin({item.casefold() for item in COCO_OBJECT_WHITELIST})
].copy()

object_hints_by_keyframe = {}
object_labels_by_keyframe = {}
object_signature_by_keyframe = {}
for keyframe_id, group in objects_df.groupby(objects_df["keyframe_id"].astype(str), sort=False):
    ranked = []
    for row in group.itertuples(index=False):
        ranked.append(
            {
                "class_name": normalize_text(row.class_name),
                "confidence": round(safe_float(row.confidence), 4),
                "prominence": round(object_prominence(row), 4),
                "bbox_norm": [
                    round(safe_float(getattr(row, name, 0.0)), 4)
                    for name in ("x1_norm", "y1_norm", "x2_norm", "y2_norm")
                ],
            }
        )
    ranked.sort(key=lambda item: (item["prominence"], item["confidence"]), reverse=True)
    ranked = ranked[:OBJECT_HINT_LIMIT]
    labels = []
    for item in ranked:
        if item["class_name"] and item["class_name"] not in labels:
            labels.append(item["class_name"])
    object_hints_by_keyframe[keyframe_id] = ranked
    object_labels_by_keyframe[keyframe_id] = labels
    object_signature_by_keyframe[keyframe_id] = {label.casefold() for label in labels}


ocr_by_keyframe = {}
for row in ocr_df.itertuples(index=False):
    detections = parse_json_list(getattr(row, "detections_json", None))
    if not detections and hasattr(row, "detections"):
        detections = parse_json_list(row.detections)
    scene_parts = []
    ticker_parts = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        text = normalize_text(item.get("text", ""))
        if not text:
            continue
        text_type = normalize_text(item.get("text_type", "scene")).casefold()
        if text_type == "ticker":
            ticker_parts.append(text)
        elif text_type != "logo":
            scene_parts.append(text)
    fallback_text = normalize_text(getattr(row, "text", ""))
    if not scene_parts and fallback_text:
        scene_parts = [fallback_text]
    ocr_by_keyframe[str(row.keyframe_id)] = {
        "text": normalize_text(" ".join(scene_parts + ticker_parts)),
        "scene_text": normalize_text(" ".join(dict.fromkeys(scene_parts))),
        "ticker_text": normalize_text(" ".join(dict.fromkeys(ticker_parts))),
        "shot_text": normalize_text(getattr(row, "shot_text", "")),
    }


def evidence_for(keyframe_id: str) -> dict:
    ocr = ocr_by_keyframe.get(keyframe_id, {})
    return {
        "objects": object_hints_by_keyframe.get(keyframe_id, []),
        "object_labels": object_labels_by_keyframe.get(keyframe_id, []),
        "ocr_text": ocr.get("text", ""),
        "ocr_scene_text": ocr.get("scene_text", ""),
        "ocr_ticker_text": ocr.get("ticker_text", ""),
        "ocr_shot_text": ocr.get("shot_text", ""),
    }


print(
    "Trusted COCO object detections:",
    len(objects_df),
    "across",
    len(object_labels_by_keyframe),
    "keyframes; OCR records:",
    len(ocr_by_keyframe),
)
'''


FRAME_SELECTION_CELL = r'''def frame_selection_score(row) -> float:
    keyframe_id = str(row.keyframe_id)
    object_score = max(
        (item["prominence"] for item in object_hints_by_keyframe.get(keyframe_id, [])),
        default=0.0,
    )
    ocr_scene_bonus = min(len(evidence_for(keyframe_id)["ocr_scene_text"]) / 200.0, 1.0)
    quality = safe_float(getattr(row, "quality_score", 0.0))
    return 0.55 * quality + 0.30 * object_score + 0.15 * ocr_scene_bonus


selection_rows = []
for shot_id, shot_df in keyframes_df.groupby("shot_id", sort=False):
    shot_df = shot_df.sort_values("frame_idx")
    rows = list(shot_df.itertuples(index=False))
    representative = max(rows, key=frame_selection_score)
    selected = [representative]
    selection_rows.append(
        {
            "keyframe_id": str(representative.keyframe_id),
            "shot_id": str(shot_id),
            "frame_idx": int(representative.frame_idx),
            "reason": "shot_representative",
            "selection_score": frame_selection_score(representative),
            "object_distance": 0.0,
            "ocr_distance": 0.0,
        }
    )
    if MAX_VLM_FRAMES_PER_SHOT <= 1:
        continue

    representative_id = str(representative.keyframe_id)
    representative_objects = object_signature_by_keyframe.get(representative_id, set())
    representative_ocr = normalized_tokens(evidence_for(representative_id)["ocr_scene_text"])
    candidates = []
    for row in rows:
        keyframe_id = str(row.keyframe_id)
        if keyframe_id == representative_id:
            continue
        object_distance = jaccard_distance(
            representative_objects,
            object_signature_by_keyframe.get(keyframe_id, set()),
        )
        ocr_distance = jaccard_distance(
            representative_ocr,
            normalized_tokens(evidence_for(keyframe_id)["ocr_scene_text"]),
        )
        novelty = max(
            object_distance / max(OBJECT_CHANGE_THRESHOLD, 1e-6),
            ocr_distance / max(OCR_CHANGE_THRESHOLD, 1e-6),
        )
        candidates.append((novelty, object_distance, ocr_distance, frame_selection_score(row), row))

    candidates.sort(key=lambda item: (item[0], item[3]), reverse=True)
    for novelty, object_distance, ocr_distance, score, row in candidates:
        if len(selected) >= MAX_VLM_FRAMES_PER_SHOT:
            break
        if object_distance < OBJECT_CHANGE_THRESHOLD and ocr_distance < OCR_CHANGE_THRESHOLD:
            continue
        selected.append(row)
        selection_rows.append(
            {
                "keyframe_id": str(row.keyframe_id),
                "shot_id": str(shot_id),
                "frame_idx": int(row.frame_idx),
                "reason": "trusted_object_or_scene_ocr_change",
                "selection_score": score,
                "object_distance": object_distance,
                "ocr_distance": ocr_distance,
            }
        )

selection_df = pd.DataFrame(selection_rows).sort_values(["frame_idx", "keyframe_id"]).reset_index(drop=True)
selection_df.to_parquet(OUTPUT_ROOT / "vlm-selection.parquet", index=False)
VLM_KEYFRAME_IDS = set(selection_df["keyframe_id"].astype(str))
print("VLM calls:", len(selection_df), "for", keyframes_df["shot_id"].nunique(), "shots")
display(selection_df.head(30))
'''


SCHEMA_CELL = r'''ENVIRONMENT_VALUES = {"indoor", "outdoor", "unknown"}
TIME_VALUES = {"daytime", "night", "unknown"}
WEATHER_VALUES = {"rain", "sunny", "unknown"}
FILTER_OPTIONS = [
    ("daytime", "Daytime"),
    ("night", "Night"),
    ("rain", "Rain"),
    ("sunny", "Sunny"),
    ("indoor", "Indoor"),
    ("outdoor", "Outdoor"),
]
FILTER_VALUES = {value for value, _ in FILTER_OPTIONS}

SPECIAL_TARGETS = (
    "crocodile, alligator, dolphin, python, snake, horse, dog, fish, "
    "carrot, potato, apple, tomato and other clearly visible specific species or produce"
)

SYSTEM_PROMPT = f"""
You analyze one video keyframe for a Vietnamese multimedia search system.
Return exactly one valid JSON object and no markdown. Write caption_vi,
name_vi, action_vi and search_text_vi in Vietnamese.

The image is authoritative. COCO object detections and OCR are noisy hints only;
ignore a hint when it conflicts with the image. Identify specific long-tail
species or produce when visually clear, including {SPECIAL_TARGETS}. For a
person, describe only visible presentation and use male_presenting,
female_presenting or unknown; never infer identity, ethnicity, religion, health
or private facts.

Scene rules:
- environment is indoor, outdoor or unknown;
- time_of_day is daytime, night or unknown;
- weather is rain, sunny or unknown;
- an operating room, surgery room, hospital room, studio, office, classroom,
  kitchen or other enclosed room is indoor;
- use rain only for visible rain, active rainy conditions or a strong visual sign;
- use sunny only for clearly visible sunlight or a clearly sunny sky;
- use unknown when the image does not prove a value;
- do not return detailed place categories or content tags.

Required JSON schema:
{{
  "caption_vi": "Một hoặc hai câu tiếng Việt, ngắn và chỉ nêu điều nhìn thấy",
  "main_subjects": [{{
    "name_en": "canonical English name",
    "name_vi": "tên tiếng Việt",
    "category": "person|animal|food|vegetable|fruit|vehicle|object|text|other",
    "species_en": "specific species or empty string",
    "species_vi": "loài cụ thể hoặc chuỗi rỗng",
    "dominant_colors": ["basic visible color"],
    "count": 1,
    "action_vi": "hành động nhìn thấy hoặc chuỗi rỗng",
    "prominence": "primary|secondary|background",
    "apparent_gender_presentation": "male_presenting|female_presenting|unknown|not_applicable",
    "age_group": "child|adult|older_adult|unknown|not_applicable",
    "confidence": 0.0
  }}],
  "visible_text": ["chữ nhìn rõ trong ảnh"],
  "actions_vi": ["hành động ngắn bằng tiếng Việt"],
  "environment": "indoor|outdoor|unknown",
  "time_of_day": "daytime|night|unknown",
  "weather": "rain|sunny|unknown",
  "search_text_vi": "cụm từ tiếng Việt gồm chủ thể, loài, màu, hành động và chữ nhìn thấy",
  "confidence": 0.0
}}

Keep at most eight main subjects. Confidence must be between 0 and 1.
"""

SCENE_DISPLAY = dict(FILTER_OPTIONS)
'''


VLM_HELPERS_CELL = r'''@lru_cache(maxsize=4096)
def resolve_frame_path(frame_value: str) -> Path:
    value = Path(frame_value)
    if value.is_absolute() and value.exists():
        return value
    candidates = [INPUT_ROOT / value, INPUT_ROOT.parent / value, INPUT_ROOT / "frames" / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    basename_matches = list(INPUT_ROOT.rglob(value.name))
    if basename_matches:
        return basename_matches[0]
    raise FileNotFoundError(frame_value)


def prompt_for(keyframe_id: str) -> str:
    evidence = evidence_for(keyframe_id)
    return (
        "Trusted COCO object hints (may be wrong): "
        + json.dumps(evidence["objects"], ensure_ascii=False, separators=(",", ":"))
        + "\nScene OCR (normal evidence): "
        + json.dumps(evidence["ocr_scene_text"], ensure_ascii=False)
        + "\nTicker OCR (low-weight evidence; do not use alone for scene): "
        + json.dumps(evidence["ocr_ticker_text"][:500], ensure_ascii=False)
        + "\nAnalyze the image and return the required JSON object."
    )


def extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip().replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response does not contain a JSON object.")
    return json.loads(text[start : end + 1])


def enum_value(value, allowed, default="unknown") -> str:
    normalized = normalize_text(value).casefold().replace(" ", "_")
    return normalized if normalized in allowed else default


def clean_string_list(value, limit=12) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        text = normalize_text(item)
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def clean_subjects(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        category = enum_value(
            item.get("category"),
            {"person", "animal", "food", "vegetable", "fruit", "vehicle", "object", "text", "other"},
            "other",
        )
        gender = enum_value(
            item.get("apparent_gender_presentation"),
            {"male_presenting", "female_presenting", "unknown", "not_applicable"},
            "unknown" if category == "person" else "not_applicable",
        )
        age_group = enum_value(
            item.get("age_group"),
            {"child", "adult", "older_adult", "unknown", "not_applicable"},
            "unknown" if category == "person" else "not_applicable",
        )
        try:
            count = max(1, int(item.get("count"))) if item.get("count") is not None else None
        except (TypeError, ValueError):
            count = None
        output.append(
            {
                "name_en": normalize_text(item.get("name_en")),
                "name_vi": normalize_text(item.get("name_vi")),
                "category": category,
                "species_en": normalize_text(item.get("species_en")),
                "species_vi": normalize_text(item.get("species_vi")),
                "dominant_colors": clean_string_list(item.get("dominant_colors"), 4),
                "count": count,
                "action_vi": normalize_text(item.get("action_vi")),
                "prominence": enum_value(item.get("prominence"), {"primary", "secondary", "background"}, "secondary"),
                "apparent_gender_presentation": gender,
                "age_group": age_group,
                "confidence": min(1.0, max(0.0, safe_float(item.get("confidence"), 0.0))),
            }
        )
    return output


def normalize_vlm_result(value: dict) -> dict:
    return {
        "caption_vi": normalize_text(value.get("caption_vi")),
        "main_subjects": clean_subjects(value.get("main_subjects")),
        "visible_text": clean_string_list(value.get("visible_text")),
        "actions_vi": clean_string_list(value.get("actions_vi")),
        "environment": enum_value(value.get("environment"), ENVIRONMENT_VALUES),
        "time_of_day": enum_value(value.get("time_of_day"), TIME_VALUES),
        "weather": enum_value(value.get("weather"), WEATHER_VALUES),
        "search_text_vi": normalize_text(value.get("search_text_vi")),
        "confidence": min(1.0, max(0.0, safe_float(value.get("confidence"), 0.0))),
    }


def _build_messages(image, keyframe_id: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_for(keyframe_id)},
            ],
        },
    ]


def run_vlm_batch(
    frame_paths: list[Path], keyframe_ids: list[str]
) -> list[tuple[dict | None, str, str | None]]:
    """Run the VLM on up to VLM_BATCH_SIZE frames in one generate() call.

    Returns one (result, raw_text, error) triple per input, same order as the
    inputs. A JSON-parse failure on one frame does not affect the others.
    """
    images = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as source:
            images.append(source.convert("RGB"))
    messages_batch = [
        _build_messages(image, keyframe_id)
        for image, keyframe_id in zip(images, keyframe_ids)
    ]
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=prompt_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.03,
        )
    trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    raw_texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    outputs = []
    for raw_text in raw_texts:
        try:
            outputs.append((normalize_vlm_result(extract_json_object(raw_text)), raw_text, None))
        except Exception as exc:
            outputs.append((None, raw_text, f"{type(exc).__name__}: {exc}"))
    return outputs


def run_vlm(frame_path: Path, keyframe_id: str) -> tuple[dict, str]:
    """Single-frame path, used as a fallback when a batch call itself raises
    (e.g. a corrupt image) so one bad frame does not drop the rest of its batch."""
    result, raw_text, error = run_vlm_batch([frame_path], [keyframe_id])[0]
    if error is not None:
        raise ValueError(error)
    return result, raw_text


print("VLM inference helpers ready.")
'''


OUTPUT_CELL = r'''SCENE_LABEL_FIELDS = ("time_of_day", "weather", "environment")
SCENE_VI = {
    "daytime": "ban ngày",
    "night": "ban đêm",
    "rain": "trời mưa",
    "sunny": "trời nắng",
    "indoor": "trong nhà",
    "outdoor": "ngoài trời",
}

SCENE_CUES = {
    "indoor": ("phòng", "phòng mổ", "phòng phẫu thuật", "bệnh viện", "studio", "trong nhà", "văn phòng", "lớp học", "nhà bếp", "phòng khách", "hành lang"),
    "outdoor": ("ngoài trời", "đường phố", "cánh đồng", "bờ sông", "con sông", "biển", "bãi biển", "công viên", "rừng", "đồng lúa", "trên thuyền", "kayak"),
    "night": ("ban đêm", "ban tối", "nửa đêm", "nighttime", "at night", "trời tối"),
    "daytime": ("ban ngày", "buổi sáng", "buổi trưa", "daytime", "day light", "ánh nắng", "trời sáng"),
    "rain": ("trời mưa", "đang mưa", "mưa lớn", "mưa to", "rainy", "in the rain"),
    "sunny": ("trời nắng", "nắng rõ", "ánh nắng", "sunny", "bright sunlight"),
    "uncertain_time": ("hoàng hôn", "chạng vạng", "dusk", "twilight"),
}


def has_cue(text: str, cues: tuple[str, ...]) -> bool:
    value = normalize_text(text).casefold()
    return any(cue.casefold() in value for cue in cues)


def crosscheck_scene(result: dict) -> tuple[dict, list[str]]:
    checked = dict(result)
    # Scene decisions must come from the generated caption/actions. Visible
    # ticker text can mention an unrelated place or time and must not override
    # the image-based scene classification.
    evidence_text = normalize_text(
        " ".join(
            [
                result.get("caption_vi", ""),
                *result.get("actions_vi", []),
            ]
        )
    )
    notes = []
    if has_cue(evidence_text, SCENE_CUES["indoor"]):
        if checked.get("environment") != "indoor":
            notes.append("caption_forced_indoor")
        checked["environment"] = "indoor"
    elif has_cue(evidence_text, SCENE_CUES["outdoor"]):
        if checked.get("environment") != "outdoor":
            notes.append("caption_forced_outdoor")
        checked["environment"] = "outdoor"

    if has_cue(evidence_text, SCENE_CUES["night"]):
        if checked.get("time_of_day") != "night":
            notes.append("caption_forced_night")
        checked["time_of_day"] = "night"
    elif has_cue(evidence_text, SCENE_CUES["daytime"]):
        if checked.get("time_of_day") != "daytime":
            notes.append("caption_forced_daytime")
        checked["time_of_day"] = "daytime"
    elif has_cue(evidence_text, SCENE_CUES["uncertain_time"]):
        if checked.get("time_of_day") != "unknown":
            notes.append("caption_cleared_uncertain_time")
        checked["time_of_day"] = "unknown"

    if has_cue(evidence_text, SCENE_CUES["rain"]):
        if checked.get("weather") != "rain":
            notes.append("caption_forced_rain")
        checked["weather"] = "rain"
    elif has_cue(evidence_text, SCENE_CUES["sunny"]):
        if checked.get("weather") != "sunny":
            notes.append("caption_forced_sunny")
        checked["weather"] = "sunny"

    if checked.get("environment") == "indoor" and checked.get("weather") != "unknown":
        checked["weather"] = "unknown"
        notes.append("indoor_cleared_weather")
    return checked, notes


def scene_labels(result: dict) -> list[str]:
    labels = []
    for field in SCENE_LABEL_FIELDS:
        value = result.get(field, "unknown")
        if value in FILTER_VALUES and value not in labels:
            labels.append(value)
    return labels


def subject_terms(subjects: list[dict]) -> list[str]:
    terms = []
    for subject in subjects:
        for field in ("name_vi", "species_vi", "name_en", "species_en", "action_vi"):
            value = normalize_text(subject.get(field))
            if value and value not in terms:
                terms.append(value)
        for color in subject.get("dominant_colors", []):
            color = normalize_text(color)
            if color and color not in terms:
                terms.append(color)
    return terms


def unique_join(parts) -> str:
    output = []
    seen = set()
    for part in parts:
        text = normalize_text(part)
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return " ".join(output)


def unknown_result() -> dict:
    return {
        "caption_vi": "",
        "main_subjects": [],
        "visible_text": [],
        "actions_vi": [],
        "environment": "unknown",
        "time_of_day": "unknown",
        "weather": "unknown",
        "search_text_vi": "",
        "confidence": 0.0,
    }


successful_direct = {
    keyframe_id: item
    for keyframe_id, item in completed.items()
    if keyframe_id in VLM_KEYFRAME_IDS and item.get("result") and not item.get("vlm_error")
}
direct_by_shot = {}
for item in successful_direct.values():
    direct_by_shot.setdefault(str(item["shot_id"]), []).append(item)

records = []
for row in keyframes_df.sort_values("frame_idx").itertuples(index=False):
    keyframe_id = str(row.keyframe_id)
    direct_item = successful_direct.get(keyframe_id)
    if direct_item is not None:
        source_item = direct_item
        description_source = "vlm"
    else:
        candidates = direct_by_shot.get(str(row.shot_id), [])
        source_item = min(candidates, key=lambda item: abs(int(item["frame_idx"]) - int(row.frame_idx))) if candidates else None
        description_source = "propagated" if source_item is not None else "evidence_only"

    result = dict(source_item["result"]) if source_item is not None else unknown_result()
    result, scene_notes = crosscheck_scene(result)
    evidence = evidence_for(keyframe_id)
    labels = scene_labels(result)

    # Ticker is retained as a separate, lower-weight evidence source. Do not add
    # merged shot ticker text, which would cause repeated false ranking boosts.
    search_text_vi = unique_join(
        [
            result.get("caption_vi", ""),
            result.get("search_text_vi", ""),
            *subject_terms(result.get("main_subjects", [])),
            *result.get("visible_text", []),
            *result.get("actions_vi", []),
            evidence["ocr_scene_text"],
            " ".join(evidence["object_labels"]),
            " ".join(SCENE_VI.get(label, label) for label in labels),
        ]
    )
    source_keyframe_id = str(source_item["keyframe_id"]) if source_item is not None else None
    records.append(
        {
            "keyframe_id": keyframe_id,
            "video_id": str(row.video_id),
            "frame_idx": int(row.frame_idx),
            "pts_time": float(row.pts_time),
            "shot_id": str(row.shot_id),
            "frame_path": str(row.frame_path),
            "caption_vi": result.get("caption_vi", ""),
            "search_text_vi": search_text_vi,
            "main_subjects": result.get("main_subjects", []),
            "visible_text": result.get("visible_text", []),
            "actions_vi": result.get("actions_vi", []),
            "environment": result.get("environment", "unknown"),
            "time_of_day": result.get("time_of_day", "unknown"),
            "weather": result.get("weather", "unknown"),
            "scene_labels": labels,
            "scene_crosscheck": scene_notes,
            "confidence": safe_float(result.get("confidence"), 0.0),
            "ocr_text": evidence["ocr_text"],
            "ocr_scene_text": evidence["ocr_scene_text"],
            "ocr_ticker_text": evidence["ocr_ticker_text"],
            "ocr_ticker_weight": OCR_TICKER_SEARCH_WEIGHT,
            "object_labels": evidence["object_labels"],
            "object_hints": evidence["objects"],
            "description_source": description_source,
            "vlm_source_keyframe_id": source_keyframe_id,
            "vlm_inferred": description_source == "vlm",
            "vlm_error": completed.get(keyframe_id, {}).get("vlm_error"),
            "pipeline_version": PIPELINE_VERSION,
            "model": MODEL_NAME,
        }
    )

understanding_df = pd.DataFrame(records).sort_values("frame_idx").reset_index(drop=True)
understanding_df.to_json(OUTPUT_ROOT / "frame-understanding.jsonl", orient="records", lines=True, force_ascii=False)

parquet_df = understanding_df.copy()
for column in (
    "main_subjects", "visible_text", "actions_vi", "scene_labels",
    "scene_crosscheck", "object_labels", "object_hints",
):
    parquet_df[f"{column}_json"] = parquet_df[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    parquet_df = parquet_df.drop(columns=[column])
parquet_df.to_parquet(OUTPUT_ROOT / "frame-understanding.parquet", index=False)

search_documents_df = pd.DataFrame(
    {
        "keyframe_id": understanding_df["keyframe_id"],
        "video_id": understanding_df["video_id"],
        "frame_idx": understanding_df["frame_idx"],
        "pts_time": understanding_df["pts_time"],
        "shot_id": understanding_df["shot_id"],
        "caption_vi": understanding_df["caption_vi"],
        "search_text_vi": understanding_df["search_text_vi"],
        "ocr_text": understanding_df["ocr_text"],
        "ocr_scene_text": understanding_df["ocr_scene_text"],
        "ocr_ticker_text": understanding_df["ocr_ticker_text"],
        "ocr_ticker_weight": understanding_df["ocr_ticker_weight"],
        "environment": understanding_df["environment"],
        "time_of_day": understanding_df["time_of_day"],
        "weather": understanding_df["weather"],
        "scene_labels_json": understanding_df["scene_labels"].map(lambda value: json.dumps(value, ensure_ascii=False)),
        "main_subjects_json": understanding_df["main_subjects"].map(lambda value: json.dumps(value, ensure_ascii=False)),
        "object_labels_text": understanding_df["object_labels"].map(lambda value: " ".join(value)),
        "scene_crosscheck_json": understanding_df["scene_crosscheck"].map(lambda value: json.dumps(value, ensure_ascii=False)),
        "description_source": understanding_df["description_source"],
        "vlm_source_keyframe_id": understanding_df["vlm_source_keyframe_id"],
        "confidence": understanding_df["confidence"],
    }
)
search_documents_df.to_parquet(OUTPUT_ROOT / "search-documents.parquet", index=False)
search_documents_df.to_json(OUTPUT_ROOT / "search-documents.jsonl", orient="records", lines=True, force_ascii=False)

source_rank = {"vlm": 0, "propagated": 1, "evidence_only": 2}
shot_scenes_df = understanding_df.copy()
shot_scenes_df["_source_rank"] = shot_scenes_df["description_source"].map(source_rank)
shot_scenes_df = (
    shot_scenes_df.sort_values(["shot_id", "_source_rank", "confidence"], ascending=[True, True, False])
    .groupby("shot_id", as_index=False)
    .first()
)
shot_scenes_df = shot_scenes_df[
    ["shot_id", "video_id", "keyframe_id", "frame_idx", "pts_time", "caption_vi", "search_text_vi", "environment", "time_of_day", "weather", "confidence"]
]
shot_scenes_df.to_parquet(OUTPUT_ROOT / "shot-scenes.parquet", index=False)

print("Frame-understanding rows:", len(understanding_df))
display(understanding_df[["frame_idx", "caption_vi", "environment", "time_of_day", "weather", "scene_labels", "description_source", "scene_crosscheck"]].head(30))
'''


def update_frame_understanding() -> None:
    path = ROOT / "pipelines/kaggle/notebooks/06-frame-understanding/06-frame-understanding-single-video.ipynb"
    notebook = load_notebook(path)
    set_cell(
        notebook,
        0,
        """# 06 - Frame understanding and six general scene filters (full-video default)

This stage joins notebook 01 keyframes, notebook 03 OCR and notebook 05 fixed
COCO objects by `keyframe_id`, then uses Qwen2.5-VL to write Vietnamese captions
and structured `main_subjects`. Prompt-free YOLOE detections are rejected.

The production default covers every shot in one selected video. It runs one VLM
call per shot and a second only when trusted COCO or scene OCR evidence changes;
nearby keyframes receive propagated metadata to keep runtime practical. Frames run
through the VLM in batches of `VLM_BATCH_SIZE` per `model.generate()` call (falls
back to one-by-one on a batch failure), and the model loads with flash-attention 2
when available, sdpa otherwise -- both cut wall-clock time without changing output
quality.
""",
    )
    set_cell(
        notebook,
        2,
        """from __future__ import annotations

import gc
import json
import math
import re
import shutil
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from PIL import Image

VIDEO_ID = "L21_V001"
PIPELINE_VERSION = "aicv3-frame-understanding-v3-coco-ocr-scene-check"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
LOAD_IN_4BIT = True
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
MAX_NEW_TOKENS = 512

# Full-video production defaults. Set MAX_SHOTS to a positive integer for smoke testing.
MAX_SHOTS = None
MAX_VLM_FRAMES_PER_SHOT = 2
# Frames processed per model.generate() call. Batching is the main lever for
# the ~30 min/video runtime: decoding on a 4-bit 7B model is memory-bandwidth
# bound, so batching amortizes weight reads across several frames per step
# instead of paying them once per single frame. Start at 4 and reduce to 2 or
# 1 if you hit CUDA OOM on your GPU; raise it if you have headroom to spare.
VLM_BATCH_SIZE = 4
OBJECT_CHANGE_THRESHOLD = 0.45
OCR_CHANGE_THRESHOLD = 0.65
OBJECT_HINT_LIMIT = 12
OBJECT_MIN_CONFIDENCE = 0.25
OCR_TICKER_SEARCH_WEIGHT = 0.15
CHECKPOINT_EVERY = 5
MAKE_ARCHIVE = False

TRUSTED_COCO_MODEL_STEMS = {"yolo26m", "yolo26s", "yolo26n", "yolo26l", "yolo26x"}
COCO_OBJECT_WHITELIST = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before loading the VLM.")

OUTPUT_ROOT = Path(f"/kaggle/working/06-frame-understanding-{VIDEO_ID}")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_ROOT / "vlm-checkpoint.jsonl"
print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL_NAME)
print("Output:", OUTPUT_ROOT)
""",
    )
    set_cell(notebook, 3, FRAME_INPUT_CELL)
    set_cell(notebook, 4, FRAME_EVIDENCE_CELL)
    set_cell(notebook, 5, FRAME_SELECTION_CELL)
    set_cell(notebook, 6, SCHEMA_CELL)
    set_cell(notebook, 8, VLM_HELPERS_CELL)
    set_cell(notebook, 10, OUTPUT_CELL)
    # Keep the existing checkpoint/inference cell (9), but ensure it uses the new version.
    source9 = cell_source(notebook, 9)
    if "PIPELINE_VERSION" not in source9:
        raise RuntimeError("Unexpected frame-understanding inference cell")
    set_cell(
        notebook,
        11,
        r'''label_counts = Counter(label for labels in understanding_df["scene_labels"] for label in labels)
options = [
    {"value": value, "label": label, "count": int(label_counts.get(value, 0))}
    for value, label in FILTER_OPTIONS
]
facet_payload = {"all_option": {"value": "all", "label": "All scenes"}, "options": options}
(OUTPUT_ROOT / "scene-facets.json").write_text(json.dumps(facet_payload, ensure_ascii=False, indent=2), encoding="utf-8")

summary = {
    "stage": "frame-understanding",
    "pipeline_version": PIPELINE_VERSION,
    "video_id": VIDEO_ID,
    "model": MODEL_NAME,
    "object_model_policy": "trusted-yolo26-coco-only",
    "trusted_object_rows": int(len(objects_df)),
    "ocr_ticker_weight": OCR_TICKER_SEARCH_WEIGHT,
    "scene_crosschecks": int(sum(bool(value) for value in understanding_df["scene_crosscheck"])),
    "quantization": "4bit-nf4" if LOAD_IN_4BIT else "fp16",
    "scene_filter_values": [value for value, _ in FILTER_OPTIONS],
    "keyframes_output": int(len(understanding_df)),
    "shots_output": int(understanding_df["shot_id"].nunique()),
    "direct_vlm_frames": int((understanding_df["description_source"] == "vlm").sum()),
    "propagated_frames": int((understanding_df["description_source"] == "propagated").sum()),
    "evidence_only_frames": int((understanding_df["description_source"] == "evidence_only").sum()),
    "vlm_errors": int(vlm_error_count),
    "max_shots": MAX_SHOTS,
    "max_vlm_frames_per_shot": MAX_VLM_FRAMES_PER_SHOT,
    "outputs": [
        "frame-understanding.jsonl", "frame-understanding.parquet", "search-documents.jsonl",
        "search-documents.parquet", "shot-scenes.parquet", "scene-facets.json", "vlm-selection.parquet",
    ],
}
(OUTPUT_ROOT / "frame-understanding-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(facet_payload, ensure_ascii=False, indent=2))
if MAKE_ARCHIVE:
    archive = shutil.make_archive(f"/kaggle/working/06-frame-understanding-{VIDEO_ID}", "zip", root_dir=OUTPUT_ROOT)
    print("Download:", archive)
''',
    )
    set_cell(
        notebook,
        12,
        r'''# Optional visual audit of direct VLM frames.
PREVIEW_COUNT = 8
if PREVIEW_COUNT > 0:
    import matplotlib.pyplot as plt
    samples = understanding_df[understanding_df["description_source"] == "vlm"].head(PREVIEW_COUNT)
    for row in samples.itertuples(index=False):
        frame_path = resolve_frame_path(str(row.frame_path))
        image = Image.open(frame_path).convert("RGB")
        figure, axis = plt.subplots(figsize=(12, 7))
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{row.video_id} | frame={row.frame_idx} | {row.pts_time:.2f}s\n"
            f"{row.caption_vi}\n{', '.join(row.scene_labels)}\n"
            f"crosscheck={', '.join(row.scene_crosscheck)}",
            fontsize=10,
            loc="left",
        )
        figure.tight_layout()
        plt.show()
        plt.close(figure)
del model
gc.collect()
torch.cuda.empty_cache()
print("Frame-understanding stage complete; GPU memory released.")
''',
    )
    save_notebook(path, notebook)


def main() -> None:
    update_ocr()
    update_objects()
    update_frame_understanding()
    print("Updated production notebooks.")


if __name__ == "__main__":
    main()
