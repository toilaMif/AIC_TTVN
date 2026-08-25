"""Enrich AIC 2026 media metadata with ASR-grounded Vietnamese summaries.

The source YouTube description is preserved in ``source_description``. The
``description`` field is replaced because it is the field imported into the
``videos`` table by ``retrieval.storage.metadata_import``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VIDEO_ID = re.compile(r"^L\d+_V\d+$")
WORD = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)

STOPWORDS = {
    "ai",
    "anh",
    "ban",
    "bang",
    "bi",
    "biet",
    "cac",
    "cach",
    "can",
    "chi",
    "cho",
    "chu",
    "chung",
    "co",
    "con",
    "cua",
    "cung",
    "da",
    "dang",
    "day",
    "de",
    "den",
    "di",
    "do",
    "duoc",
    "gi",
    "hay",
    "hien",
    "hon",
    "khi",
    "khong",
    "la",
    "lai",
    "lam",
    "len",
    "luc",
    "ma",
    "minh",
    "mot",
    "muon",
    "nao",
    "nay",
    "neu",
    "ngay",
    "nguoi",
    "nhieu",
    "nhung",
    "noi",
    "o",
    "qua",
    "ra",
    "rat",
    "roi",
    "sau",
    "se",
    "su",
    "tai",
    "tat",
    "the",
    "theo",
    "thi",
    "thoi",
    "trong",
    "truoc",
    "tu",
    "tung",
    "va",
    "van",
    "ve",
    "vi",
    "voi",
    "vua",
}

BOILERPLATE_MARKERS = (
    "cam on cac ban da theo doi",
    "hen gap lai",
    "hay subscribe",
    "khong bo lo nhung video",
    "dang ky kenh",
    "subscribe kenh",
    "fanpage",
    "youtube channel",
    "ban quyen",
    "theo doi thong tin tai",
    "chuong trinh duoc tai tro",
    "cong ty ajinomoto viet nam",
    "xin chao tat ca cac ban",
    "chao mung quy vi den voi chuong trinh",
    "rat han hanh duoc gap lai",
)

PROMOTIONAL_DESCRIPTION_MARKERS = (
    "http://",
    "https://",
    "www.",
    "#",
    "đăng ký",
    "subscribe",
    "fanpage",
    "tiktok",
    "youtube",
    "website",
    "bản quyền",
    "copyright",
    "điện thoại",
    "email",
    "theo dõi các kênh",
    "các show hấp dẫn khác",
    "theo dõi thông tin",
    "phát sóng lúc",
    "hàng tuần trên kênh",
    "htv entertainment",
    "full playlist",
    "các chương trình hấp dẫn khác",
    "tin tức nhanh nhất",
    "kênh thông tin chính thức",
    "đồng hành cùng",
    "cảm ơn",
    "video nóng",
    "tin thời sự",
    "tin tức thế giới",
    "địa chỉ",
    "đường dây nóng",
    "tòa soạn",
    "kênh chính thức",
)

COMMON_ASR_FIXES = (
    (r"\bsông Cổ Long\b", "sông Cửu Long"),
    (r"\bsông Củ Long\b", "sông Cửu Long"),
    (r"\bsụt lúng\b", "sụt lún"),
    (r"\bsạt lỡ\b", "sạt lở"),
    (r"\bnước biển dân\b", "nước biển dâng"),
    (r"\bthiên tài\b", "thiên tai"),
    (r"\bxã lũ\b", "xả lũ"),
    (r"\bmạnh thần quân\b", "mạnh thường quân"),
    (r"\bthiên nguyện\b", "thiện nguyện"),
    (r"\bđầy ấp\b", "đầy ắp"),
    (r"\btrạng đua\b", "chặng đua"),
    (r"\bbà đồng viên\b", "vận động viên"),
    (r"\btân tốc\b", "tăng tốc"),
    (r"\bthết đải\b", "thiết đãi"),
    (r"\btiêu say\b", "tiêu xay"),
)

GROUP_LABELS = {
    "L21": "Thời sự - Bản tin 60 Giây Sáng",
    "L22": "Thời sự - Bản tin 60 Giây Chiều",
    "L23": "Thể thao - Đua xe đạp Cúp Truyền hình TP.HCM 2024",
    "L24": "Thể thao biểu diễn - Lân sư rồng Cúp Chợ Lớn 2024",
    "L25": "Giáo dục - Ôn thi tốt nghiệp THPT 2024",
    "L26": "Ẩm thực - Hướng dẫn nấu ăn",
    "L27": "Du lịch và ẩm thực Việt Nam",
    "L28": "Phim tài liệu - Văn hóa và đời sống vùng Mê Kông",
    "L29": "Phóng sự - Con người và sinh kế vùng Mê Kông",
    "L30": "Cộng đồng - Câu chuyện truyền cảm hứng",
}

GROUP_CUES = {
    "L23": ("tay đua", "về đích", "nước rút", "chiến thắng", "áo vàng", "chặng đua"),
    "L25": ("bài", "câu", "đáp án", "công thức", "phương pháp", "ví dụ", "lưu ý", "kiến thức"),
    "L26": (
        "nguyên liệu",
        "ướp",
        "trộn",
        "chiên",
        "xào",
        "nấu",
        "hấp",
        "nướng",
        "luộc",
        "cắt",
        "muỗng",
        "nêm",
        "sốt",
        "cho vào",
        "trình bày",
    ),
    "L27": ("khám phá", "đặc sản", "văn hóa", "du lịch", "ẩm thực", "địa điểm", "làng nghề"),
    "L28": ("sông", "vùng đất", "người dân", "lịch sử", "văn hóa", "sinh kế", "Mê Kông"),
    "L29": ("nghề", "người dân", "Cà Mau", "Mê Kông", "sinh kế", "văn hóa", "làng nghề"),
    "L30": ("giúp", "tặng", "hỗ trợ", "cộng đồng", "khó khăn", "ước mơ", "tình nguyện", "lan tỏa"),
}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    normalized: str


@dataclass
class AsrRecord:
    video_id: str
    transcript_paths: list[Path]
    chosen_path: Path | None
    transcript: str
    transcript_hash: str | None
    segments: list[Segment]
    summary: dict[str, object]
    error: str | None


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    text: str
    normalized: str
    tokens: tuple[str, ...]
    score: float


def ascii_fold(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", value) if unicodedata.category(char) != "Mn"
    )


def normalize(value: str) -> str:
    value = ascii_fold(value).lower()
    value = re.sub(r"[^0-9a-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def words(value: str) -> list[str]:
    return [normalize(token) for token in WORD.findall(value) if normalize(token)]


def content_tokens(value: str) -> list[str]:
    return [token for token in words(value) if token not in STOPWORDS and len(token) > 1]


def clean_text(value: str, *, max_chars: int = 320) -> str:
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-–—,;:")
    for pattern, replacement in COMMON_ASR_FIXES:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r"^(thưa|kính thưa) quý vị[, ]+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(các bạn ơi|các bạn thân mến)[, ]+", "", value, flags=re.IGNORECASE)
    if len(value) > max_chars:
        shortened = value[: max_chars + 1].rsplit(" ", 1)[0]
        value = shortened.rstrip(" ,;:") + "…"
    if value:
        value = value[0].upper() + value[1:]
    if value and value[-1] not in ".!?…":
        value += "."
    return value


def is_boilerplate(value: str) -> bool:
    folded = normalize(value)
    return any(marker in folded for marker in BOILERPLATE_MARKERS)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_texts(values: Iterable[str], *, threshold: float = 0.72) -> list[str]:
    result: list[str] = []
    result_tokens: list[list[str]] = []
    for value in values:
        cleaned = clean_text(value)
        tokens = content_tokens(cleaned)
        if len(tokens) < 3:
            continue
        if any(jaccard(tokens, existing) >= threshold for existing in result_tokens):
            continue
        result.append(cleaned)
        result_tokens.append(tokens)
    return result


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_segments(path: Path) -> list[Segment]:
    result: list[Segment] = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if text:
            result.append(
                Segment(
                    start=float(item.get("start_time", 0.0)),
                    end=float(item.get("end_time", 0.0)),
                    text=text,
                    normalized=normalize(text),
                )
            )
    return result


def discover_asr(asr_root: Path) -> dict[str, AsrRecord]:
    transcript_paths: dict[str, list[Path]] = defaultdict(list)
    error_paths: dict[str, list[Path]] = defaultdict(list)

    for path in asr_root.rglob("transcript.txt"):
        if VIDEO_ID.fullmatch(path.parent.name):
            transcript_paths[path.parent.name].append(path)
    for path in asr_root.rglob("_ERROR.txt"):
        if VIDEO_ID.fullmatch(path.parent.name):
            error_paths[path.parent.name].append(path)

    records: dict[str, AsrRecord] = {}
    for video_id in sorted(set(transcript_paths) | set(error_paths)):
        paths = sorted(transcript_paths.get(video_id, []), key=lambda item: str(item).lower())
        choices: list[tuple[int, str, Path, str]] = []
        for path in paths:
            transcript = path.read_text(encoding="utf-8", errors="replace").strip()
            digest = hashlib.sha256(normalize(transcript).encode("utf-8")).hexdigest()
            choices.append((len(content_tokens(transcript)), digest, path, transcript))
        choices.sort(key=lambda item: (-item[0], str(item[2]).lower()))

        if choices:
            _, digest, chosen, transcript = choices[0]
            segments = read_segments(chosen.with_name("asr-segments.jsonl"))
            summary_path = chosen.with_name("asr-summary.json")
            summary = load_json(summary_path) if summary_path.exists() else {}
        else:
            chosen = None
            transcript = ""
            digest = None
            segments = []
            summary = {}

        error = None
        if error_paths.get(video_id):
            error = error_paths[video_id][0].read_text(encoding="utf-8", errors="replace").strip()

        records[video_id] = AsrRecord(
            video_id=video_id,
            transcript_paths=paths,
            chosen_path=chosen,
            transcript=transcript,
            transcript_hash=digest,
            segments=segments,
            summary=summary,
            error=error,
        )
    return records


def clean_source_description(value: str) -> list[str]:
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", value):
        lines = []
        for raw_line in block.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip(" -–—\t")
            if not line:
                continue
            # Descriptions often put the editorial blurb and channel promo on
            # the same physical line. Preserve the editorial prefix only.
            cut_markers = (
                "http://",
                "https://",
                " #",
                "****************",
                "👉",
                "📱",
                "►",
                "- Tin tức nhanh",
                "Tin tức nhanh:",
                "Facebook:",
                "TÒA SOẠN",
            )
            cut_at = min(
                (line.find(marker) for marker in cut_markers if line.find(marker) >= 0),
                default=len(line),
            )
            line = line[:cut_at].strip(" -–—|:")
            folded = line.lower()
            if not line or any(marker in folded for marker in PROMOTIONAL_DESCRIPTION_MARKERS):
                continue
            lines.append(line)
        paragraph = " ".join(lines).strip()
        if (
            len(content_tokens(paragraph)) >= 7
            and not is_boilerplate(paragraph)
            and not any(marker in paragraph.lower() for marker in PROMOTIONAL_DESCRIPTION_MARKERS)
        ):
            paragraphs.append(clean_text(paragraph, max_chars=460))
    return dedupe_texts(paragraphs, threshold=0.68)[:3]


def title_detail(group: str, title: str) -> str:
    detail = re.sub(r"\s+", " ", title).strip(" -|–—")
    if group == "L26":
        detail = re.split(r"\bM[ÓO]N NGON M[ỖO]I NG[ÀA]Y\b", detail, maxsplit=1, flags=re.IGNORECASE)[0]
        detail = re.sub(r"\bVIVU TV\b", "", detail, flags=re.IGNORECASE).strip(" -|–—")
    elif group == "L30":
        detail = re.sub(
            r"^Lan tỏa năng lượng tích cực\s*2024\s*", "", detail, flags=re.IGNORECASE
        ).strip(" -|–—")
    elif group == "L25":
        detail = re.sub(
            r"^BÍ QUYẾT ÔN THI THPT\s*2024\s*", "", detail, flags=re.IGNORECASE
        ).strip(" -|–—")
    elif group in {"L21", "L22"}:
        detail = re.sub(r"^60 Giây (Sáng|Chiều)\s*[-–—]?\s*", "", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\s*[-–—]?\s*HTV Tin Tức.*$", "", detail, flags=re.IGNORECASE)
        detail = detail.strip(" -|–—")
    elif group == "L28":
        detail = re.sub(
            r"^Tản Mạn Mê Kông,? Đến Và Ở Lại\s*", "", detail, flags=re.IGNORECASE
        ).strip(" -|–—")
    elif group == "L29":
        detail = re.sub(r"^Đôi Mắt M[eê]Kong\s*", "", detail, flags=re.IGNORECASE).strip(
            " -|–—"
        )
    return detail or title


def topic_for(group: str, title: str) -> str:
    label = GROUP_LABELS.get(group, "Nội dung video")
    detail = title_detail(group, title)
    if group in {"L21", "L22", "L28"}:
        return f"{label} - {detail}"
    return f"{label}: {detail}"


def topic_for_with_source(group: str, title: str, source_description: str) -> str:
    topic = topic_for(group, title)
    if group == "L24" and len(content_tokens(title_detail(group, title))) < 4:
        source_points = clean_source_description(source_description)
        if source_points:
            topic = f"{GROUP_LABELS[group]}: {source_points[0]}"
    return topic


def informative_segments(
    record: AsrRecord, segment_doc_frequency: Counter[str]
) -> list[Segment]:
    result: list[Segment] = []
    seen: set[str] = set()
    for segment in record.segments:
        if segment.normalized in seen or is_boilerplate(segment.text):
            continue
        seen.add(segment.normalized)
        tokens = content_tokens(segment.text)
        if len(tokens) < 3:
            continue
        if segment_doc_frequency[segment.normalized] >= 4 and len(tokens) < 18:
            continue
        result.append(segment)
    return result


def candidate_windows(
    segments: list[Segment],
    *,
    title_tokens: set[str],
    group: str,
) -> list[Candidate]:
    if not segments:
        return []
    token_frequency = Counter(
        token for segment in segments for token in content_tokens(segment.text)
    )
    max_frequency = max(token_frequency.values(), default=1)
    cues = tuple(normalize(cue) for cue in GROUP_CUES.get(group, ()))
    candidates: list[Candidate] = []

    for index, segment in enumerate(segments):
        text = segment.text
        end = segment.end
        cursor = index + 1
        while len(text) < 120 and cursor < len(segments):
            following = segments[cursor]
            if following.start - end > 2.5 or len(text) + len(following.text) > 280:
                break
            text = f"{text} {following.text}"
            end = following.end
            cursor += 1

        tokens = tuple(content_tokens(text))
        if len(tokens) < 6:
            continue
        centrality = sum(token_frequency[token] / max_frequency for token in tokens) / math.sqrt(
            len(tokens)
        )
        title_overlap = len(set(tokens) & title_tokens) / max(1, len(title_tokens))
        folded = normalize(text)
        cue_hits = sum(cue in folded for cue in cues)
        numeric_bonus = min(0.35, 0.08 * len(re.findall(r"\d", text)))
        length_bonus = min(0.45, len(tokens) / 80)
        score = centrality + 2.0 * title_overlap + 0.32 * cue_hits + numeric_bonus + length_bonus
        if normalize(segment.text).startswith(("xin chao", "chao cac", "chao mung")):
            score -= 1.2
        candidates.append(
            Candidate(
                start=segment.start,
                end=end,
                text=text,
                normalized=normalize(text),
                tokens=tokens,
                score=score,
            )
        )
    return candidates


def select_points(
    record: AsrRecord,
    segments: list[Segment],
    *,
    group: str,
    title: str,
    count: int,
) -> list[str]:
    candidates = candidate_windows(
        segments,
        title_tokens=set(content_tokens(title)),
        group=group,
    )
    if not candidates:
        return []

    duration = max(
        float(record.summary.get("duration_sec", 0.0) or 0.0),
        max((candidate.end for candidate in candidates), default=1.0),
    )
    selected: list[Candidate] = []
    used_bins: set[int] = set()
    while candidates and len(selected) < count:
        best: Candidate | None = None
        best_adjusted = float("-inf")
        for candidate in candidates:
            similarity = max(
                (jaccard(candidate.tokens, item.tokens) for item in selected), default=0.0
            )
            timeline_bin = min(4, int(5 * candidate.start / max(duration, 1.0)))
            coverage_bonus = 0.35 if timeline_bin not in used_bins else 0.0
            adjusted = candidate.score - 1.6 * similarity + coverage_bonus
            if adjusted > best_adjusted:
                best, best_adjusted = candidate, adjusted
        if best is None:
            break
        selected.append(best)
        used_bins.add(min(4, int(5 * best.start / max(duration, 1.0))))
        candidates = [
            candidate
            for candidate in candidates
            if candidate is not best and jaccard(candidate.tokens, best.tokens) < 0.76
        ]

    selected.sort(key=lambda item: item.start)
    return dedupe_texts((item.text for item in selected), threshold=0.67)[:count]


def news_headlines(record: AsrRecord) -> list[str]:
    cue_index = None
    for index, segment in enumerate(record.segments[:12]):
        folded = segment.normalized
        if any(
            phrase in folded
            for phrase in (
                "noi bat sau day",
                "noi dung dang chu y sau day",
                "noi dung chinh sau day",
                "thong tin duoc quan tam",
            )
        ):
            cue_index = index
            break
    start_index = (cue_index + 1) if cue_index is not None else 0
    values = []
    for segment in record.segments[start_index:]:
        if segment.start > 38.0:
            break
        if is_boilerplate(segment.text) or len(content_tokens(segment.text)) < 5:
            continue
        values.append(segment.text)
    return dedupe_texts(values, threshold=0.62)[:5]


def news_story_openings(record: AsrRecord, segments: list[Segment]) -> list[str]:
    """Return likely story leads after the opening headline block."""
    if not segments:
        return []
    values: list[str] = []
    previous_end = 0.0
    cue_phrases = (
        "trong phan sau",
        "tiep theo",
        "mo dau chuong trinh",
        "thong tin an ninh",
        "thong tin quoc te",
    )
    for segment in segments:
        if segment.start < 30.0:
            previous_end = segment.end
            continue
        gap = segment.start - previous_end
        if gap >= 2.4 or any(cue in segment.normalized for cue in cue_phrases):
            values.append(segment.text)
        previous_end = segment.end
    return dedupe_texts(values, threshold=0.58)[:6]


def status_for(record: AsrRecord, segments: list[Segment]) -> tuple[str, str]:
    if record.chosen_path is None:
        return ("no_speech" if record.error else "missing", "unavailable")
    total_words = len(words(record.transcript))
    informative_words = sum(len(words(segment.text)) for segment in segments)
    if total_words <= 25 or informative_words <= 20:
        return "insufficient", "low"
    if informative_words / max(total_words, 1) < 0.12:
        return "boilerplate_only", "low"
    if informative_words < 140:
        return "sparse", "low"

    average_logprob = float(record.summary.get("avg_logprob_mean", -0.5) or -0.5)
    if informative_words >= 700 and average_logprob >= -0.35:
        return "usable", "high"
    return "usable", "medium"


def combine_points(
    source_points: list[str], asr_points: list[str], *, limit: int
) -> list[str]:
    return dedupe_texts([*source_points, *asr_points], threshold=0.65)[:limit]


def summary_intro(group: str, title: str) -> str:
    detail = title_detail(group, title)
    if group == "L21":
        return f'Bản tin "60 Giây Sáng" ({detail}) tổng hợp các sự kiện đáng chú ý trong nước và quốc tế.'
    if group == "L22":
        return f'Bản tin "60 Giây Chiều" ({detail}) cập nhật thời sự, đời sống và tin quốc tế trong ngày.'
    if group == "L23":
        return f"Video thể thao ghi lại diễn biến {detail}."
    if group == "L24":
        return f"Video ghi lại tiết mục lân sư rồng {detail}."
    if group == "L25":
        return f"Bài giảng ôn thi tốt nghiệp THPT 2024 trình bày {detail}."
    if group == "L26":
        return f'Chương trình "Món ngon mỗi ngày" hướng dẫn chế biến {detail}.'
    if group == "L27":
        return f"Tập du lịch - ẩm thực khám phá {detail}."
    if group == "L28":
        return f'Phim tài liệu "Tản mạn Mê Kông, Đến và Ở Lại" ({detail}) tìm hiểu dòng sông, vùng đất và đời sống cư dân Nam Bộ.'
    if group == "L29":
        return f'Phóng sự "Đôi Mắt Mê Kông" giới thiệu {detail}.'
    if group == "L30":
        return f"Video kể câu chuyện truyền cảm hứng về {detail}."
    return f"Video có nội dung: {detail}."


def build_summary(
    *,
    group: str,
    title: str,
    status: str,
    points: list[str],
) -> str:
    intro = summary_intro(group, title)
    if points:
        bullets = "\n".join(f"- {point}" for point in points)
        return f"{intro}\n\nNội dung chính:\n{bullets}"
    if status == "no_speech":
        note = "ASR không nhận dạng được lời nói; chủ đề được xác định từ tiêu đề và metadata nguồn."
    elif status == "missing":
        note = "Không tìm thấy kết quả ASR; chủ đề được xác định từ tiêu đề và metadata nguồn."
    elif status == "boilerplate_only":
        note = "ASR chủ yếu chứa lời mời đăng ký kênh hoặc lời kết, không đủ dữ liệu để mô tả chi tiết diễn biến."
    else:
        note = "Lời thoại nhận dạng quá ít để tạo bản tóm tắt nội dung đáng tin cậy."
    return f"{intro}\n\nGhi chú dữ liệu: {note}"


def conflict_note(record: AsrRecord, asr_info: dict[str, object]) -> str:
    if not asr_info.get("possible_title_transcript_conflict"):
        return ""
    return (
        " Lưu ý: tiêu đề và lời thoại ASR có dấu hiệu không khớp; cần kiểm tra lại video/nhãn nguồn."
    )


def phrase_candidates(text: str, title: str) -> Counter[str]:
    original_tokens = WORD.findall(text)
    normalized_tokens = [normalize(token) for token in original_tokens]
    title_set = set(content_tokens(title))
    result: Counter[str] = Counter()
    for size in (2, 3):
        for index in range(len(normalized_tokens) - size + 1):
            normalized_phrase = normalized_tokens[index : index + size]
            if not all(normalized_phrase) or normalized_phrase[0] in STOPWORDS or normalized_phrase[-1] in STOPWORDS:
                continue
            if sum(token not in STOPWORDS for token in normalized_phrase) < 2:
                continue
            original_phrase = " ".join(original_tokens[index : index + size])
            result[original_phrase] += 1 + int(bool(set(normalized_phrase) & title_set))
    return result


def subject_terms(
    record: AsrRecord,
    title: str,
    phrase_document_frequency: Counter[str],
    document_count: int,
) -> list[str]:
    counts = phrase_candidates(record.transcript, title)
    scored: list[tuple[float, str]] = []
    title_content_tokens = set(content_tokens(title))
    for phrase, count in counts.items():
        folded = normalize(phrase)
        if is_boilerplate(phrase):
            continue
        if any(
            marker in normalize(phrase)
            for marker in (
                "video hap dan",
                "kenh ghien",
                "subscribe",
                "tin tuc nhanh",
                "tin thoi su",
                "tin tuc the gioi",
                "chuong trinh mon",
                "hap dan",
                "video",
                "kenh",
                "chuong trinh",
            )
        ):
            continue
        phrase_tokens = folded.split()
        if len(phrase_tokens) < 2 or any(len(token) < 3 for token in phrase_tokens):
            continue
        document_frequency = phrase_document_frequency[folded]
        score = (1.0 + math.log(max(count, 1))) * math.log(
            (document_count + 1) / (document_frequency + 1) + 1.0
        )
        if count < 2 and not set(content_tokens(phrase)).issubset(title_content_tokens):
            continue
        scored.append((score, phrase))
    scored.sort(key=lambda item: (-item[0], normalize(item[1])))
    result: list[str] = []
    seen: set[str] = set()
    for _, phrase in scored:
        phrase = phrase.strip(" .,:;!?-")
        folded = normalize(phrase)
        if not folded or folded in seen:
            continue
        if any(jaccard(content_tokens(phrase), content_tokens(existing)) >= 0.65 for existing in result):
            continue
        seen.add(folded)
        result.append(phrase)
        if len(result) >= 8:
            break
    return result


def generated_points(
    record: AsrRecord,
    *,
    group: str,
    title: str,
    source_description: str,
    segments: list[Segment],
    status: str,
) -> list[str]:
    source_points = clean_source_description(source_description)
    if status in {"missing", "no_speech", "boilerplate_only", "insufficient"}:
        return source_points[:3]

    if group in {"L21", "L22"}:
        headlines = news_headlines(record)
        story_openings = news_story_openings(record, segments)
        return combine_points(headlines, story_openings, limit=8)

    count_by_group = {
        "L23": 4,
        "L24": 3,
        "L25": 6,
        "L26": 5,
        "L27": 5,
        "L28": 6,
        "L29": 6,
        "L30": 5,
    }
    asr_points = select_points(
        record,
        segments,
        group=group,
        title=title,
        count=count_by_group.get(group, 5),
    )
    source_limit = 2 if group in {"L27", "L29", "L30"} else 0
    return combine_points(source_points[:source_limit], asr_points, limit=count_by_group.get(group, 5))


def transcript_title_alignment(group: str, title: str, transcript: str) -> float | None:
    if not transcript:
        return None
    title_tokens = {
        token
        for token in content_tokens(title_detail(group, title))
        if not token.isdigit() and len(token) >= 3
    }
    transcript_tokens = set(content_tokens(transcript))
    if not title_tokens:
        return None
    return round(len(title_tokens & transcript_tokens) / len(title_tokens), 3)


def topic_conflict(group: str, title: str, transcript: str, alignment: float | None) -> bool:
    if not transcript:
        return False
    folded = normalize(transcript)
    if group == "L24":
        title_tokens = set(content_tokens(title_detail(group, title)))
        transcript_tokens = set(content_tokens(transcript))
        title_specific = title_tokens - {
            "lan",
            "su",
            "rong",
            "cup",
            "cho",
            "lon",
            "htv",
            "2024",
            "doan",
            "mua",
            "trinh",
            "dien",
        }
        performance_markers = {
            "lan",
            "rong",
            "mai hoa thung",
            "dia buu",
            "nam su",
            "mua rong",
            "mua lan",
        }
        has_performance_marker = any(marker in folded for marker in performance_markers)
        specific_overlap = len(title_specific & transcript_tokens) / max(1, len(title_specific))
        return (
            len(content_tokens(transcript)) >= 80
            and not has_performance_marker
            and specific_overlap < 0.2
        )
    return bool(
        alignment is not None
        and group not in {"L21", "L22"}
        and alignment < 0.12
        and len(content_tokens(title_detail(group, title))) >= 5
    )


def enrich(
    *,
    asr_root: Path,
    media_root: Path,
    write: bool,
) -> dict[str, object]:
    records = discover_asr(asr_root)
    metadata_paths = sorted(media_root.glob("*.json"))
    metadata_ids = {path.stem for path in metadata_paths}

    # A preview directory may contain only a subset of the corpus. Restrict
    # expensive corpus statistics to the records that can actually be written.
    active_records = {
        video_id: record for video_id, record in records.items() if video_id in metadata_ids
    }

    segment_doc_frequency: Counter[str] = Counter()
    phrase_document_frequency: Counter[str] = Counter()
    for record in active_records.values():
        segment_doc_frequency.update({segment.normalized for segment in record.segments})
        phrase_document_frequency.update(
            {normalize(phrase) for phrase in phrase_candidates(record.transcript, "")}
        )

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for record in records.values():
        if record.transcript_hash:
            hash_groups[record.transcript_hash].append(record.video_id)

    stats: Counter[str] = Counter()
    changed = 0
    summaries: dict[str, list[str]] = defaultdict(list)
    conflict_ids: list[str] = []
    pending: dict[Path, dict[str, object]] = {}
    missing_metadata = sorted(set(records) - metadata_ids)

    for path in metadata_paths:
        metadata = load_json(path)
        video_id = path.stem
        group = video_id.split("_", 1)[0]
        title = str(metadata.get("title", video_id))
        source_description = str(
            metadata.get("source_description", metadata.get("description", ""))
        )
        record = records.get(
            video_id,
            AsrRecord(video_id, [], None, "", None, [], {}, None),
        )
        useful_segments = informative_segments(record, segment_doc_frequency)
        status, quality = status_for(record, useful_segments)
        points = generated_points(
            record,
            group=group,
            title=title,
            source_description=source_description,
            segments=useful_segments,
            status=status,
        )
        if not points:
            title_point = clean_text(title_detail(group, title), max_chars=260)
            if title_point:
                points = [f"Chủ đề theo tiêu đề: {title_point}"]
        description = build_summary(
            group=group,
            title=title,
            status=status,
            points=points,
        )
        topic = topic_for_with_source(group, title, source_description)
        terms = subject_terms(
            record,
            title,
            phrase_document_frequency,
            max(len(active_records), 1),
        )

        existing_keywords = [str(value) for value in metadata.get("keywords", [])]
        generated_keywords = [GROUP_LABELS.get(group, "Nội dung video"), title_detail(group, title), *terms]
        keyword_seen: set[str] = set()
        keywords: list[str] = []
        for keyword in [*existing_keywords, *generated_keywords]:
            folded = normalize(keyword)
            if not folded or folded in keyword_seen:
                continue
            keyword_seen.add(folded)
            keywords.append(keyword)

        duplicate_ids = sorted(
            item
            for item in hash_groups.get(record.transcript_hash or "", [])
            if item != video_id
        )
        title_alignment = transcript_title_alignment(group, title, record.transcript)
        title_conflict = topic_conflict(group, title, record.transcript, title_alignment)
        asr_info = {
            "status": status,
            "quality": quality,
            "language": record.summary.get("language", "vi") if record.chosen_path else None,
            "duration_sec": record.summary.get("duration_sec"),
            "word_count": record.summary.get("word_count", len(words(record.transcript))),
            "segment_count": record.summary.get("segments", len(record.segments)),
            "pipeline_version": record.summary.get("pipeline_version"),
            "transcript_sha256": record.transcript_hash,
            "source_copy_count": len(record.transcript_paths),
            "duplicate_content_ids": duplicate_ids,
            "title_token_coverage": title_alignment,
            "possible_title_transcript_conflict": title_conflict,
            "note": (
                "ASR may contain recognition errors in names, numbers, and specialized terms."
                if record.chosen_path
                else "No usable transcript was produced by the ASR stage."
            ),
        }
        if title_conflict:
            description += conflict_note(record, asr_info)
            conflict_ids.append(video_id)

        enriched = dict(metadata)
        enriched["video_id"] = video_id
        enriched["description"] = description
        enriched["source_description"] = source_description
        enriched["topic"] = topic
        enriched["key_points"] = points
        enriched["subject_terms"] = terms
        enriched["keywords"] = keywords
        enriched["asr"] = asr_info
        enriched["summary_provenance"] = {
            "method": "deterministic-title-description-asr-extractive-v1",
            "sources": [
                source
                for source, present in (
                    ("title", bool(title)),
                    ("source_description", bool(clean_source_description(source_description))),
                    ("asr", bool(record.chosen_path)),
                )
                if present
            ],
        }

        pending[path] = enriched
        stats[status] += 1
        summaries[normalize(description)].append(video_id)

    # Titles are occasionally reused for an uploaded duplicate clip. Make the
    # topic key unique while retaining the human-readable topic text.
    topic_groups: dict[str, list[str]] = defaultdict(list)
    for path, metadata in pending.items():
        topic_groups[normalize(str(metadata.get("topic", "")))].append(path.stem)
    for topic_key, ids in topic_groups.items():
        if not topic_key or len(ids) < 2:
            continue
        for video_id in sorted(ids):
            path = media_root / f"{video_id}.json"
            metadata = pending[path]
            metadata["topic"] = f"{metadata.get('topic', '')} [{video_id}]"

    for path, metadata in pending.items():
        serialized = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        if serialized != path.read_text(encoding="utf-8"):
            changed += 1
            if write:
                path.write_text(serialized, encoding="utf-8", newline="\n")

    duplicate_summaries = {
        key: value for key, value in summaries.items() if key and len(value) > 1
    }
    return {
        "metadata_files": len(metadata_paths),
        "asr_video_ids": len(records),
        "changed_files": changed,
        "written": write,
        "status_counts": dict(sorted(stats.items())),
        "asr_ids_without_metadata": missing_metadata,
        "metadata_ids_without_asr": sorted(metadata_ids - set(records)),
        "duplicate_summary_groups": len(duplicate_summaries),
        "duplicate_summary_ids": sorted(
            {video_id for ids in duplicate_summaries.values() for video_id in ids}
        ),
        "possible_title_transcript_conflict_count": len(conflict_ids),
        "possible_title_transcript_conflict_ids": sorted(conflict_ids),
        "cross_video_duplicate_asr_groups": sum(
            len(set(ids)) > 1 for ids in hash_groups.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asr-root",
        type=Path,
        default=Path(r"D:\AIC_TTVN_DATA\artifacts\kaggle"),
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path("data/source/aic2026/media-info"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write enriched JSON files. Without this flag, only report proposed changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = enrich(asr_root=args.asr_root, media_root=args.media_root, write=args.write)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
