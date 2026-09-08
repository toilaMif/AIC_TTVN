"""Stage 07: reproducible retrieval quality and latency benchmark.

The benchmark runs against the portable SQLite snapshot emitted by stage 06.
Ground-truth labels are optional JSONL. A line can use direct fields such as
``relevant_doc_ids``, ``relevant_shot_ids`` and ``relevant_video_ids`` or a
compact qrels form such as ``{"relevant": [{"doc_id": "...", "relevance": 2}]}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from v2_runtime import (  # noqa: E402
    PIPELINE_VERSION,
    atomic_write_json,
    config_hash,
    search_hybrid_snapshot,
    tokenize,
    write_success,
)

STAGE_VERSION = f"{PIPELINE_VERSION}-eval"
LEVELS = ("doc_id", "shot_id", "video_id")
DEFAULT_KS = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "07-index-pack",
    )
    parser.add_argument(
        "--query-zip",
        type=Path,
        default=Path("data/source/aic2026/Query/query-p1-groupA.zip"),
    )
    parser.add_argument("--labels", type=Path, default=None, help="Optional qrels JSONL file")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.getenv("AIC_V2_OUTPUT_ROOT", "/kaggle/working/aic-v2")) / "08-eval",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--ks",
        default=",".join(str(value) for value in DEFAULT_KS),
        help="Comma-separated quality cutoffs (default: 1,5,10)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Untimed executions per query")
    parser.add_argument("--repeat", type=int, default=3, help="Timed executions per query")
    return parser.parse_args()


def _parse_ks(value: str) -> tuple[int, ...]:
    values: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid quality cutoff: {token!r}") from exc
        if parsed < 1:
            raise ValueError("quality cutoffs must be positive")
        values.add(parsed)
    if not values:
        raise ValueError("at least one quality cutoff is required")
    return tuple(sorted(values))


def read_queries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".txt"):
                continue
            query_id = Path(name).stem
            rows.append({
                "query_id": query_id,
                "query_type": query_id.rsplit("-", 1)[-1].casefold(),
                "text": archive.read(name).decode("utf-8", errors="replace").strip(),
            })
    return rows


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _finite_grade(value: Any, default: float = 1.0) -> float:
    try:
        grade = float(value)
    except (TypeError, ValueError):
        return default
    return grade if math.isfinite(grade) and grade > 0 else default


def _add_grade(target: dict[str, float], identifier: Any, grade: Any = 1.0) -> None:
    if identifier is None or isinstance(identifier, (dict, list, tuple, set)):
        return
    value = str(identifier).strip()
    if value:
        target[value] = max(target.get(value, 0.0), _finite_grade(grade))


def _level_from_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "doc": "doc_id", "document": "doc_id", "documents": "doc_id", "docid": "doc_id",
        "shot": "shot_id", "shots": "shot_id", "shotid": "shot_id",
        "video": "video_id", "videos": "video_id", "videoid": "video_id",
    }
    return aliases.get(text, text if text in LEVELS else None)


def _extract_items(value: Any, level: str) -> Iterable[tuple[Any, float]]:
    if value is None:
        return
    if isinstance(value, Mapping):
        explicit_id = value.get(level) or value.get("id") or value.get("value")
        if explicit_id is not None:
            yield explicit_id, value.get("relevance", value.get("relevance_score", value.get("score", 1)))
            return
        for identifier, grade in value.items():
            yield identifier, grade
        return
    for item in _as_list(value):
        if isinstance(item, Mapping):
            identifier = item.get(level) or item.get("id") or item.get("value")
            if identifier is not None:
                yield identifier, item.get("relevance", item.get("relevance_score", item.get("score", 1)))
        else:
            yield item, 1.0


def _collect_level_values(container: Mapping[str, Any], level: str, target: dict[str, float]) -> None:
    aliases = (
        level,
        f"relevant_{level}s", f"relevant_{level}",
        f"gold_{level}s", f"gold_{level}",
        f"positive_{level}s", f"positive_{level}",
        f"{level}s",
    )
    for key in aliases:
        if key in container:
            for identifier, grade in _extract_items(container[key], level):
                _add_grade(target, identifier, grade)


def _normalise_label_line(raw: Mapping[str, Any], line_number: int) -> tuple[str, dict[str, dict[str, float]]]:
    query_id = raw.get("query_id", raw.get("qid", raw.get("query", raw.get("id"))))
    if query_id is None and len(raw) == 1:
        only_key, only_value = next(iter(raw.items()))
        if isinstance(only_value, Mapping):
            query_id, raw = only_key, only_value
    if query_id is None or not str(query_id).strip():
        raise ValueError(f"labels line {line_number}: missing query_id")

    result: dict[str, dict[str, float]] = {level: {} for level in LEVELS}
    for level in LEVELS:
        _collect_level_values(raw, level, result[level])
    for key in ("relevant", "relevance", "qrels", "ground_truth", "positives", "relevant_items"):
        nested = raw.get(key)
        if nested is None:
            continue
        if isinstance(nested, Mapping):
            for level in LEVELS:
                _collect_level_values(nested, level, result[level])
            if not any(level in nested or f"relevant_{level}s" in nested for level in LEVELS):
                for identifier, grade in _extract_items(nested, "doc_id"):
                    _add_grade(result["doc_id"], identifier, grade)
        else:
            for item in _as_list(nested):
                if isinstance(item, Mapping):
                    level = _level_from_value(item.get("level", item.get("type")))
                    if level:
                        identifier = item.get(level, item.get("id", item.get("value")))
                        _add_grade(result[level], identifier, item.get("relevance", item.get("score", 1)))
                    else:
                        for level in LEVELS:
                            if level in item:
                                _add_grade(result[level], item[level], item.get("relevance", item.get("score", 1)))
                else:
                    _add_grade(result["doc_id"], item)
    for key in ("relevant_ids", "positive_ids", "gold_ids"):
        if key in raw:
            for identifier, grade in _extract_items(raw[key], "doc_id"):
                _add_grade(result["doc_id"], identifier, grade)
    return str(query_id).strip(), {level: values for level, values in result.items() if values}


def read_labels(path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    """Read optional JSONL qrels into query_id -> level -> id -> grade."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels line {line_number}: invalid JSON") from exc
            if not isinstance(decoded, Mapping):
                raise ValueError(f"labels line {line_number}: expected an object")
            query_id, values = _normalise_label_line(decoded, line_number)
            existing = labels.setdefault(query_id, {level: {} for level in LEVELS})
            for level, items in values.items():
                for identifier, grade in items.items():
                    existing[level][identifier] = max(existing[level].get(identifier, 0.0), grade)
    return labels


def _legacy_fts_rows(connection: sqlite3.Connection, query: str, limit: int) -> list[dict[str, Any]]:
    terms = [token for token in tokenize(query) if len(token) > 1]
    if not terms:
        return []
    expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
    try:
        rows = connection.execute(
            "SELECT doc_id, bm25(documents_fts) AS score FROM documents_fts WHERE documents_fts MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"doc_id": str(row[0]), "bm25_score": float(row[1] or 0.0)} for row in rows]


def fts_search(connection: sqlite3.Connection, query: str, limit: int) -> list[str]:
    """Compatibility helper retained for callers of the original benchmark."""
    return [str(row["doc_id"]) for row in _legacy_fts_rows(connection, query, limit)]


def _search_rows(index_root: Path, query: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    try:
        rows = search_hybrid_snapshot(
            index_root / "text-index.sqlite",
            query,
            limit=limit,
            candidate_limit=max(limit * 5, limit),
            levels=("event", "shot", "frame"),
        )
        return rows, "hybrid_snapshot"
    except (sqlite3.Error, OSError, KeyError, ValueError):
        with sqlite3.connect(index_root / "text-index.sqlite") as connection:
            return _legacy_fts_rows(connection, query, limit), "fts5_legacy"


def _unique_identity(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return values


def _metric_values(ranking: Sequence[str], truth: Mapping[str, float], ks: Sequence[int]) -> dict[str, float | int | None]:
    relevant = {str(identifier): _finite_grade(grade) for identifier, grade in truth.items()}
    if not relevant:
        return {
            "relevant": 0, "mrr": None,
            **{f"recall@{k}": None for k in ks},
            **{f"ndcg@{k}": None for k in ks},
        }
    first_rank = next((rank for rank, identifier in enumerate(ranking, start=1) if identifier in relevant), None)
    metrics: dict[str, float | int | None] = {
        "relevant": len(relevant),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }
    ideal_grades = sorted(relevant.values(), reverse=True)
    for k in ks:
        top = list(ranking[:k])
        metrics[f"recall@{k}"] = sum(identifier in relevant for identifier in top) / len(relevant)
        dcg = sum(
            (2.0 ** relevant[identifier] - 1.0) / math.log2(rank + 1)
            for rank, identifier in enumerate(top, start=1)
            if identifier in relevant
        )
        ideal = sum(
            (2.0 ** grade - 1.0) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades[:k], start=1)
        )
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0
    return metrics


def _aggregate(metric_rows: Sequence[Mapping[str, float | int | None]], ks: Sequence[int]) -> dict[str, Any]:
    if not metric_rows:
        return {
            "queries_labeled": 0, "relevant": 0, "mrr": None,
            **{f"recall@{k}": None for k in ks},
            **{f"ndcg@{k}": None for k in ks},
        }
    output: dict[str, Any] = {
        "queries_labeled": len(metric_rows),
        "relevant": int(sum(int(row.get("relevant", 0) or 0) for row in metric_rows)),
    }
    for name in ("mrr", *[f"recall@{k}" for k in ks], *[f"ndcg@{k}" for k in ks]):
        values = [float(row[name]) for row in metric_rows if row.get(name) is not None]
        output[name] = float(np.mean(values)) if values else None
    return output


def _quality_report(
    query_rows: Sequence[Mapping[str, Any]],
    per_query: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    ks: Sequence[int],
) -> dict[str, Any]:
    by_level = {
        level: _aggregate([
            per_query[query["query_id"]][level]
            for query in query_rows
            if query["query_id"] in per_query and level in per_query[query["query_id"]]
        ], ks)
        for level in LEVELS
    }
    by_type: dict[str, dict[str, Any]] = {}
    for query in query_rows:
        query_id = str(query["query_id"])
        query_type = str(query.get("query_type", "unknown"))
        bucket = by_type.setdefault(query_type, {"queries": 0, "levels": {level: [] for level in LEVELS}})
        bucket["queries"] += 1
        for level in LEVELS:
            if query_id in per_query and level in per_query[query_id]:
                bucket["levels"][level].append(per_query[query_id][level])
    for bucket in by_type.values():
        bucket["levels"] = {level: _aggregate(values, ks) for level, values in bucket["levels"].items()}
    query_ids = {str(query["query_id"]) for query in query_rows}
    return {
        "labels_queries": len(labels),
        "matched_queries": sum(query_id in labels for query_id in query_ids),
        "unmatched_label_queries": sorted(set(labels) - query_ids),
        "by_level": by_level,
        "by_query_type": by_type,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype="float64")
    if not len(array):
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "mean": None, "min": None, "max": None}
    return {
        "samples": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _index_size(index_root: Path) -> dict[str, Any]:
    files = []
    total = 0
    if index_root.exists():
        paths = sorted((item for item in index_root.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold())
        for path in paths:
            try:
                size = int(path.stat().st_size)
            except OSError:
                continue
            total += size
            files.append({"name": path.relative_to(index_root).as_posix(), "bytes": size})
    return {"bytes": total, "mb": total / (1024.0 * 1024.0), "files": len(files), "by_file": files}


def main() -> None:
    options = parse_args()
    if options.top_k < 1:
        raise ValueError("top_k must be positive")
    if options.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if options.repeat < 1:
        raise ValueError("repeat must be positive")
    ks = _parse_ks(options.ks)
    options.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = options.index_root / "index-manifest.json"
    sqlite_path = options.index_root / "text-index.sqlite"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing index manifest: {manifest_path}")
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Missing text index: {sqlite_path}")
    index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = read_queries(options.query_zip)
    labels = read_labels(options.labels)
    labels_provided = options.labels is not None

    rows: list[dict[str, Any]] = []
    measured_latencies: list[float] = []
    latencies_by_type: dict[str, list[float]] = {}
    per_query_metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    retrieval_modes: set[str] = set()
    for query in queries:
        for _ in range(options.warmup):
            _search_rows(options.index_root, query["text"], options.top_k)
        samples: list[float] = []
        result: list[dict[str, Any]] = []
        mode = "unknown"
        for _ in range(options.repeat):
            started = time.perf_counter_ns()
            result, mode = _search_rows(options.index_root, query["text"], options.top_k)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            samples.append(elapsed_ms)
            measured_latencies.append(elapsed_ms)
            latencies_by_type.setdefault(query["query_type"], []).append(elapsed_ms)
            retrieval_modes.add(mode)
        rankings = {
            "doc_id": _unique_identity(result, "doc_id"),
            "shot_id": _unique_identity(result, "shot_id"),
            "video_id": _unique_identity(result, "video_id"),
        }
        query_metrics: dict[str, dict[str, float | int | None]] = {}
        truth = labels.get(query["query_id"], {})
        for level, ranking in rankings.items():
            if level in truth:
                query_metrics[level] = _metric_values(ranking, truth[level], ks)
        if query_metrics:
            per_query_metrics[query["query_id"]] = query_metrics
        latency = _latency_summary(samples)
        rows.append({
            **query,
            "top_doc_ids_json": json.dumps(rankings["doc_id"], ensure_ascii=False),
            "top_shot_ids_json": json.dumps(rankings["shot_id"], ensure_ascii=False),
            "top_video_ids_json": json.dumps(rankings["video_id"], ensure_ascii=False),
            "latency_ms": latency["p50"],
            "latency_p95_ms": latency["p95"],
            "latency_p99_ms": latency["p99"],
            "latency_samples_ms_json": json.dumps(samples),
            "hit_count": len(result),
            "retrieval_mode": mode,
            "metrics_json": json.dumps(query_metrics, ensure_ascii=False) if query_metrics else None,
        })

    frame = pd.DataFrame(rows)
    frame.to_json(options.output_root / "query-results.jsonl", orient="records", lines=True, force_ascii=False)
    quality = _quality_report(rows, per_query_metrics, labels, ks)
    index_size = _index_size(options.index_root)
    latency = _latency_summary(measured_latencies)
    modes = sorted(retrieval_modes)
    report_config = {
        "stage": "eval", "top_k": options.top_k, "ks": list(ks),
        "query_zip": str(options.query_zip), "labels": str(options.labels) if options.labels else None,
        "warmup": options.warmup, "repeat": options.repeat,
    }
    report: dict[str, Any] = {
        "schema_version": "aic-video-v2/1",
        "stage": "eval",
        "pipeline_version": STAGE_VERSION,
        "config_hash": config_hash(report_config),
        "index_pipeline_version": index_manifest.get("pipeline_version"),
        "queries": len(queries),
        "top_k": options.top_k,
        "quality_cutoffs": list(ks),
        "retrieval_mode": modes[0] if len(modes) == 1 else modes,
        "latency_ms": latency,
        "latency_by_query_type_ms": {
            query_type: _latency_summary(values)
            for query_type, values in sorted(latencies_by_type.items())
        },
        "latency_config": {"warmup": options.warmup, "repeat": options.repeat, "timed_queries": len(queries)},
        "index_size": index_size,
        "index_size_bytes": index_size["bytes"],
        "index_size_mb": index_size["mb"],
        "labels": {"provided": options.labels is not None, "path": str(options.labels) if options.labels else None, "queries": len(labels)},
        "metrics": quality["by_level"] if labels_provided else None,
        "quality": quality if labels_provided else None,
        "metrics_pending_ground_truth": [] if labels_provided else [
            *[f"recall@{k}" for k in ks],
            "MRR",
            *[f"nDCG@{k}" for k in ks],
            "temporal_recall",
            "TRAKE_order_accuracy",
        ],
        "notes": (
            "Quality metrics are macro-averaged over labeled queries; shot/video metrics use IDs in ranked documents."
            if labels_provided
            else "KIS/QA/TRAKE ground truth is not inferred from query text; provide a labels JSONL to compute recall safely."
        ),
    }
    atomic_write_json(options.output_root / "eval-report.json", report)
    write_success(options.output_root, {
        "stage": "eval", "pipeline_version": STAGE_VERSION,
        "config_hash": report["config_hash"], "queries": len(queries), "labels": labels_provided,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
