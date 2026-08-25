from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import psycopg
from minio import Minio
from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.config import settings
from retrieval.indexes.milvus import SELF_AICV3_COLLECTION
from retrieval.storage.asr_import import import_asr
from retrieval.storage.multimodal_import import import_objects, import_ocr
from retrieval.storage.self_extracted_import import import_self_extracted

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = settings.aic_kaggle_artifact_root
BATCH = settings.aic_kaggle_batch.lower()
STAGE_ARTIFACT_FILES = {
    "03-ocr": (
        "ocr.parquet",
        "ocr.jsonl",
        "ocr-summary.json",
        "_SUCCESS.json",
    ),
    "05-object-detection": (
        "objects.parquet",
        "objects.jsonl",
        "objects.csv",
        "object-summary.json",
        "detected-classes.csv",
        "frame-timings.csv",
        "object-errors.json",
        "resolved-frames.txt",
        "_SUCCESS.json",
    ),
    "06-frame-understanding": (
        "frame-understanding.parquet",
        "frame-understanding.jsonl",
        "search-documents.parquet",
        "search-documents.jsonl",
        "shot-scenes.parquet",
        "scene-facets.json",
        "vlm-selection.parquet",
        "frame-understanding-summary.json",
        "_SUCCESS.json",
    ),
}


def video_id(zip_path: Path) -> str:
    return zip_path.stem.rsplit("-", 1)[-1]


def resolve_output(batch: str | None, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    return (ARTIFACT_ROOT / (batch or BATCH).lower()).resolve()


def store_stage_artifacts(output: Path, video_ids: set[str]) -> dict[str, object]:
    client = Minio(
        settings.minio_endpoint,
        settings.minio_root_user,
        settings.minio_root_password,
        secure=False,
    )
    if not client.bucket_exists("aic-artifacts"):
        client.make_bucket("aic-artifacts")

    stages: dict[str, dict[str, int]] = {}
    failures: list[dict[str, str]] = []
    for stage, allowed_names in STAGE_ARTIFACT_FILES.items():
        stage_dir = output / stage
        uploaded_files = 0
        uploaded_videos = 0
        if stage_dir.is_dir():
            batch_summary = stage_dir / "batch-summary.json"
            if batch_summary.is_file():
                client.fput_object(
                    "aic-artifacts",
                    f"kaggle/{output.name}/{stage}/batch-summary.json",
                    str(batch_summary),
                    content_type="application/json",
                )
                uploaded_files += 1
            for video_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
                if video_dir.name not in video_ids:
                    continue
                video_files = 0
                for name in allowed_names:
                    source = video_dir / name
                    if not source.is_file():
                        continue
                    try:
                        client.fput_object(
                            "aic-artifacts",
                            f"kaggle/{output.name}/{stage}/{video_dir.name}/{name}",
                            str(source),
                        )
                        uploaded_files += 1
                        video_files += 1
                    except Exception as exc:
                        failures.append(
                            {
                                "video_id": video_dir.name,
                                "stage": stage,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                uploaded_videos += int(video_files > 0)
        stages[stage] = {"videos": uploaded_videos, "files": uploaded_files}
    return {"stages": stages, "failed": failures}


def clear_derived() -> None:
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        for table in (
            "feature_records",
            "feature_jobs",
            "ocr_records",
            "object_detections",
            "asr_segments",
            "keyframes",
            "shots",
            "ingest_runs",
        ):
            cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
    milvus = MilvusClient(uri=settings.milvus_uri)
    if milvus.has_collection(SELF_AICV3_COLLECTION):
        milvus.drop_collection(SELF_AICV3_COLLECTION)
    client = Minio(
        settings.minio_endpoint,
        settings.minio_root_user,
        settings.minio_root_password,
        secure=False,
    )
    for bucket in ("aic-frames", "aic-artifacts"):
        if client.bucket_exists(bucket):
            for obj in client.list_objects(bucket, recursive=True):
                client.remove_object(bucket, obj.object_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild PostgreSQL, MinIO and Milvus from a Kaggle artifact batch."
    )
    parser.add_argument(
        "--confirm-rebuild",
        action="store_true",
        help="Confirm deletion of the current derived index before importing the batch.",
    )
    parser.add_argument(
        "--batch",
        help=f"Artifact batch under {ARTIFACT_ROOT}. Defaults to AIC_KAGGLE_BATCH={BATCH}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Use an explicit artifact batch directory instead of AIC_KAGGLE_ARTIFACT_ROOT/batch.",
    )
    parser.add_argument(
        "--skip-stage-artifacts",
        action="store_true",
        help="Do not upload OCR, object-detection or frame-understanding files to MinIO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_output(args.batch, args.output_root)
    shot_dir = output / "01-shot-keyframes"
    visual_dir = output / "02-visual-embeddings"
    asr_dir = output / "04-asr"
    ocr_dir = output / "03-ocr"
    object_dir = output / "05-object-detection"
    shots = {video_id(p): p for p in shot_dir.glob("*.zip")}
    visuals = {video_id(p): p for p in visual_dir.glob("*.zip")}
    asrs = {video_id(p): p for p in asr_dir.glob("*.zip")}
    ocrs = {p.name: p for p in ocr_dir.iterdir() if p.is_dir()} if ocr_dir.is_dir() else {}
    objects = (
        {p.name: p for p in object_dir.iterdir() if p.is_dir()}
        if object_dir.is_dir()
        else {}
    )

    if not shots:
        raise SystemExit(f"No shot/keyframe ZIP files found in {shot_dir}")
    missing_visuals = sorted(set(shots) - set(visuals))
    if missing_visuals:
        raise SystemExit(
            "Missing visual embedding ZIP files for: " + ", ".join(missing_visuals)
        )
    if not args.confirm_rebuild:
        raise SystemExit(
            "Refusing to delete the current derived index. "
            "Review the selected batch and rerun with --confirm-rebuild.\n"
            f"batch={output.name} root={output} videos={len(shots)} asr_files={len(asrs)}"
        )

    clear_derived()
    results = {
        "batch": output.name,
        "root": str(output),
        "visual": [],
        "asr": [],
        "ocr": [],
        "object": [],
        "failed": [],
    }
    with tempfile.TemporaryDirectory(prefix="aic-import-") as td:
        temp = Path(td)
        for vid, shot_zip in sorted(shots.items()):
            print(f"visual {vid}...", flush=True)
            if vid not in visuals:
                results["failed"].append({"video_id": vid, "stage": "visual", "error": "missing visual ZIP"})
                continue
            root = temp / vid
            root.mkdir()
            with zipfile.ZipFile(shot_zip) as z:
                z.extractall(root)
            with zipfile.ZipFile(visuals[vid]) as z:
                for name in ("embedding-records.parquet", "visual.npy", "visual-summary.json", "_SUCCESS.json"):
                    z.extract(name, root)
            # The batch visual export stores model metadata separately from the
            # shot summary; enrich the combined artifact expected by importer.
            shot_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            visual_summary = json.loads((root / "visual-summary.json").read_text(encoding="utf-8"))
            shot_summary["embedding_model"] = visual_summary.get("model", "ViT-B-32/laion2b_s34b_b79k")
            (root / "summary.json").write_text(json.dumps(shot_summary, ensure_ascii=False), encoding="utf-8")
            try:
                results["visual"].append(import_self_extracted(root, SELF_AICV3_COLLECTION))
            except Exception as exc:
                results["failed"].append({"video_id": vid, "stage": "visual", "error": f"{type(exc).__name__}: {exc}"})
        for vid, asr_zip in sorted(asrs.items()):
            print(f"asr {vid}...", flush=True)
            root = temp / f"asr-{vid}"
            root.mkdir()
            with zipfile.ZipFile(asr_zip) as z:
                z.extract("asr-segments.parquet", root)
            try:
                results["asr"].append({"video_id": vid, **import_asr(root)})
            except Exception as exc:
                results["failed"].append({"video_id": vid, "stage": "asr", "error": f"{type(exc).__name__}: {exc}"})
        for vid, root in sorted(ocrs.items()):
            print(f"ocr {vid}...", flush=True)
            try:
                results["ocr"].append(import_ocr(root))
            except Exception as exc:
                results["failed"].append(
                    {"video_id": vid, "stage": "ocr", "error": f"{type(exc).__name__}: {exc}"}
                )
        for vid, root in sorted(objects.items()):
            print(f"object {vid}...", flush=True)
            try:
                results["object"].append(import_objects(root))
            except Exception as exc:
                results["failed"].append(
                    {
                        "video_id": vid,
                        "stage": "object",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if not args.skip_stage_artifacts:
        print("stage artifacts...", flush=True)
        stage_result = store_stage_artifacts(output, set(shots))
        results["stage_artifacts"] = stage_result["stages"]
        results["failed"].extend(stage_result["failed"])
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
