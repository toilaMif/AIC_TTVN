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
from retrieval.storage.self_extracted_import import import_self_extracted

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = settings.aic_kaggle_artifact_root
BATCH = settings.aic_kaggle_batch.lower()
OUTPUT = ARTIFACT_ROOT / BATCH


def video_id(zip_path: Path) -> str:
    return zip_path.stem.rsplit("-", 1)[-1]


def clear_derived() -> None:
    url = settings.database_url.replace("+psycopg", "")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        for table in (
            "feature_records",
            "feature_jobs",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shot_dir = OUTPUT / "01-shot-keyframes"
    visual_dir = OUTPUT / "02-visual-embeddings"
    asr_dir = OUTPUT / "04-asr"
    shots = {video_id(p): p for p in shot_dir.glob("*.zip")}
    visuals = {video_id(p): p for p in visual_dir.glob("*.zip")}
    asrs = {video_id(p): p for p in asr_dir.glob("*.zip")}

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
            f"batch={BATCH} root={OUTPUT} videos={len(shots)} asr_files={len(asrs)}"
        )

    clear_derived()
    results = {"visual": [], "asr": [], "failed": []}
    with tempfile.TemporaryDirectory(prefix="aic-import-") as td:
        temp = Path(td)
        for vid, shot_zip in sorted(shots.items()):
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
            root = temp / f"asr-{vid}"
            root.mkdir()
            with zipfile.ZipFile(asr_zip) as z:
                z.extract("asr-segments.parquet", root)
            try:
                results["asr"].append({"video_id": vid, **import_asr(root)})
            except Exception as exc:
                results["failed"].append({"video_id": vid, "stage": "asr", "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
