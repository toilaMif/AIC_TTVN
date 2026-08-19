from pathlib import Path

import typer

from retrieval.indexes.milvus import (
    DEFAULT_COLLECTION,
    SELF_AICV3_COLLECTION,
    create_visual_collection,
    search_visual_vector,
)
from retrieval.ingest.audit import audit as run_audit
from retrieval.ingest.audit import write_report
from retrieval.ingest.videos import download_videos as run_download_videos
from retrieval.ingest.videos import select_demo as run_select_demo
from retrieval.search.global_visual import search_text_diversified
from retrieval.storage.asr_import import import_asr as run_import_asr
from retrieval.storage.metadata_import import import_metadata as run_import_metadata
from retrieval.storage.self_extracted_import import import_self_extracted as run_import_self
from retrieval.storage.self_extracted_import import validate_output

app = typer.Typer(help="Audit and prepare AIC datasets.")


@app.callback()
def main() -> None:
    """Manage AIC dataset preparation commands."""


@app.command()
def audit(
    mapping_root: Path = Path("data/source/aic2026/map-keyframes"),
    media_info_root: Path = Path("data/source/aic2026/media-info"),
    output_dir: Path = Path("data/manifests"),
) -> None:
    """Validate BTC metadata/mapping pairs and write reproducible manifests."""
    manifests, issues = run_audit(mapping_root, media_info_root)
    write_report(manifests, issues, output_dir)
    errors = sum(issue.severity == "error" for issue in issues)
    typer.echo(
        f"videos={len(manifests)} keyframes={sum(int(item['keyframe_count']) for item in manifests)} errors={errors}"
    )
    typer.echo(f"manifest={output_dir / 'dataset.jsonl'}")
    if errors:
        raise typer.Exit(code=1)


@app.command("import-metadata")
def import_metadata(manifest_path: Path = Path("data/manifests/dataset.jsonl")) -> None:
    """Import audited video and keyframe metadata into PostgreSQL."""
    videos, keyframes, run_id = run_import_metadata(manifest_path)
    typer.echo(f"run_id={run_id} videos={videos} keyframes={keyframes}")


@app.command("select-demo")
def select_demo(
    source: Path = Path("data/manifests/dataset.jsonl"),
    output: Path = Path("data/manifests/demo-10.jsonl"),
    limit: int = 10,
) -> None:
    """Freeze the first valid videos into a reproducible demo manifest."""
    selected = run_select_demo(source, output, limit)
    typer.echo(f"manifest={output} videos={len(selected)}")
    for item in selected:
        typer.echo(f"  {item['video_id']} {item['watch_url']}")


@app.command("download-videos")
def download_videos(
    manifest: Path = Path("data/manifests/demo-10.jsonl"),
    output_root: Path = Path(".runtime/demo-10/videos"),
    max_height: int = 720,
) -> None:
    """Download demo videos and persist checksums for idempotent reruns."""
    results = run_download_videos(manifest, output_root, max_height)
    for item in results:
        typer.echo(
            f"{item['video_id']} status={item['status']} "
            f"height={item.get('height')} bytes={item['bytes']}"
        )


@app.command("create-milvus-collection")
def create_milvus_collection(
    collection_name: str = DEFAULT_COLLECTION,
    dimension: int = 512,
) -> None:
    """Create the demo visual collection using FLAT/IP."""
    created = create_visual_collection(
        uri="http://127.0.0.1:19530",
        collection_name=collection_name,
        dimension=dimension,
    )
    typer.echo(f"collection={collection_name} created={created} dimension={dimension}")


@app.command("import-self-extracted")
def import_self_extracted(
    output_root: Path,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> None:
    """Import a validated Kaggle V3 output into MinIO, PostgreSQL and Milvus."""
    result = run_import_self(output_root, collection_name)
    typer.echo(" ".join(f"{key}={value}" for key, value in result.items()))


@app.command("import-asr")
def import_asr(output_root: Path, model_name: str = "faster-whisper/large-v3") -> None:
    """Import ASR segment parquet into PostgreSQL."""
    result = run_import_asr(output_root, model_name)
    typer.echo(" ".join(f"{key}={value}" for key, value in result.items()))


@app.command("verify-self-extracted")
def verify_self_extracted(
    output_root: Path,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> None:
    """Validate output and verify Milvus by querying the first image vector."""
    _, _, keyframes, vectors = validate_output(output_root)
    results = search_visual_vector(
        uri="http://127.0.0.1:19530",
        collection_name=collection_name,
        vector=vectors[0].tolist(),
        limit=5,
    )
    expected = keyframes.iloc[0]["keyframe_id"]
    first = results[0]["entity"]["keyframe_id"] if results else None
    typer.echo(f"expected={expected} top1={first} results={len(results)}")
    if first != expected:
        raise typer.Exit(code=1)


@app.command("search-text")
def search_text(
    query: str,
    top_k: int = 10,
    candidate_k: int = 200,
    max_results_per_shot: int = 1,
    min_temporal_gap_seconds: float = 3.0,
    collection_name: str = SELF_AICV3_COLLECTION,
) -> None:
    """Search keyframes and diversify results across shots."""
    for rank, result in enumerate(
        search_text_diversified(
            query,
            top_k,
            candidate_k,
            max_results_per_shot,
            min_temporal_gap_seconds,
            collection_name,
        ),
        start=1,
    ):
        typer.echo(
            f"{rank}. score={result.score:.5f} video_id={result.video_id} "
            f"frame_idx={result.frame_idx} pts_time={result.pts_time} "
            f"shot_id={result.shot_id} keyframe_id={result.keyframe_id} url={result.frame_url}"
        )


if __name__ == "__main__":
    app()
