"""Generate the small, reproducible V2 notebook wrappers.

The implementation lives in `scripts/*.py` so it can be tested and resumed from
the command line. Notebook cells only install the profile dependencies, locate
the uploaded V2 directory, and run the same script; this avoids copy/paste drift
between Kaggle and local runs.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGES = [
    {
        "directory": "00-manifest-normalize",
        "filename": "00-manifest-normalize-v2.ipynb",
        "title": "00 - Manifest, timeline and cached audio (V2)",
        "script": "00_manifest_normalize.py",
        "install": "%pip install -q av opencv-python-headless pandas pyarrow\n!command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)",
        "args": "[\"--max-videos\", \"2\"]",
        "full_args": "[]",
        "outputs": "`00-manifest/video-manifest.parquet`, cached 16 kHz mono WAV, and `_SUCCESS.json`.",
    },
    {
        "directory": "01-shot-keyframe-visual",
        "filename": "01-shot-keyframe-visual-v2.ipynb",
        "title": "01 - Adaptive shots + keyframes + visual vectors (V2)",
        "script": "01_shot_keyframe_visual.py",
        "install": (
            "%pip install -q av opencv-python-headless open_clip_torch pandas pyarrow pillow ffmpeg-python\n"
            "!test -d /kaggle/working/TransNetV2 || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "
            "https://github.com/soCzech/TransNetV2.git /kaggle/working/TransNetV2\n"
            "# The upstream repo's Git LFS bandwidth budget is exhausted (confirmed on a real Kaggle\n"
            "# run), so inference/transnetv2-weights/* never comes down via plain `git clone`. Attach\n"
            "# the \"aic-ttvn-transnetv2-weights\" Kaggle dataset (same files, mirrored) and copy them in.\n"
            "!mkdir -p /kaggle/working/TransNetV2/inference/transnetv2-weights/variables\n"
            "!test -f /kaggle/working/TransNetV2/inference/transnetv2-weights/saved_model.pb || "
            "cp /kaggle/input/aic-ttvn-transnetv2-weights/saved_model.pb "
            "/kaggle/working/TransNetV2/inference/transnetv2-weights/saved_model.pb\n"
            "!test -f /kaggle/working/TransNetV2/inference/transnetv2-weights/variables/variables.index || "
            "cp /kaggle/input/aic-ttvn-transnetv2-weights/variables/variables.* "
            "/kaggle/working/TransNetV2/inference/transnetv2-weights/variables/"
        ),
        "args": "[\"--max-videos\", \"1\", \"--no-make-compat-zips\"]",
        "full_args": "[]",
        "outputs": "Legacy-compatible `shots.parquet`, `keyframes.parquet`, `visual.npy`, embedding records, shot vectors, WebP frames, and validation markers. OpenCLIP is encoded once.",
    },
    {
        "directory": "02-ocr-batched",
        "filename": "02-ocr-batched-v2.ipynb",
        "title": "02 - Batched OCR with resumable shards (V2)",
        "script": "02_ocr_batched.py",
        "install": "%pip install -q pandas pyarrow pillow opencv-python-headless\n# Install a PaddleOCR GPU wheel matching the Kaggle CUDA image before this cell.",
        "args": "[\"--max-videos\", \"1\", \"--max-frames\", \"32\"]",
        "full_args": "[]",
        "outputs": "`ocr.parquet/jsonl`, scene/ticker fields, append-only checkpoint, and `_SUCCESS.json` per video.",
    },
    {
        "directory": "03-asr-fast",
        "filename": "03-asr-fast-v2.ipynb",
        "title": "03 - Cached-audio Vietnamese ASR (V2)",
        "script": "03_asr_fast.py",
        "install": "%pip install -q faster-whisper==1.1.0 \"ctranslate2>=4.5.0,<5\" pandas pyarrow\n!command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)\n# ctranslate2==4.4.0 only links against cuDNN8 (libcudnn_ops_infer.so.8); Kaggle's\n# PyTorch image ships cuDNN9, which removed that file. 4.5.0+ added cuDNN9 support.",
        "args": "[\"--max-videos\", \"1\"]",
        "full_args": "[\"--make-compat-zips\"]",
        "outputs": "`asr-segments.parquet`, transcript, cached audio, and segment-level resume markers.",
    },
    {
        "directory": "04-objects-batched",
        "filename": "04-objects-batched-v2.ipynb",
        "title": "04 - Batched fixed-vocabulary object detection (V2)",
        "script": "04_objects_batched.py",
        "install": "%pip install -q ultralytics==8.4.123 pandas pyarrow pillow\n# The model is loaded once and reused for every video.",
        "args": "[\"--max-videos\", \"1\", \"--max-frames\", \"32\"]",
        "full_args": "[]",
        "outputs": "`objects.parquet/jsonl`, normalized boxes, per-frame counts (including zero-detection frames), and checkpoint markers.",
    },
    {
        "directory": "05-context-fusion",
        "filename": "05-context-fusion-v2.ipynb",
        "title": "05 - Timestamp-aware context fusion and temporal events (V2)",
        "script": "05_context_fusion.py",
        "install": "%pip install -q pandas pyarrow pillow\n# The full run below enables Qwen2.5-VL (OptionalVLM); the smoke-test sample list\n# disables it, so this install only matters for the full run. Bump transformers\n# if Qwen2_5_VLForConditionalGeneration fails to import.\n%pip install -q \"transformers>=4.49\" accelerate",
        "args": "[\"--max-videos\", \"1\", \"--max-shots\", \"20\", \"--no-enable-vlm\"]",
        "full_args": "[\"--enable-vlm\"]",
        "outputs": "Frame/shot/event docs, ASR-OCR-object joins, scene facets, `temporal-edges.parquet`, and provenance fields.",
    },
    {
        "directory": "06-index-pack",
        "filename": "06-index-pack-v2.ipynb",
        "title": "06 - BM25/FTS5 + ANN-ready retrieval index pack (V2)",
        "script": "06_index_pack.py",
        "install": "%pip install -q pandas pyarrow numpy\n!pip install -q faiss-cpu || true\n# Milvus loading remains optional; this stage also builds a portable FAISS shot index when available.",
        "args": "[]",
        "full_args": "[\"--milvus-index\", \"HNSW\"]",
        "outputs": "Compact frame/shot/event documents, weighted sparse postings, FTS5 snapshot, ANN configuration, vectors, and checksums.",
    },
    {
        "directory": "07-eval-benchmark",
        "filename": "07-eval-benchmark-v2.ipynb",
        "title": "07 - Retrieval latency and quality benchmark (V2)",
        "script": "07_eval_benchmark.py",
        "install": "%pip install -q pandas numpy pyarrow",
        "args": "[]",
        "full_args": "[]",
        "outputs": "Per-query rankings plus p50/p95/p99 latency, index size, and Recall@K/MRR/nDCG by doc, shot, video, and query type when qrels JSONL is supplied.",
    },
    {
        "directory": "08-hybrid-query",
        "filename": "08-hybrid-query-v2.ipynb",
        "title": "08 - Low-latency hybrid query with RRF (V2)",
        "script": "08_hybrid_query.py",
        "install": "%pip install -q pandas pyarrow numpy open_clip_torch\n!pip install -q faiss-cpu || true\n# FAISS is optional; the script falls back to a NumPy memmap search.",
        "args": "[\"người ở chợ\", \"--top-k\", \"5\", \"--no-visual\"]",
        "full_args": "[\"--query-file\", \"/kaggle/working/query.txt\", \"--top-k\", \"20\", \"--visual\"]",
        "outputs": "JSON results with level-aware FTS, metadata filters, optional FAISS/NumPy visual retrieval, weighted RRF, shot diversity, timestamps, and latency breakdown.",
        # full_args points at a query file the caller must create first; running
        # this stage isn't required to produce saved features, so keep it sample-only.
        "keep_smoke_default": True,
    },
]


def markdown(stage: dict) -> str:
    if stage.get("keep_smoke_default"):
        run_note = "`RUN_ARGS` below defaults to a 1-item smoke test. Set it to the full-run list after checking a sample."
    else:
        run_note = "`RUN_ARGS` below defaults to a **full run** (no `--max-videos`/`--max-frames` cap). Swap in the smaller sample list first if you want to sanity-check on 1 video before spending full GPU time."
    return f"""# {stage['title']}\n\n{stage['outputs']}\n\n**Execution model:** the notebook is a thin wrapper around the versioned script in `notebooks_V2/scripts/`. The cell is safe to rerun; completed videos are skipped by `_SUCCESS.json`. {run_note}\n"""


def bootstrap_cell(stage: dict) -> str:
    default_args = stage["args"] if stage.get("keep_smoke_default") else stage["full_args"]
    return f'''from __future__ import annotations
import json
import runpy
import sys
from pathlib import Path

SCRIPT_NAME = {stage["script"]!r}
RUN_ARGS = {default_args}

def locate_v2_root() -> Path:
    candidates = [
        Path.cwd() / "pipelines" / "kaggle" / "notebooks_V2",
        Path.cwd() / "notebooks_V2",
        Path("/kaggle/working/notebooks_V2"),
    ]
    candidates.extend(path.parent.parent for path in Path("/kaggle/input").rglob("v2_runtime.py"))
    for candidate in candidates:
        if (candidate / "scripts" / SCRIPT_NAME).exists():
            return candidate
    raise FileNotFoundError("Attach the complete notebooks_V2 directory or run from the repository root")

V2_ROOT = locate_v2_root()
SCRIPT = V2_ROOT / "scripts" / SCRIPT_NAME
sys.path.insert(0, str(V2_ROOT / "lib"))
print("V2 root:", V2_ROOT)
print("Script:", SCRIPT)
'''


def run_cell(stage: dict) -> str:
    if stage.get("keep_smoke_default"):
        comment = f'# Smoke test. For production, replace RUN_ARGS with {stage["full_args"]}.'
    else:
        comment = (
            f'# Full run (RUN_ARGS above = {stage["full_args"]}). '
            f'To sample-test first, set RUN_ARGS = {stage["args"]} instead.'
        )
    return f'''{comment}
sys.argv = [str(SCRIPT), *RUN_ARGS]
runpy.run_path(str(SCRIPT), run_name="__main__")
'''


def validation_cell(stage: dict) -> str:
    return '''from pathlib import Path
import json

output_root = Path("/kaggle/working/aic-v2")
markers = list(output_root.rglob("_SUCCESS.json")) if output_root.exists() else []
print("Success markers:", len(markers))
for marker in markers[-10:]:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        print(marker, payload.get("stage"), payload.get("video_id"), payload.get("success"))
    except Exception as exc:
        print("Invalid marker:", marker, exc)
'''


def notebook(stage: dict) -> dict:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": markdown(stage).splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": (stage["install"] + "\n").splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": bootstrap_cell(stage).splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_cell(stage).splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": validation_cell(stage).splitlines(keepends=True)},
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for stage in STAGES:
        directory = ROOT / stage["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / stage["filename"]
        path.write_text(json.dumps(notebook(stage), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
