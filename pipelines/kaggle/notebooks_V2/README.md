# AIC-TTVN Kaggle Pipeline V2

V2 is a performance-oriented replacement for `pipelines/kaggle/notebooks`.
It keeps the existing `keyframe_id`, `shot_id`, `frame_idx`, `pts_time`,
`shots.parquet`, `keyframes.parquet`, `visual.npy`, and modality output names so
the current import path can still be used. It adds a manifest, timestamp-aware
fusion, shot/event documents, a portable sparse index, and an evaluation report.

## Why this version is faster

- Stage 01 writes the OpenCLIP vector while selecting the keyframe. The old
  stage 02 encoded every WebP a second time; V2 removes that duplicate GPU pass.
- Every later stage reads `video-manifest.parquet` and a cached path map. It does
  not repeatedly `rglob` the input tree or choose a stale duplicate by filename.
- Checkpoints are append-only JSONL shards with atomic manifests. A restart
  resumes by `video_id` or `keyframe_id` without rewriting a growing file.
- ASR is joined to keyframes by an interval sweep, not one SQL query per result.
- Search documents are emitted at frame, shot, and event levels. Retrieval can
  search a much smaller shot/event index first and expand to keyframes only for
  presentation.
- `06-index-pack` creates a SQLite FTS5/BM25-compatible snapshot and normalized
  ANN-ready vectors. The existing Milvus importer can consume the frame-level
  `visual.npy`; a separate ANN loader can use the shot vectors.

## Execution order

```text
00 manifest/normalization
          |
01 shot + adaptive keyframe + visual (single GPU pass)
       /       |        \
  02 OCR   03 ASR    04 objects
       \       |        /
        05 temporal fusion + compact captions
                       |
                06 retrieval index pack
                       |
                07 benchmark/evaluation
                       |
                08 hybrid query / RRF
```

Stages 02, 03, and 04 are independent after stage 01. Each notebook has a
smoke-test limit (`MAX_VIDEOS`, `MAX_FRAMES`, or `MAX_SHOTS`) that defaults to a
small value; set it to `None` for the full run.

**Stage 01 requires an extra dataset attachment.** The upstream
`soCzech/TransNetV2` repo stores its pretrained weights
(`inference/transnetv2-weights/*`) via Git LFS, and that repo's LFS bandwidth
budget is currently exhausted (confirmed on a real Kaggle T4 run — plain
`git clone` fails with `This repository exceeded its LFS budget`). Attach the
Kaggle dataset `thnhtrungnguynmif/aic-ttvn-transnetv2-weights` (a same-bytes
mirror of just the 3 weight files) via "Add Data" before running stage 01 —
the install cell copies them into place after a Git-LFS-free clone.

## Kaggle paths

The defaults are:

```text
input:  /kaggle/input
output: /kaggle/working/aic-v2
```

Set environment variables before running a notebook when using a different
dataset layout:

```python
import os
os.environ["AIC_V2_INPUT_ROOT"] = "/kaggle/input/my-videos"
os.environ["AIC_V2_OUTPUT_ROOT"] = "/kaggle/working/aic-v2"
```

The notebook wrappers locate `scripts/` and `lib/` from either the checked-out
repository or an uploaded Kaggle dataset containing this directory.

## Output contract

Each video under `01-shot-keyframe/<VIDEO_ID>/` contains the legacy-compatible
frame artifacts plus `visual.npy` and `embedding-records.parquet`. Later stage
outputs are under their own stage and video directories. The final pack is:

```text
07-index-pack/
  frame-documents.parquet
  shot-documents.parquet
  event-documents.parquet
  sparse-postings.parquet
  text-index.sqlite          # FTS5 + metadata/temporal tables when supported
  frame-visual.npy           # consolidated, L2-normalized frame vectors
  frame-embedding-records.parquet
  shot-visual.faiss          # optional FAISS HNSW/IVF snapshot
  shot-visual.faiss.ids.parquet
  index-manifest.json
```

`text-index.sqlite` is queryable without loading Parquet. It contains the
`documents` table, level/video/time indexes, `document_facets`,
`document_objects`, `temporal_edges`, and the backwards-compatible
`documents_fts` FTS5 table. Use `v2_runtime.search_hybrid_snapshot(...)` to
run level-aware FTS and fuse it with an external ANN ranking using weighted RRF.
When `faiss-cpu` is installed, stage 06 writes a real shot HNSW/IVF index and
an ID mapping; otherwise `index-manifest.json` records `ann.built=false` and
the normalized vector payload remains available for Milvus or another loader.

Stage 08 is a portable serving reference:

```bash
python scripts/08_hybrid_query.py "người đứng bên ngoài trời mưa" \
  --index-root /kaggle/working/aic-v2/07-index-pack \
  --levels shot,event --top-k 20 --object person --scene outdoor
```

It pushes lexical filters into SQLite, prefers the FAISS shot index, falls back
to exact NumPy memmap search, and returns one result per shot by default. Add
`--visual` after testing OpenCLIP, or pass `--query-vector path.npy` when the
caller already has a compatible query embedding. This is an integration
reference; production can route the same rankings through Milvus/PostgreSQL.

All joins use IDs and timestamps. Never join by row order or by the basename of
an image. Every completed video has `_SUCCESS.json`; failed videos have
`_ERROR.txt` and are retried on the next run.

When compatibility ZIPs are enabled in stage 01, they are written directly to
`01-shot-keyframes/` and `02-visual-embeddings/` with the filename pattern
expected by `scripts/rebuild_kaggle_index.py`. OCR, ASR, and object outputs stay
as per-video directories, matching the current batch importer.

## Models and profiles

The default quality profile uses the same models as V1 for compatibility:

| Stage | Default | Fast profile |
|---|---|---|
| shot/visual | TransNetV2 + OpenCLIP ViT-B-32 | TransNet fallback + OpenCLIP |
| OCR | PaddleOCR PP-OCRv5 | same model, smaller batch |
| ASR | faster-whisper large-v3, `int8_float16` | medium/small model |
| objects | Ultralytics YOLO26m COCO | YOLO26s COCO |
| caption | evidence-first fusion; optional Qwen2.5-VL | skip VLM and use evidence |

VLM calls are disabled by default in stage 05 because stable OCR/ASR/object
evidence is cheaper and immediately indexable. Set `AIC_V2_ENABLE_VLM=1` only
after the fused evidence has passed a smoke test. If enabled, call the VLM for
one representative frame per shot and for evidence changes only; the frame
document records whether a field was direct or propagated.

## Integration notes

`visual.npy` is float32 and L2-normalized to remain compatible with
`retrieval/storage/self_extracted_import.py`. The new shot vectors are also
float32 by default; they can be quantized by the downstream Milvus loader after
recall has been measured. Use an ANN index (HNSW/IVF_FLAT/AUTOINDEX) for the
shot collection rather than the current frame-level FLAT index when the corpus
grows beyond a smoke test.

The current application does not yet import frame-understanding documents into
PostgreSQL. V2 therefore exports all fields needed for a small importer: use
`search_text_vi`, `asr_text`, `ocr_scene_text`, `object_labels_text`, scene
facets, and the temporal edge tables. Do not spend VLM compute until this pack
is connected to the server-side hybrid/RRF path.
