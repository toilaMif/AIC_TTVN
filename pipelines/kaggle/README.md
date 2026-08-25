# Kaggle pipelines

`downloads/` chua notebook tai video theo batch. `notebooks/` chua cac stage
trich xuat doc lap:

1. Shot detection va keyframe selection.
2. Visual embedding.
3. OCR.
4. ASR.
5. Object detection.
6. Frame understanding: join object + OCR, then generate VLM descriptions and
   six general scene labels (`daytime`, `night`, `rain`, `sunny`, `indoor`,
   `outdoor`).

Recommended single-video order:

```text
01-shot-keyframe
05-object-detection
03-ocr
06-frame-understanding
```

Stages 05 and 03 can run independently after stage 01. Stage 06 joins their
artifacts by `keyframe_id`; do not join by row order or filename. It writes
`search-documents.parquet` for indexing and `scene-facets.json` for the UI scene
filter. See `notebooks/06-frame-understanding/README.md` for the output contract.

Quick test limits are `MAX_FRAMES = 64` in stage 05,
`OCR_DEBUG_MAX_FRAMES = 64` in stage 03 and `MAX_SHOTS = 20` in stage 06. Set all
three values to `None` only after the sample output looks correct.

Notebook ghi output ZIP theo cấu trúc trong `artifacts/README.md`. Artifact lớn
nên tải về một ổ dữ liệu riêng và cấu hình `AIC_KAGGLE_ARTIFACT_ROOT` trong
`.env`; không commit chúng vào repository.
