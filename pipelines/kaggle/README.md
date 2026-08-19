# Kaggle pipelines

`downloads/` chua notebook tai video theo batch. `notebooks/` chua cac stage
trich xuat doc lap:

1. Shot detection va keyframe selection.
2. Visual embedding.
3. OCR.
4. ASR.
5. Object detection.

Notebook ghi output ZIP theo cấu trúc trong `artifacts/README.md`. Artifact lớn
nên tải về một ổ dữ liệu riêng và cấu hình `AIC_KAGGLE_ARTIFACT_ROOT` trong
`.env`; không commit chúng vào repository.
