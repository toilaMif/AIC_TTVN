# Data layout

```text
data/
|-- source/aic2026/
|   |-- map-keyframes/   BTC keyframe mapping CSV files
|   `-- media-info/      BTC video metadata JSON files
`-- manifests/           generated audit and selection manifests
```

Only small source metadata and manifests belong here. Videos, extracted frames,
embeddings, OCR, ASR and object detections belong under `.runtime` by default,
or in a separate data drive configured through `.env`.
