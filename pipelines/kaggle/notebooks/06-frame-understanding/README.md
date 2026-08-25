# Frame understanding output contract

Run the single-video notebooks in this order:

1. `01-shot-keyframe-single-video.ipynb`
2. `05-object-detection-single-video-optimized.ipynb`
3. `03-ocr-single-video-optimized.ipynb`
4. `06-frame-understanding-single-video.ipynb`

Object detection and OCR are independent, but both artifacts must be attached as
Kaggle inputs before stage 06 starts. Every join uses `keyframe_id`; filenames and
directory order are not identifiers.

For a quick end-to-end test, keep these defaults:

```python
# 05 object detection
MAX_FRAMES = 64

# 03 OCR
OCR_DEBUG_MAX_FRAMES = 64

# 06 frame understanding
MAX_SHOTS = 20
```

After checking the sample outputs, run the complete video with:

```python
MAX_FRAMES = None
OCR_DEBUG_MAX_FRAMES = None
MAX_SHOTS = None
```

Run stages 01, 05 and 03 as separate Kaggle notebooks. Attach all three output
datasets to stage 06, then run every cell in stage 06 from top to bottom.

## Outputs

- `frame-understanding.jsonl`: complete nested record for every keyframe.
- `frame-understanding.parquet`: analytics form with nested values encoded as JSON columns.
- `search-documents.parquet`: compact input for PostgreSQL or a text/vector index.
- `search-documents.jsonl`: the same search documents for streaming import.
- `shot-scenes.parquet`: one representative scene record per shot.
- `scene-facets.json`: the six fixed dropdown values and their result counts.
- `vlm-selection.parquet`: audit trail showing which frames actually used the VLM.
- `frame-understanding-summary.json`: model, coverage, propagation and error counts.

## Scene fields

The UI exposes only six general scene labels:

- `daytime`
- `night`
- `rain`
- `sunny`
- `indoor`
- `outdoor`

The authoritative structured fields are `environment`, `time_of_day` and `weather`.
`scene_labels` combines their known values for a simple dropdown membership filter.
Detailed places and content categories are intentionally not scene filters.

For example, one frame can have:

```json
{
  "environment": "outdoor",
  "time_of_day": "night",
  "weather": "rain",
  "scene_labels": ["night", "rain", "outdoor"]
}
```

## System integration

Use `search-documents.parquet` as the import boundary. Upsert by `keyframe_id` and
keep it as a foreign key to the existing `keyframes` table.

- Full-text or semantic search input: `search_text_vi`.
- Result subtitle: `caption_vi`.
- Scene filter: array membership in `scene_labels_json`. Only the six values above
  are valid.
- Detail panel: `main_subjects_json`, `ocr_text` and `object_labels_text`.
- Video navigation: `video_id`, `frame_idx`, `pts_time` and `shot_id`.

The frontend can build its scene menu from `scene-facets.json`. Selecting `rain`
filters for array membership in `scene_labels`. Object names, animal species, produce,
colors and actions remain searchable through `main_subjects_json` and
`search_text_vi`.
