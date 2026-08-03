# PIPELINE — Hệ thống truy vấn video AIC

> Kế hoạch xây dựng hệ thống truy vấn sự kiện từ video (AI Challenge).
> Stack: **React + FastAPI + AI pipeline**, chạy trên **Docker**, deploy ra ngoài bằng **Vast.ai**.
> Cập nhật: 2026-08-02.

---

## 1. Mục tiêu & bối cảnh

- **Bài toán:** người dùng gõ mô tả (tiếng Việt) → hệ thống trả về các keyframe/khoảnh khắc khớp nhất trong kho video.
- **Cuộc thi AIC** → cần **xuất file nộp bài** đúng format `(video_id, frame_idx)` và các tính năng chuyên biệt (temporal query, KIS/QA).
- **Đa phương thức tìm kiếm:** Text→ảnh (CLIP) + OCR (chữ trong ảnh) + ASR (lời thoại) + lọc object.
- **Quy mô lớn:** nghìn+ giờ video, hàng triệu keyframe.
- **Truy vấn:** nhập tiếng Việt → **dịch sang tiếng Anh** → đưa vào CLIP tiếng Anh (ViT-L/H).
- **Máy dev:** chỉ CPU (data nhỏ để test) → **GPU thuê Vast.ai** khi build index thật.

---

## 2. Kiến trúc tổng thể

```
┌─────────────┐   HTTP/JSON   ┌──────────────┐   vector search   ┌─────────────┐
│ React (UI)  │ ─────────────▶│ FastAPI      │ ─────────────────▶│ FAISS index │
│ - ô tìm kiếm│               │ - /search    │                   │ + metadata  │
│ - lưới ảnh  │◀───────────── │ - /video     │◀───────────────── │ (keyframes) │
│ - export    │   kết quả     │ - /export    │                   └─────────────┘
└─────────────┘               └──────┬───────┘
                                     │ dịch TV→EN → CLIP text encode
                                     ▼
                       AI pipeline (offline indexing)
```

### Luồng offline (index dữ liệu — chạy trên GPU)
```
video/*.mp4 → [1] trích keyframe → [2] CLIP embed ảnh → [3] FAISS index
                                 → [4] OCR / ASR / object
                             tất cả → metadata store (parquet/sqlite)
```

### Luồng online (truy vấn)
```
query TV → dịch EN → CLIP text encode → FAISS search → gộp metadata → trả React
```

---

## 3. Cấu trúc thư mục dự kiến

```
AIC_TTVN/
├── ai/                      # AI pipeline (offline indexing + search engine)
│   ├── config.yaml          # cấu hình trung tâm (model, path, faiss...)
│   ├── requirements.txt
│   ├── videoquery/          # package Python
│   │   ├── config.py        # đọc config
│   │   ├── keyframes.py     # trích keyframe (OpenCV)
│   │   ├── embedder.py      # CLIP encode ảnh/text (open_clip)
│   │   ├── index.py         # build/load FAISS + metadata
│   │   ├── search.py        # engine truy vấn (dịch → encode → search)
│   │   ├── ocr.py           # (GĐ4) OCR chữ trong khung hình
│   │   ├── asr.py           # (GĐ4) ASR lời thoại
│   │   └── objects.py       # (GĐ4) object detection / lọc
│   └── scripts/
│       ├── build_index.py   # CLI: extract → embed → index
│       └── query.py         # CLI: test truy vấn nhanh
├── backend/                 # FastAPI serve API
│   ├── main.py
│   ├── routers/ (search, video, export)
│   └── requirements.txt
├── frontend/                # React (Vite)
│   └── src/ (SearchBar, ResultGrid, VideoPlayer, ExportPanel)
├── docker-compose.yml       # (GĐ5) chạy cả stack
└── data/                    # (gitignore) raw video + processed index
```

---

## 4. Lộ trình theo giai đoạn

### GĐ1 — AI pipeline lõi ⭐ (làm trước)
Mục tiêu: chạy được end-to-end trên vài video mẫu bằng CPU.
- [ ] `config.yaml` + `config.py` — cấu hình trung tâm.
- [ ] `keyframes.py` — trích keyframe (interval / scene) bằng OpenCV.
- [ ] `embedder.py` — CLIP encode ảnh & text (open_clip), normalize vector.
- [ ] `index.py` — build FAISS (Flat cho dev) + lưu metadata parquet.
- [ ] `search.py` — dịch TV→EN → encode text → search → trả kết quả.
- [ ] `scripts/build_index.py`, `scripts/query.py` — CLI test.
- **Kiểm chứng:** bỏ 1–2 video vào `data/raw/videos`, chạy `build_index.py`, rồi `query.py "người đang nấu ăn"` xem có trả về keyframe hợp lý.

### GĐ2 — Backend FastAPI
- [ ] `POST /search` — nhận query → gọi search engine → trả JSON (list keyframe + score + timestamp).
- [ ] `GET /keyframe/{id}` và `GET /video/{id}` — serve ảnh & stream video theo mốc thời gian.
- [ ] `POST /export` — xuất CSV nộp bài đúng format AIC.
- [ ] CORS cho frontend.

### GĐ3 — Frontend React (Vite)
- [ ] Ô tìm kiếm + chọn top_k.
- [ ] Lưới keyframe (thumbnail + video_id + timestamp + score).
- [ ] Click keyframe → mở player nhảy đúng giây.
- [ ] Giỏ chọn kết quả + nút Export CSV.

### GĐ4 — Đa phương thức (OCR / ASR / object)
- [ ] `ocr.py` (EasyOCR) — index chữ trong keyframe → tìm theo text.
- [ ] `asr.py` (faster-whisper) — transcript lời thoại theo mốc thời gian.
- [ ] `objects.py` (YOLO) — nhãn object để lọc.
- [ ] Gộp điểm đa phương thức + lọc trong `/search`.
- [ ] Temporal query (nhiều sự kiện nối tiếp) cho AIC.

### GĐ5 — Docker & deploy Vast.ai
- [ ] `Dockerfile` cho ai/backend (base `nvidia/cuda` để dùng GPU).
- [ ] `docker-compose.yml`: backend + frontend (+ volume cho index).
- [ ] Đổi `faiss-cpu → faiss-gpu`, `device: cuda`, index `Flat → IVFPQ`.
- [ ] Hướng dẫn thuê GPU Vast.ai, mount data, expose port.

---

## 5. Quyết định kỹ thuật đã chốt

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Đọc video | **OpenCV** (`opencv-python-headless`) | bundle sẵn codec, khỏi cài ffmpeg riêng → dễ Docker |
| CLIP | **open_clip** ViT-B/32 (dev) → ViT-L/H (prod) | linh hoạt đổi model qua config |
| Vector search | **FAISS** Flat (dev) → IVFPQ (prod) | Flat chính xác cho data nhỏ; IVFPQ nén + nhanh khi triệu vector |
| Metadata | **Parquet** (pandas/pyarrow) | nhẹ, đọc nhanh, không cần DB server |
| Dịch query | **deep-translator** (TV→EN) | đơn giản cho dev; có thể đổi model offline sau |
| Ngôn ngữ | query TV → dịch EN → CLIP EN | độ chính xác cao, cách đa số team AIC dùng |

**Scale-up:** device (cpu/cuda), model, loại FAISS index đều đọc từ `config.yaml` → chuyển dev↔prod chỉ bằng sửa config, không sửa code.

---

## 6. Định dạng dữ liệu

**metadata.parquet** (mỗi dòng = 1 keyframe, khớp thứ tự vector trong FAISS):

| cột | ý nghĩa |
|---|---|
| `id` | chỉ số vector trong FAISS (0..N-1) |
| `video_id` | tên video, vd `L01_V001` |
| `frame_idx` | chỉ số frame gốc (dùng để nộp bài) |
| `pts_time` | mốc thời gian (giây) |
| `keyframe_path` | đường dẫn ảnh keyframe |

**Export nộp bài AIC:** CSV `video_id, frame_idx` theo thứ tự điểm giảm dần.

---

## 7. Ghi chú tiến độ

- Đã tạo khung ban đầu ở GĐ1: `ai/requirements.txt`, `ai/config.yaml`, `ai/videoquery/{__init__,config,keyframes}.py`.
- **Bước tiếp theo khi bắt đầu code:** hoàn thiện `embedder.py` → `index.py` → `search.py` → 2 script CLI, rồi test trên video mẫu.
