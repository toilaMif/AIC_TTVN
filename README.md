# AIC_TTVN

**Hệ thống truy vấn sự kiện từ video bằng đa phương thức** — _Vietnamese AI Challenge video retrieval system_

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb)](https://vite.dev/)
[![AI Challenge](https://img.shields.io/badge/competition-AI%20Challenge-7c3aed)](#)

---

Người dùng nhập mô tả **tiếng Việt** → hệ thống trả về các **keyframe / khoảnh khắc khớp nhất** trong kho video, sử dụng tìm kiếm đa phương thức: **CLIP** (text↔ảnh), **OCR** (chữ trong ảnh), **ASR** (lời thoại) và **lọc object**.

> Users type a **Vietnamese** description and get the **best-matching keyframes** across a video corpus, powered by multimodal retrieval: **CLIP** (text↔image), **OCR** (in-frame text), **ASR** (speech), and **object filtering**.

## ✨ Tính năng / Features

| Tiếng Việt | English |
|---|---|
| Truy vấn ngữ nghĩa Text→Ảnh bằng CLIP | Semantic text-to-image retrieval with CLIP |
| Dịch tự động query TV→EN trước khi encode | Automatic query translation VI→EN before encoding |
| Nhận diện chữ (OCR) & lời thoại (ASR) | In-frame text (OCR) & speech (ASR) recognition |
| Lọc theo đối tượng (object detection) | Object-based filtering |
| Temporal query — chuỗi sự kiện nối tiếp | Temporal query — sequences of events |
| Xuất file nộp bài đúng format AIC | Export submissions in the official AIC format |

## 🏗 Kiến trúc / Architecture

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

- **Luồng offline / Offline:** `video → keyframe → CLIP embed → FAISS index + metadata (OCR/ASR/object)` — chạy trên GPU.
- **Luồng online / Online:** `query TV → dịch EN → encode → FAISS search → gộp metadata → trả về UI`.

## 📁 Cấu trúc thư mục / Project structure

```
AIC_TTVN/
├── ai/                      # AI pipeline — indexing & search engine (Python)
├── backend/                 # FastAPI — phục vụ API / serves the API
├── frontend/                # React (Vite) — giao diện / web UI
├── docs/                    # Tài liệu bổ sung / supplementary docs
├── data/                    # (gitignored) raw video + processed index
├── scripts/                 # Tiện ích thao tác repo / repo utilities
├── docker-compose.yml       # (GĐ5) chạy cả stack / run the full stack
└── README.md
```

Chi tiết hướng dẫn trong từng thư mục — _see each subfolder for details_:
[`ai/`](ai/README.md) · [`backend/`](backend/README.md) · [`frontend/`](frontend/README.md) · [`docs/`](docs/README.md)

## 🚀 Bắt đầu nhanh / Quick start

> Yêu cầu / _Prereqs_: Python ≥ 3.11, Node.js ≥ 20, npm

### 1. Cài đặt / Install

```bash
# AI pipeline
cd ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Backend
cd ../backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2. Cấu hình / Configure

```bash
cp .env.example .env        # điền các giá trị cần thiết / fill in the values
```

### 3. Chạy dev / Run

```bash
# Backend (port 8000)
uvicorn main:app --reload

# Frontend (port 5173)
npm run dev
```

## 🧪 Kiểm thử / Testing

```bash
# Python (ai/ + backend/)
pytest

# Frontend
npm test
```

## 📄 Tài liệu tham khảo / References

- [`pipeline.md`](pipeline.md) — kế hoạch & lộ trình chi tiết / detailed plan & roadmap (VI)
- [`doing.md`](doing.md) — ghi chú việc đang làm / work-in-progress notes

## 🧑‍💻 Đóng góp / Contributing

Vui lòng đọc [`CONTRIBUTING.md`](CONTRIBUTING.md) trước khi tạo pull request — _please read before opening a PR_.

## ⚖️ Giấy phép / License

[MIT](LICENSE) © 2026 AIC_TTVN contributors
