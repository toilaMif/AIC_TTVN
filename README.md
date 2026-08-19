# AIC-TTVN

Hệ thống truy xuất video đa phương thức dành cho AIC, gồm giao diện React,
FastAPI, PostgreSQL, MinIO, Milvus và các pipeline trích xuất đặc trưng chạy
trên Kaggle.

Hiện tại ứng dụng hỗ trợ tìm kiếm hình ảnh bằng OpenCLIP và tìm kiếm lời nói
ASR. Notebook OCR và object detection đã được tổ chức trong pipeline, nhưng
chưa nối thành chỉ mục tìm kiếm trên giao diện.

## Kiến trúc

```text
Kaggle artifacts ──> Import pipeline ──> PostgreSQL (metadata, ASR)
                                      ├─> Milvus (visual vectors)
                                      └─> MinIO (keyframes, artifacts)
                                                  │
React UI <────────────── FastAPI <────────────────┘
```

| Thành phần | Vai trò |
|---|---|
| `apps/frontend` | Giao diện React/Vite |
| `apps/api` | REST API và phục vụ bản frontend đã build |
| `retrieval` | Import dữ liệu, lưu trữ và tìm kiếm |
| `pipelines/kaggle` | Notebook trích xuất shot, embedding, OCR, ASR và object |
| `data/source/aic2026` | Metadata và keyframe mapping nhỏ từ ban tổ chức |
| `artifacts` | Quy ước lưu artifact; không chứa dữ liệu lớn trên Git |
| `alembic` | Migration PostgreSQL |
| `scripts` | Script cài đặt, chạy, dừng và kiểm tra repository |

## Lưu ý về dữ liệu

GitHub chỉ chứa mã nguồn và metadata nhỏ. Repository **không chứa** video,
frame ảnh, embedding, transcript ASR, dữ liệu PostgreSQL, MinIO hoặc Milvus.

Vì vậy, sau khi clone:

- Có thể dựng hạ tầng, mở UI và kiểm tra API health ngay.
- Chức năng tìm kiếm chỉ hoạt động sau khi import artifact Kaggle; trước đó API
  tìm kiếm có thể báo chưa tồn tại collection/index.
- Thành viên trong nhóm cần nhận bộ artifact từ người quản lý dự án hoặc tự
  chạy notebook Kaggle để tạo lại.

## Yêu cầu

Khuyến nghị máy có ít nhất 8 GB RAM và còn khoảng 10 GB dung lượng trống cho
dependency, Docker image và model. Dữ liệu thật sẽ cần thêm dung lượng tùy số
lượng video.

| Công cụ | Phiên bản khuyến nghị |
|---|---|
| Git | Bản mới ổn định |
| Docker Desktop | Có Docker Compose, Docker Engine đang chạy |
| Python | `3.11.x` |
| uv | `0.8+` |
| Node.js | `20+` |
| npm | Đi kèm Node.js |

Cài `uv` nếu máy chưa có:

```powershell
winget install --id=astral-sh.uv -e
```

## Chạy nhanh trên Windows

Đây là luồng được kiểm tra chính thức của repository.

### 1. Clone repository

```powershell
git clone https://github.com/toilaMif/AIC_TTVN.git
cd AIC_TTVN
```

### 2. Thiết lập lần đầu

Mở Docker Desktop, đợi Docker Engine chạy xong rồi thực hiện:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-local.ps1
```

Script này sẽ:

1. Tạo `.env` từ `.env.example` nếu chưa có.
2. Tạo thư mục dữ liệu local trong `.runtime`.
3. Cài dependency Python từ `uv.lock`.
4. Cài và build frontend React.
5. Khởi động PostgreSQL, MinIO, etcd và Milvus.
6. Chạy migration database.

### 3. Mở ứng dụng

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-local-ui.ps1
```

Các địa chỉ local:

| Dịch vụ | Địa chỉ |
|---|---|
| Giao diện | <http://127.0.0.1:8000/ui/> |
| API health | <http://127.0.0.1:8000/health> |
| API docs | <http://127.0.0.1:8000/docs> |
| MinIO console | <http://127.0.0.1:9001> |

### 4. Dừng hệ thống

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop-local.ps1
```

Lệnh dừng không xóa dữ liệu trong `.runtime`. Những lần sau chỉ cần chạy lại
`scripts/start-local-ui.ps1`.

## Cài đặt thủ công trên macOS/Linux

Sửa các đường dẫn trong `.env` nếu cần, sau đó chạy. Trên máy Apple Silicon,
Milvus có thể cần bật chế độ tương thích `amd64` của Docker Desktop.

```bash
cp .env.example .env
mkdir -p .runtime/postgres .runtime/minio .runtime/milvus .runtime/etcd
mkdir -p .runtime/artifacts/kaggle
uv sync --frozen
npm --prefix apps/frontend ci
npm --prefix apps/frontend run build
docker compose up -d
uv run alembic upgrade head
uv run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000/ui/>. Dừng API bằng `Ctrl+C`, sau đó dừng Docker:

```bash
docker compose down
```

## Nạp dữ liệu để tìm kiếm

### 1. Import metadata nguồn

Trên database mới, chạy:

```powershell
uv run python -m retrieval.cli.data audit
uv run python -m retrieval.cli.data import-metadata
```

### 2. Chuẩn bị artifact Kaggle

Tạo cấu trúc cho một batch, ví dụ `l21`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/init-kaggle-batch.ps1 -Batch l21
```

Đặt ZIP được tải từ Kaggle vào đúng stage:

```text
.runtime/artifacts/kaggle/l21/
├── 01-shot-keyframes/
├── 02-visual-embeddings/
├── 03-ocr/
├── 04-asr/
├── 05-object-detection/
└── _runtime/
```

Ví dụ tên file:

```text
01-shot-keyframe-L21_V001.zip
02-visual-embedding-L21_V001.zip
04-asr-vietnamese-L21_V001.zip
```

Nếu artifact lớn, nên lưu ngoài repository. Ví dụ sửa `.env`:

```dotenv
AIC_KAGGLE_ARTIFACT_ROOT=D:/AIC_TTVN_DATA/artifacts/kaggle
AIC_KAGGLE_BATCH=l21
```

### 3. Rebuild chỉ mục

Kiểm tra đúng batch trong `.env`, sau đó chạy:

```powershell
uv run python scripts/rebuild_kaggle_index.py --confirm-rebuild
```

> Cảnh báo: lệnh này xóa và dựng lại dữ liệu đặc trưng hiện tại trong
> PostgreSQL, MinIO và Milvus. Không chạy chỉ để mở UI. Với máy đã có index,
> hãy sao lưu trước khi rebuild.

Lần truy vấn visual đầu tiên có thể chậm do OpenCLIP tải model về máy. Các lần
sau model được dùng từ cache local.

## Chạy frontend ở chế độ phát triển

Terminal 1:

```powershell
uv run uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
npm --prefix apps/frontend run dev
```

Mở <http://127.0.0.1:5173>. Vite sẽ proxy các request API sang FastAPI.

## Kiểm tra trước khi đưa lên GitHub

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check-repository.ps1
git status
git diff --check
```

Nếu mọi kiểm tra đều đạt, xem kỹ danh sách file rồi mới commit:

```powershell
git add -A
git status
git commit -m "Prepare AIC-TTVN for local development"
git push origin main
```

Không được commit `.env`, `.runtime`, video, model, embedding, file ZIP Kaggle
hoặc credential thật. Xem thêm [hướng dẫn artifact](artifacts/README.md) và
[hướng dẫn pipeline Kaggle](pipelines/kaggle/README.md).

## Xử lý lỗi thường gặp

### Docker chưa chạy

Mở Docker Desktop và chờ trạng thái Engine running, sau đó chạy lại setup.

### Port đã được sử dụng

Các port mặc định là `8000`, `55432`, `9000`, `9001` và `19530`. Có thể đổi
port tương ứng trong `.env` nếu máy đang dùng các port này.

### UI vẫn là bản cũ

Build lại frontend rồi tải cứng trình duyệt bằng `Ctrl+F5`:

```powershell
npm --prefix apps/frontend run build
```

### Truy vấn chưa hoạt động hoặc không có kết quả

Kiểm tra đã import metadata và artifact Kaggle chưa. Một bản clone mới chỉ có
database rỗng nên UI và health endpoint chạy được nhưng chưa có index để tìm
kiếm.
