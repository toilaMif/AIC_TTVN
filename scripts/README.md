# Scripts

Các script vận hành chính:

| Script | Mục đích |
|---|---|
| `setup-local.ps1` | Thiết lập dependency, frontend, Docker và database lần đầu |
| `start-local-ui.ps1` | Khởi động Docker, FastAPI và mở trình duyệt |
| `stop-local.ps1` | Dừng FastAPI và Docker, không xóa dữ liệu |
| `check-repository.ps1` | Kiểm tra build, cấu hình, file lớn và credential trước khi push |
| `init-kaggle-batch.ps1` | Tạo cấu trúc thư mục cho một batch Kaggle |
| `rebuild_kaggle_index.py` | Xóa và dựng lại visual, ASR, OCR và object index từ artifact Kaggle |
| `repair_feature_records.py` | Sửa metadata feature khi import bị gián đoạn |

`rebuild_kaggle_index.py` là lệnh destructive đối với dữ liệu derived trong
PostgreSQL, MinIO và Milvus. Luôn kiểm tra batch và sao lưu trước khi chạy.

Dùng `start-local-ui.ps1 -NoBrowser` khi chỉ muốn khởi động/kiểm tra API mà
không tự mở trình duyệt.
