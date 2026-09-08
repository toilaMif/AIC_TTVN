# Báo cáo kiểm tra `notebooks_V2`

Ngày kiểm tra: 2026-08-26  
Phạm vi: `pipelines/kaggle/notebooks_V2`

## 1. Kết quả Kaggle

Đã kiểm tra qua Kaggle MCP:

- Notebook: `thnhtrungnguynmif/aic-v2-l21-v001-extraction`
- Tiêu đề: `AIC V2 L21 V001 Extraction`
- Notebook ID: `131969520`
- Version hiện tại: `17`
- Phiên Kaggle: `COMPLETE` (phiên đã kết thúc, không đồng nghĩa pipeline thành công)
- GPU: `NvidiaTeslaT4`
- Internet: bật
- Video mục tiêu: `L21_V001`
- Output ZIP: `L21_V001-v2-result.zip`
- SHA-256: `418A875808391E89AB990046E4B9ACFC8A3FB78B85A0CF39A6BB152A2D377434`

`pipeline-summary.json` trong output ghi rõ `success: false`. Pipeline chỉ hoàn
tất các bước sau:

| Bước | Trạng thái |
|---|---|
| Tải dataset L21 | Thành công, return code 0 |
| Cài `av`, OpenCV, pandas, PyArrow, Pillow | Thành công, return code 0 |
| Cài `open_clip_torch`, `timm`, `ftfy` | Thành công, return code 0 |
| Cài `scenedetect[opencv]` | Thành công, return code 0 |
| Cài `paddlepaddle-gpu` | Pipeline lỗi trước khi ghi log |
| Stage 00--08 | Chưa chạy |

Lỗi gốc:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/kaggle/working/04-install-paddlepaddle-gpu__3.3.1---index-url-https:/www.paddlepaddle.org.cn/packages/stable/cu126/.log'
```

Nguyên nhân trực tiếp là tên file log được tạo từ URL `https://...`; dấu `/`
trở thành thư mục con chưa tồn tại. Đây là lỗi của wrapper notebook, không phải
lỗi xử lý video của stage V2.

Vì vậy, số notebook Kaggle chạy thành công toàn pipeline là **0/1** trong link
được kiểm tra. Có output Kaggle để lưu, nhưng chỉ là output chẩn đoán; không có
`_SUCCESS.json`, `index-manifest.json`, `eval-report.json` hay dữ liệu stage V2.
Output gốc được lưu tại `L21_V001-v2-result.zip`.

## 2. Kiểm thử cục bộ bổ sung

Đã chạy script smoke test CPU có sẵn trong repo:

```powershell
& '.venv\Scripts\python.exe' `
  'pipelines\kaggle\notebooks_V2\scripts\smoke_test.py'
```

Kết quả: mã thoát `0`, dòng kết thúc `AIC V2 CPU smoke test: PASS`.

Smoke test dùng dữ liệu tổng hợp `L99_V001`, không tải video, không gọi Kaggle
và không đại diện cho chất lượng trên dữ liệu cuộc thi. Các kiểm tra chính:

- Tạo 2 frame document, 2 shot document và 1 event document.
- Tạo 80 sparse postings và snapshot SQLite FTS5 với 5 document.
- Tạo 2 vector frame và 2 vector shot, dimension 4, `float32`, L2-normalized.
- Kiểm tra tìm kiếm FTS, hybrid/RRF, lọc object và truy vấn visual NumPy.
- Chạy benchmark 2 query; p50 `2.14795 ms`, p95 `2.66181 ms`, p99
  `2.700762 ms` ở lần chạy không nhãn.
- Chạy benchmark có nhãn mẫu; recall@1 theo `video_id` là `1.0`, theo
  `shot_id` là `0.5`, theo `doc_id` là `0.25`.
- FAISS không được cài trong môi trường local nên ANN index không được build;
  smoke test xác nhận fallback payload vector vẫn hoạt động.

## 3. Artifact đã lưu

- `execution-status.json`: số liệu kiểm tra 9 notebook.
- `local-smoke-test-output.txt`: kết quả smoke test và các chỉ số chính.

Khi có file ZIP hoặc thư mục output từ một Kaggle session, có thể đặt artifact
vào `result` và chạy lại audit để cập nhật báo cáo này.
