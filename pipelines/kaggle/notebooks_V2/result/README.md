# Kết quả kiểm tra notebooks_V2

Thư mục này lưu báo cáo kiểm tra execution, output Kaggle và kết quả smoke test
của `pipelines/kaggle/notebooks_V2`.

Đã kiểm tra notebook Kaggle `thnhtrungnguynmif/aic-v2-l21-v001-extraction` qua
MCP. Notebook có trạng thái phiên `COMPLETE`, nhưng pipeline bên trong ghi
`success: false` và dừng trước stage 00 vì lỗi tạo tên file log chứa dấu `/`.
ZIP output Kaggle đã được lưu lại; không có artifact stage 00--08 trong ZIP.
Smoke test được lưu ở đây là kết quả chạy cục bộ bằng dữ liệu tổng hợp, không
phải kết quả Kaggle.

Các file:

- `audit-report.md`: báo cáo audit chi tiết.
- `execution-status.json`: số liệu máy đọc được từ 9 notebook.
- `kaggle-info.json`: metadata, trạng thái và checksum output Kaggle.
- `L21_V001-v2-result.zip`: output ZIP tải từ Kaggle, gồm log, summary và traceback.
- `local-smoke-test-output.txt`: lệnh, môi trường và kết quả smoke test CPU.
