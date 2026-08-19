# Artifact storage

Repository chỉ giữ tài liệu và placeholder. Frame, embedding, transcript và
object detection output từ Kaggle không được commit lên GitHub. Cấu hình mặc
định cho máy mới lưu tại:

```text
.runtime/artifacts/kaggle/<batch>/
|-- 01-shot-keyframes\
|-- 02-visual-embeddings\
|-- 03-ocr\
|-- 04-asr\
|-- 05-object-detection\
`-- _runtime\
```

Nếu dữ liệu lớn, nên đặt ngoài repository, ví dụ:

```text
D:\AIC_TTVN_DATA\artifacts\kaggle\<batch>\
```

Quy ước:

- `<batch>` dung chu thuong, vi du `l21`, `l22`.
- Moi video la mot ZIP, ten file ket thuc bang `Lxx_Vxxx.zip`.
- `batch-summary.json` nam trong thu muc stage tuong ung.
- `_runtime` chi chua model/cache can cho viec tai lap pipeline.
- Khong giai nen hang loat vao repository.
- Khong sua ZIP sau khi import; output moi phai co pipeline version moi.

Bien moi truong:

```text
AIC_KAGGLE_ARTIFACT_ROOT=./.runtime/artifacts/kaggle
AIC_KAGGLE_BATCH=l21
```

Tao nhanh mot batch moi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/init-kaggle-batch.ps1 -Batch l22
```

Repository không chứa bản sao của các ZIP này. Không đặt artifact vào Git nếu
chúng chứa dữ liệu cuộc thi, dữ liệu lớn hoặc nội dung không được phép công bố.
