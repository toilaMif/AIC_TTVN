# Triển khai lên vast.ai cho cả đội dùng chung

Tài liệu này hướng dẫn đưa AIC-TTVN (đang chạy local trên máy bạn) lên một
instance thuê trên [vast.ai](https://vast.ai) để đồng đội truy cập qua
internet bằng chung một link, không cần mỗi người tự dựng hạ tầng.

## Lưu ý trước khi thuê máy

- **vast.ai là marketplace thuê GPU theo giờ**, không phải VPS truyền thống
  như DigitalOcean/Vultr. Máy có thể bị **thu hồi bất kỳ lúc nào** nếu bạn
  chọn loại rẻ hơn ("Interruptible"/"Spot"). Chọn instance loại **On-Demand**,
  ưu tiên máy có **Max Duration** dài (VD: 3 tháng) để không bị giật lại giữa
  mùa thi.
- Hệ thống không cần GPU để chạy (Postgres/Milvus/MinIO/API đều chạy CPU),
  nhưng vast.ai luôn bán kèm GPU — chấp nhận trả thêm cho phần GPU không
  dùng tới. Máy đã chọn: 2x RTX 3090, 48 CPU / 80 GB RAM, NVMe, ~$0.33/giờ,
  Max Duration 3 tháng, Reliability 96.7% — dư dùng thoải mái.
- **Ổ đĩa**: chọn Disk/Container Size **≥ 60 GB** — dữ liệu hiện có đã
  ~11 GB (`D:\AIC_TTVN_DATA`), cộng thêm Docker image của Postgres/Milvus/
  MinIO và khoảng trống cho dữ liệu tăng thêm về sau. Đã chọn 130 GB.
- **Chọn đúng template — "Ubuntu 22.04 VM"** (tìm bằng ô search template,
  gõ "ubuntu"; lọc thêm tag **VM**). Đây là **VM thật** (root đầy đủ), khác
  với hầu hết template khác trên vast.ai (HuggingFace TGI, RAPIDS, Hashcat...)
  vốn chỉ là 1 container Docker chạy sẵn 1 app cụ thể — dùng nhầm loại đó sẽ
  vướng vấn đề Docker-lồng-Docker khi chạy `docker compose`.
- **VM boot rất chậm**: máy cấu hình CPU/RAM lớn có thể mất **15-30+ phút**
  mới xong (vast.ai tự cảnh báo điều này). Trong lúc "Starting...", SSH sẽ
  báo `Connection closed by remote host` (kết nối mạng được nhưng sshd chưa
  chạy) hoặc `Connection refused` — không phải lỗi, **cứ đợi rồi thử lại**.
- Không cần đăng nhập, chỉ chia sẻ link nội bộ — bất kỳ ai có link
  `http://<ip>:<port>` đều dùng được. Chấp nhận được với đội thi nhỏ, tin
  tưởng nhau, nhưng **đổi mật khẩu mặc định** trong `.env` trước khi chạy
  (`POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD` — mặc định đang là mật khẩu
  demo `aic_local_postgres_2026` / `aic_local_minio_2026`).

## Bước 1 — Tạo instance trên vast.ai

1. Đổi template thành **"Ubuntu 22.04 VM"** (không dùng template mặc định
   được gợi ý sẵn).
2. Vào **Search**, lọc Disk ≥ 60GB, chọn máy On-Demand phù hợp, bấm **RENT**.
3. Ở màn cấu hình instance (trước khi Create): mục **Ports**, thêm dòng
   port **8000/TCP** (cổng API) bên cạnh port hệ thống vast.ai tự sinh —
   đây chính là port teammate sẽ dùng để truy cập, không cần SSH tunnel.
4. Bấm **Create & Use**.
5. Vào tab **Instances**, đợi trạng thái chuyển từ *Starting...* sang
   *Running* (xem lưu ý ở trên — có thể mất 15-30+ phút).

## Bước 2 — Lấy thông tin kết nối SSH

Bấm icon 🔑 (Manage SSH Keys) hoặc **Connect** trên instance để lấy thông
tin dạng khối cấu hình SSH:

```
Host vast
    HostName <ip>
    User root
    Port <port>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    LocalForward 8080 localhost:8080
```

Lưu khối này vào cuối file `C:\Users\<bạn>\.ssh\config` (đặt tên `Host` khác
nhau nếu thuê nhiều máy, VD `vast`, `vast2`...) rồi thêm dòng
`LocalForward 8000 localhost:8000` để tiện tự kiểm tra UI qua
`http://localhost:8000/ui/` ngay trên máy Windows của bạn mà không cần biết
IP/port public — **tunnel này chỉ có tác dụng cho máy bạn**, không phải cách
đồng đội truy cập (đồng đội dùng thẳng IP:port public đã mở ở Bước 1.3).

Kết nối: `ssh <tên Host>` (VD `ssh vast5`).

Nếu báo `Connection closed by remote host` hoặc `Connection refused`: máy
còn đang boot, **đợi thêm rồi thử lại**, không phải lỗi cấu hình.

## Bước 3 — Cài công cụ trên máy thuê

```bash
apt-get update && apt-get install -y git curl build-essential tmux
curl -fsSL https://get.docker.com | sh
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs

docker run hello-world   # PHẢI ra "Hello from Docker!" mới đi tiếp được
```

## Bước 4 — Lấy code và cấu hình `.env`

```bash
git clone https://github.com/toilaMif/AIC_TTVN.git
cd AIC_TTVN
cp .env.example .env
nano .env
```

Sửa `.env`:

```dotenv
BACKEND_HOST=0.0.0.0

AIC_DATA_ROOT=/workspace/aic-data
AIC_POSTGRES_ROOT=/workspace/aic-data/postgres
AIC_MINIO_ROOT=/workspace/aic-data/minio
AIC_MILVUS_ROOT=/workspace/aic-data/milvus
AIC_ETCD_ROOT=/workspace/aic-data/etcd
AIC_KAGGLE_ARTIFACT_ROOT=/workspace/aic-data/artifacts/kaggle
AIC_KAGGLE_BATCH=L21_a

POSTGRES_PASSWORD=<mật khẩu mới, không dùng mật khẩu demo>
MINIO_ROOT_PASSWORD=<mật khẩu mới, không dùng mật khẩu demo>
```

## Bước 5 — Copy dữ liệu hiện có (~11 GB) lên máy thuê

Chạy **từ máy Windows của bạn** (không phải trên VPS), sau khi đã dừng
container local (`docker compose down`) để dữ liệu không bị ghi đè giữa
chừng. Thời gian tuỳ tốc độ **upload** mạng nhà bạn (bên vast.ai băng thông
rất cao, không phải điểm nghẽn) — ước tính 15 phút (100 Mbps) tới vài giờ
(10 Mbps).

```powershell
scp -P <port> -r "D:\AIC_TTVN_DATA" root@<ip>:/workspace/aic-data
```

Vì `scp` không tự resume khi rớt mạng giữa chừng với khối lượng lớn thế
này, khuyến nghị dùng **WinSCP** (giao diện, tự resume) hoặc `rsync` qua
WSL thay vì `scp` một lệnh chạy suốt:

```bash
rsync -avP -e "ssh -p <port>" "/mnt/d/AIC_TTVN_DATA/" root@<ip>:/workspace/aic-data/
```

## Bước 6 — Dựng hạ tầng và ứng dụng (trên máy thuê)

```bash
uv sync --frozen
npm --prefix apps/frontend ci
npm --prefix apps/frontend run build

docker compose up -d
docker compose ps          # đợi tất cả healthy trước khi migrate

uv run alembic upgrade head
```

## Bước 7 — Chạy API và lấy link cho team

```bash
bash scripts/start-remote.sh
```

Script này chạy `uvicorn` ở nền (bản Linux của `scripts/start-local-ui.ps1`),
ghi log vào `.runtime/logs/` và PID vào `.runtime/aic-api.pid`.

Gửi cho đồng đội: `http://<public-ip>:<public-port>/ui/` (IP:port public đã
mở ở Bước 1.3 — **không phải** cổng SSH LocalForward ở Bước 2).

Kiểm tra nhanh: `curl http://127.0.0.1:8000/health` (chạy trên máy thuê).

## Dừng / khởi động lại

```bash
kill "$(cat .runtime/aic-api.pid)"   # dừng API
bash scripts/start-remote.sh          # chạy lại API
docker compose down                   # dừng hạ tầng (không xoá dữ liệu)
docker compose up -d                  # chạy lại hạ tầng
```

## Việc nên làm thêm nếu dùng lâu dài (không bắt buộc)

- Dùng `systemd` thay vì chạy tay để API tự khởi động lại khi máy reboot.
- Backup định kỳ `/workspace/aic-data` về máy khác — dữ liệu trên vast.ai
  mất khi **Terminate** instance (khác với **Stop**, vẫn giữ dữ liệu nhưng
  vẫn tính phí ổ đĩa).
- Nếu muốn có domain + HTTPS thay vì IP:port thô, thêm Nginx reverse proxy
  + Let's Encrypt phía trước cổng 8000 — không bắt buộc để dùng nội bộ.
