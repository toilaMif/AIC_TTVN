#!/usr/bin/env bash
# Chay API o nen tren VPS/instance thue (Linux). Ban Linux cua
# scripts/start-local-ui.ps1, dung cho hg dan trong DEPLOY.md.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

env_file="$project_root/.env"
runtime_root="$project_root/.runtime"
log_root="$runtime_root/logs"
pid_file="$runtime_root/aic-api.pid"

if [ ! -f "$env_file" ]; then
  echo "Chua co file .env. Hay `cp .env.example .env` va sua cau hinh truoc." >&2
  exit 1
fi

get_env() {
  local name="$1" default="$2"
  local value
  value="$(grep -E "^\s*${name}\s*=" "$env_file" | tail -n1 | cut -d'=' -f2- \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^['\"]//" -e "s/['\"]\$//")"
  echo "${value:-$default}"
}

backend_host="$(get_env BACKEND_HOST 0.0.0.0)"
backend_port="$(get_env BACKEND_PORT 8000)"
health_url="http://127.0.0.1:${backend_port}/health"

if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "API da dang chay (PID $(cat "$pid_file")). Dung 'kill \$(cat $pid_file)' truoc neu muon khoi dong lai."
  exit 0
fi

echo "Khoi dong ha tang (docker compose up -d)..."
docker compose up -d

mkdir -p "$log_root"

if [ ! -d "$project_root/apps/frontend/dist" ]; then
  echo "Frontend chua duoc build. Chay: npm --prefix apps/frontend ci && npm --prefix apps/frontend run build" >&2
  exit 1
fi

echo "Khoi dong API tai ${backend_host}:${backend_port}..."
nohup uv run uvicorn apps.api.main:app --host "$backend_host" --port "$backend_port" \
  > "$log_root/api.out.log" 2> "$log_root/api.error.log" &
echo $! > "$pid_file"

ready=false
for _ in $(seq 1 60); do
  if curl -sf "$health_url" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 0.5
done

if [ "$ready" != true ]; then
  echo "API khong san sang tai $health_url. Xem log:" >&2
  tail -n 20 "$log_root/api.error.log" >&2 || true
  exit 1
fi

echo "AIC-TTVN da san sang. Health: $health_url"
echo "Neu cong ${backend_port} da duoc forward cong khai tren vast.ai, gui link:"
echo "  http://<public-ip>:<public-port>/ui/"
