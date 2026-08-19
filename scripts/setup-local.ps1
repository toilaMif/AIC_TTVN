$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'
$envExample = Join-Path $projectRoot '.env.example'
Set-Location $projectRoot

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay '$Name' trong PATH. Xem phan Yeu cau trong README.md."
    }
}

function Get-DotEnvValue {
    param(
        [string]$Name,
        [string]$Default = ''
    )

    $line = Get-Content -LiteralPath $envFile | Where-Object {
        $_ -match "^\s*$([regex]::Escape($Name))\s*="
    } | Select-Object -Last 1
    if (-not $line) {
        return $Default
    }
    $value = (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
    if ($value) { return $value }
    return $Default
}

function Resolve-ProjectPath {
    param([string]$Value)

    $expanded = [Environment]::ExpandEnvironmentVariables($Value)
    if ([System.IO.Path]::IsPathRooted($expanded)) {
        return [System.IO.Path]::GetFullPath($expanded)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $expanded))
}

foreach ($command in @('uv', 'node', 'npm', 'docker')) {
    Assert-Command $command
}

& docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Engine chua chay. Hay mo Docker Desktop roi chay lai script.'
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
    Write-Host 'Da tao .env tu .env.example.' -ForegroundColor Green
} else {
    Write-Host 'Giu nguyen file .env hien co.' -ForegroundColor Yellow
}

$directoryKeys = @(
    'AIC_POSTGRES_ROOT',
    'AIC_MINIO_ROOT',
    'AIC_MILVUS_ROOT',
    'AIC_ETCD_ROOT',
    'AIC_KAGGLE_ARTIFACT_ROOT'
)
foreach ($key in $directoryKeys) {
    $value = Get-DotEnvValue $key
    if (-not $value) {
        throw "Thieu $key trong file .env."
    }
    $path = Resolve-ProjectPath $value
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot '.runtime\logs') | Out-Null

Write-Host 'Dang cai dependency Python...' -ForegroundColor Cyan
& uv sync --frozen
if ($LASTEXITCODE -ne 0) { throw 'uv sync that bai.' }

Write-Host 'Dang cai va build frontend...' -ForegroundColor Cyan
Push-Location (Join-Path $projectRoot 'apps\frontend')
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci that bai.' }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw 'npm run build that bai.' }
} finally {
    Pop-Location
}

Write-Host 'Dang khoi dong PostgreSQL, MinIO, etcd va Milvus...' -ForegroundColor Cyan
& docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Cau hinh Docker Compose khong hop le.' }
& docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'Khong the khoi dong Docker Compose.' }

$postgresUser = Get-DotEnvValue 'POSTGRES_USER' 'aic_app'
$postgresDb = Get-DotEnvValue 'POSTGRES_DB' 'aic_ttvn'
$postgresReady = $false
for ($attempt = 0; $attempt -lt 45; $attempt++) {
    & docker compose exec -T postgres pg_isready -U $postgresUser -d $postgresDb *> $null
    if ($LASTEXITCODE -eq 0) {
        $postgresReady = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $postgresReady) {
    throw 'PostgreSQL chua san sang sau 45 giay. Chay docker compose ps de kiem tra.'
}

Write-Host 'Dang cap nhat database schema...' -ForegroundColor Cyan
& uv run alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw 'Alembic migration that bai.' }

Write-Host ''
Write-Host 'Thiet lap local hoan tat.' -ForegroundColor Green
Write-Host 'Chay ung dung bang:'
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts/start-local-ui.ps1'
