$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host "[CHECK] $Label" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Kiem tra that bai: $Label"
    }
}

foreach ($command in @('git', 'uv', 'node', 'npm', 'docker')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay '$command' trong PATH."
    }
}

& git check-ignore --quiet .env
if ($LASTEXITCODE -ne 0) {
    throw 'File .env chua duoc .gitignore bao ve.'
}
if (@(& git ls-files -- .env).Count -gt 0) {
    throw 'File .env dang bi Git theo doi. Khong duoc push file nay.'
}

$candidateFiles = @(& git -c core.quotePath=false ls-files --cached --others --exclude-standard)
$largeFiles = @()
$forbiddenFiles = @()
$forbiddenExtensions = @(
    '.ckpt', '.faiss', '.h5', '.npy', '.npz', '.onnx', '.pkl', '.pt', '.pth',
    '.safetensors', '.zip'
)
foreach ($relativePath in $candidateFiles) {
    $fullPath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { continue }
    $item = Get-Item -LiteralPath $fullPath
    if ($item.Length -gt 95MB) {
        $largeFiles += "$relativePath ($([math]::Round($item.Length / 1MB, 1)) MB)"
    }
    if ($forbiddenExtensions -contains $item.Extension.ToLowerInvariant()) {
        $forbiddenFiles += $relativePath
    }
}
if ($largeFiles) {
    throw "Co file lon gan/vuot gioi han GitHub:`n$($largeFiles -join "`n")"
}
if ($forbiddenFiles) {
    throw "Co artifact khong nen commit:`n$($forbiddenFiles -join "`n")"
}

$textExtensions = @(
    '.env', '.example', '.ini', '.ipynb', '.js', '.json', '.jsx', '.md', '.ps1',
    '.py', '.sh', '.toml', '.ts', '.tsx', '.yaml', '.yml'
)
$secretPatterns = @(
    'AKIA[0-9A-Z]{16}',
    'gh[pousr]_[A-Za-z0-9]{30,}',
    'github_pat_[A-Za-z0-9_]{20,}',
    'sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    '-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'Authorization:\s*Bearer\s+(?!\$\{)[A-Za-z0-9._-]{20,}'
)
$secretMatches = @()
foreach ($relativePath in $candidateFiles) {
    $fullPath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { continue }
    if ($textExtensions -notcontains ([System.IO.Path]::GetExtension($fullPath).ToLowerInvariant())) {
        continue
    }
    $content = Get-Content -LiteralPath $fullPath -Raw -ErrorAction SilentlyContinue
    foreach ($pattern in $secretPatterns) {
        if ($content -match $pattern) {
            $secretMatches += $relativePath
            break
        }
    }
}
if ($secretMatches) {
    throw "Phat hien chuoi giong credential trong:`n$($secretMatches -join "`n")"
}

Invoke-Checked 'uv.lock dong bo voi pyproject.toml' { & uv lock --check }
Invoke-Checked 'Python compile' { & uv run python -m compileall -q apps retrieval alembic scripts }
Invoke-Checked 'Frontend build' { & npm --prefix apps/frontend run build }
Invoke-Checked 'Docker Compose dung .env.example' { & docker compose --env-file .env.example config --quiet }
Invoke-Checked 'Git whitespace (unstaged)' { & git diff --check }
Invoke-Checked 'Git whitespace (staged)' { & git diff --cached --check }

$changeCount = @(& git status --short).Count
Write-Host ''
Write-Host "Repository dat kiem tra co ban. Co $changeCount dong thay doi trong git status." -ForegroundColor Green
Write-Host 'Hay chay git status va xem ky danh sach file truoc khi commit/push.'
