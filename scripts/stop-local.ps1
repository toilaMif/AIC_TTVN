$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot '.runtime\aic-api.pid'
Set-Location $projectRoot

if (Test-Path -LiteralPath $pidFile) {
    $apiPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($apiPid -match '^\d+$') {
        $process = Get-Process -Id ([int]$apiPid) -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $process.Id
            Write-Host "Da dung FastAPI (PID $apiPid)."
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
} else {
    Write-Host 'Khong tim thay PID cua FastAPI do script quan ly.' -ForegroundColor Yellow
}

& docker compose down
if ($LASTEXITCODE -ne 0) { throw 'Khong the dung Docker Compose.' }

Write-Host 'Da dung he thong. Du lieu local van duoc giu nguyen.' -ForegroundColor Green
