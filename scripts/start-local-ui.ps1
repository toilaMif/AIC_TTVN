param(
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'
$runtimeRoot = Join-Path $projectRoot '.runtime'
$logRoot = Join-Path $runtimeRoot 'logs'
$pidFile = Join-Path $runtimeRoot 'aic-api.pid'
Set-Location $projectRoot

function Get-DotEnvValue {
    param(
        [string]$Name,
        [string]$Default
    )

    $line = Get-Content -LiteralPath $envFile | Where-Object {
        $_ -match "^\s*$([regex]::Escape($Name))\s*="
    } | Select-Object -Last 1
    if (-not $line) { return $Default }
    $value = (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
    if ($value) { return $value }
    return $Default
}

if (-not (Test-Path -LiteralPath $envFile)) {
    throw 'Chua co file .env. Hay chay scripts/setup-local.ps1 truoc.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Khong tim thay Docker trong PATH.'
}

$backendHost = Get-DotEnvValue 'BACKEND_HOST' '127.0.0.1'
$backendPort = [int](Get-DotEnvValue 'BACKEND_PORT' '8000')
$healthUrl = "http://127.0.0.1:$backendPort/health"
$uiUrl = "http://127.0.0.1:$backendPort/ui/"

& docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'Khong the khoi dong Docker Compose.' }

$apiReady = $false
try {
    $response = Invoke-RestMethod $healthUrl -TimeoutSec 2
    $apiReady = $response.status -eq 'ok'
} catch {
    $apiReady = $false
}

if (-not $apiReady) {
    $listener = Get-NetTCPConnection -LocalPort $backendPort -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port $backendPort dang duoc mot chuong trinh khac su dung. Hay doi BACKEND_PORT trong .env."
    }

    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Chua co moi truong Python .venv. Hay chay scripts/setup-local.ps1 truoc.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'apps\frontend\dist'))) {
        throw 'Frontend chua duoc build. Hay chay scripts/setup-local.ps1 truoc.'
    }

    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @('-m', 'uvicorn', 'apps.api.main:app', '--host', $backendHost, '--port', "$backendPort") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot 'api.out.log') `
        -RedirectStandardError (Join-Path $logRoot 'api.error.log') `
        -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-RestMethod $healthUrl -TimeoutSec 2
            if ($response.status -eq 'ok') {
                $apiReady = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
}

if (-not $apiReady) {
    $errorLog = Join-Path $logRoot 'api.error.log'
    if (Test-Path -LiteralPath $errorLog) {
        Get-Content -LiteralPath $errorLog -Tail 20
    }
    throw "API khong san sang tai $healthUrl."
}

if (-not $NoBrowser) {
    Start-Process $uiUrl
}
Write-Host "AIC-TTVN da san sang tai $uiUrl" -ForegroundColor Green
