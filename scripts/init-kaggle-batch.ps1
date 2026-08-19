param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9_-]*$')]
    [string]$Batch
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'

function Get-DotEnvValue {
    param([string]$Name)

    if (-not (Test-Path -LiteralPath $envFile)) {
        return $null
    }
    $line = Get-Content -LiteralPath $envFile | Where-Object {
        $_ -match "^\s*$([regex]::Escape($Name))\s*="
    } | Select-Object -Last 1
    if (-not $line) {
        return $null
    }
    return (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
}

$root = if ($env:AIC_KAGGLE_ARTIFACT_ROOT) {
    $env:AIC_KAGGLE_ARTIFACT_ROOT
} elseif (Get-DotEnvValue 'AIC_KAGGLE_ARTIFACT_ROOT') {
    Get-DotEnvValue 'AIC_KAGGLE_ARTIFACT_ROOT'
} else {
    Join-Path $projectRoot '.runtime\artifacts\kaggle'
}

if (-not [System.IO.Path]::IsPathRooted($root)) {
    $root = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $root))
}

$batchRoot = Join-Path $root $Batch.ToLowerInvariant()
$stages = @(
    '01-shot-keyframes',
    '02-visual-embeddings',
    '03-ocr',
    '04-asr',
    '05-object-detection',
    '_runtime'
)

foreach ($stage in $stages) {
    New-Item -ItemType Directory -Force -Path (Join-Path $batchRoot $stage) | Out-Null
}

Write-Host "Kaggle artifact batch ready: $batchRoot"
