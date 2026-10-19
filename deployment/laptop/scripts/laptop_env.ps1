param([string]$BulkRoot = '')

$ErrorActionPreference = 'Stop'
$ActBundleRoot = Split-Path -Parent $PSScriptRoot
$ActRuntimeRoot = Join-Path $ActBundleRoot 'local\runtime'
$ActCacheRoot = Join-Path $ActRuntimeRoot 'cache'
$ActTemporaryRoot = Join-Path $ActRuntimeRoot 'tmp'

if ($BulkRoot) {
    if ($BulkRoot -notmatch '^[A-Za-z]:[\\/]') {
        throw 'BulkRoot must be an absolute path on the selected USB drive.'
    }
    $ActBulkPath = [System.IO.Path]::GetFullPath($BulkRoot)
    if (-not (Test-Path -LiteralPath ([System.IO.Path]::GetPathRoot($ActBulkPath)))) {
        throw "Bulk storage drive is unavailable: $ActBulkPath"
    }
    $ActCacheRoot = Join-Path $ActBulkPath 'cache'
    $ActTemporaryRoot = Join-Path $ActBulkPath 'tmp'
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $ActBundleRoot '.python'
$env:UV_PYTHON_BIN_DIR = Join-Path $ActBundleRoot 'tools\python-bin'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_LINK_MODE = 'copy'
$env:UV_HTTP_TIMEOUT = '120'
$env:UV_CACHE_DIR = Join-Path $ActCacheRoot 'uv'
$env:PIP_CACHE_DIR = Join-Path $ActCacheRoot 'pip'
$env:HF_HOME = Join-Path $ActCacheRoot 'huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME 'datasets'
$env:TORCH_HOME = Join-Path $ActCacheRoot 'torch'
$env:XDG_CACHE_HOME = $ActCacheRoot
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TEMP = $ActTemporaryRoot
$env:TMP = $ActTemporaryRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

foreach ($ActDirectory in @($ActCacheRoot, $ActTemporaryRoot)) {
    if (-not (Test-Path -LiteralPath $ActDirectory)) {
        New-Item -ItemType Directory -Path $ActDirectory -Force | Out-Null
    }
}
