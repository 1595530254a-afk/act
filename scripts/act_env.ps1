$ActRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $ActRoot 'cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME 'datasets'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TORCH_HOME = Join-Path $ActRoot 'cache\torch'
$env:XDG_CACHE_HOME = Join-Path $ActRoot 'cache'
$env:UV_CACHE_DIR = Join-Path $ActRoot 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ActRoot '.python'
$env:UV_PYTHON_BIN_DIR = Join-Path $ActRoot 'tools\python-bin'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_LINK_MODE = 'copy'
$env:UV_HTTP_TIMEOUT = '120'
$env:TEMP = Join-Path $ActRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PIP_CACHE_DIR = Join-Path $ActRoot 'cache\pip'

foreach ($ActDirectory in @($env:TEMP, $env:UV_CACHE_DIR, $env:PIP_CACHE_DIR)) {
    if (-not (Test-Path -LiteralPath $ActDirectory)) {
        New-Item -ItemType Directory -Path $ActDirectory | Out-Null
    }
}
