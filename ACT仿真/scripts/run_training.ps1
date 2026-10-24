param(
    [ValidateRange(1, 100000)][int]$Steps = 100,
    [ValidateRange(1, 64)][int]$BatchSize = 2
)

. (Join-Path $PSScriptRoot 'act_env.ps1')
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
& (Join-Path $ActRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'short_training.py') --steps $Steps --batch-size $BatchSize
exit $LASTEXITCODE
