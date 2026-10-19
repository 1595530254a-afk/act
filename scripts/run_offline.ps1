param([ValidateSet('cuda', 'cpu')][string]$Device = 'cuda')

. (Join-Path $PSScriptRoot 'act_env.ps1')
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
& (Join-Path $ActRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'offline_inference.py') --device $Device
exit $LASTEXITCODE
