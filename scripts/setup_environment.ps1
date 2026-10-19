param([string]$BootstrapPython = 'D:\Python\python.exe')

. (Join-Path $PSScriptRoot 'act_env.ps1')
$ActUv = Join-Path $ActRoot 'tools\uv\bin\uv.exe'
$ActPython = Join-Path $ActRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $ActUv)) {
    & $BootstrapPython -m pip install --target (Join-Path $ActRoot 'tools\uv') --no-warn-script-location 'uv==0.13.1'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $ActUv python install 3.12.15
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path -LiteralPath $ActPython)) {
    & $ActUv venv --python 3.12.15 (Join-Path $ActRoot '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$ActRequirements = Join-Path $ActRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $ActRequirements)) {
    $ActRequirements = Join-Path $ActRoot 'requirements.in'
}
& $ActUv pip install --python $ActPython -r $ActRequirements --index 'https://download.pytorch.org/whl/cu130' --default-index 'https://pypi.org/simple' --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $ActUv pip check --python $ActPython
exit $LASTEXITCODE
