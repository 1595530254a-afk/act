$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$simulationPython = Join-Path $projectRoot 'local\mujoco-env\Scripts\python.exe'
$previewScript = Join-Path $PSScriptRoot 'preview.py'
if (-not (Test-Path -LiteralPath $simulationPython)) {
    throw 'Simulation environment is missing. Read simulation/README.md first.'
}
Write-Host 'Pure simulation. Candidate coordinate mapping. No robot connection.'
Write-Host 'Space: play/pause; 1: demonstration; 2: ACT prediction; R: restart.'
& $simulationPython -u $previewScript --viewer
if ($LASTEXITCODE -ne 0) { throw "Simulation viewer exited with code $LASTEXITCODE" }
