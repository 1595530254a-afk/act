$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $taskRoot 'local\mujoco-env\Scripts\python.exe'
$model = Join-Path $taskRoot 'runs\mujoco_iteration_20261023\adapt_flat40_v1\checkpoints\step_000300\pretrained_model'
$weights = Join-Path $model 'model.safetensors'
$expected = 'e82c8794497ab288c6c58fd16008b78770eddb39bc5f5faab3143b1a530c6c76'
if ((Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Selected simulation checkpoint has changed.'
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$output = Join-Path $taskRoot "runs\mujoco_validation_20261023\fixed_flat40_20261023\replay_$stamp"
& $python (Join-Path $PSScriptRoot 'run_task.py') `
    --model-dir $model --flat-brick --brick-edge-mm 40 --action-steps 100 --duration 15 `
    --wrist-camera-fit (Join-Path $taskRoot 'runs\mujoco_iteration_20261023\camera_fit\camera_fit.json') `
    --output-dir $output
if ($LASTEXITCODE -ne 0) { throw "Simulation command exited $LASTEXITCODE" }
$result = Get-Content -LiteralPath (Join-Path $output 'custom\result.json') -Raw | ConvertFrom-Json
Write-Host "Output: $output"
Write-Host "Physical pick-and-place success: $($result.metrics.success)"
if ($result.status -ne 'completed' -or -not $result.metrics.success) {
    throw 'The command completed without passing the physical pick-and-place criteria. Inspect result.json.'
}
