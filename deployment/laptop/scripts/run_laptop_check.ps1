param([ValidateSet('cuda', 'cpu')][string]$Device = 'cuda')

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'verify_bundle.ps1')
# Runtime uses only local payloads; it does not require the installation USB drive.
. (Join-Path $PSScriptRoot 'laptop_env.ps1')
$ActPython = Join-Path $ActBundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ActPython)) {
    throw 'Local environment is missing. Run setup_laptop.ps1 first.'
}
& $ActPython (Join-Path $PSScriptRoot 'check_inference.py') --device $Device
if ($LASTEXITCODE -ne 0) { throw "Offline inference failed on $Device." }
