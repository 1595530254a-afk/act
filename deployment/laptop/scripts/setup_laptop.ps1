param([Parameter(Mandatory = $true)][string]$BulkRoot)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'verify_bundle.ps1')
. (Join-Path $PSScriptRoot 'laptop_env.ps1') -BulkRoot $BulkRoot
$ActUv = Join-Path $ActBundleRoot 'tools\uv\uv.exe'
$ActPython = Join-Path $ActBundleRoot '.venv\Scripts\python.exe'
$ActUvVersion = & $ActUv --version
if ($LASTEXITCODE -ne 0 -or $ActUvVersion -notmatch '^uv 0\.13\.1(\s|$)') {
    throw "Unexpected uv executable/version: $ActUvVersion"
}
& $ActUv python install 3.12.15
if ($LASTEXITCODE -ne 0) { throw 'Managed Python installation failed.' }
if (-not (Test-Path -LiteralPath $ActPython)) {
    & $ActUv venv --python 3.12.15 (Join-Path $ActBundleRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $ActUv pip install --python $ActPython -r (Join-Path $ActBundleRoot 'requirements.lock.txt') --index 'https://download.pytorch.org/whl/cu130' --default-index 'https://pypi.org/simple' --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $ActUv pip check --python $ActPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
$ActLocal = Join-Path $ActBundleRoot 'local'
New-Item -ItemType Directory -Path $ActLocal -Force | Out-Null
@{
    bundle_root = $ActBundleRoot
    install_bulk_root = [System.IO.Path]::GetFullPath($BulkRoot)
    installed_at = [System.DateTimeOffset]::Now.ToString('o')
    runtime_storage = 'Local environment, model, fixtures and small runtime files; USB installation cache.'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ActLocal 'storage.local.json') -Encoding utf8
Write-Output 'Installation passed. Next run scripts\run_laptop_check.ps1 -Device cuda.'
