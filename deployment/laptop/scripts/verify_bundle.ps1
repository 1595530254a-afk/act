$ErrorActionPreference = 'Stop'
$ActBundleRoot = Split-Path -Parent $PSScriptRoot
$ActManifest = Get-Content -LiteralPath (Join-Path $ActBundleRoot 'bundle_manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($ActEntry in $ActManifest.files) {
    $ActPath = Join-Path $ActBundleRoot $ActEntry.path
    if (-not (Test-Path -LiteralPath $ActPath -PathType Leaf)) {
        throw "Missing bundle file: $($ActEntry.path)"
    }
    if ((Get-Item -LiteralPath $ActPath).Length -ne $ActEntry.bytes) {
        throw "Bundle file size mismatch: $($ActEntry.path)"
    }
    if ((Get-FileHash -LiteralPath $ActPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ActEntry.sha256) {
        throw "Bundle SHA256 mismatch: $($ActEntry.path)"
    }
}
Write-Output "Bundle verified: $($ActManifest.bundle_id), $($ActManifest.files.Count) files."
