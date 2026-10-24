$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$simulationPython = Join-Path $projectRoot 'local\mujoco-env\Scripts\python.exe'
$serverScript = Join-Path $PSScriptRoot 'serve_report.py'
$reportUrl = 'http://127.0.0.1:8786/index.html'
$serverReady = $false
try {
    $response = Invoke-WebRequest -Uri $reportUrl -UseBasicParsing -TimeoutSec 2
    $serverReady = $response.Content -match 'B601-RS'
} catch { }
if (-not $serverReady) {
    $arguments = @('-u', ('"' + $serverScript + '"'))
    Start-Process -FilePath $simulationPython -ArgumentList $arguments -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $response = Invoke-WebRequest -Uri $reportUrl -UseBasicParsing -TimeoutSec 1
            if ($response.Content -match 'B601-RS') { $serverReady = $true; break }
        } catch { }
    }
}
if (-not $serverReady) { throw 'Report server did not start. Port 8786 may be occupied.' }
Start-Process $reportUrl
