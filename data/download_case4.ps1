# Download only the documented case 4 dev data. Never print credentials or bodies.
$ErrorActionPreference = 'Stop'
$dataRoot = $PSScriptRoot
$rawRoot = Join-Path $dataRoot 'raw'
New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null
$apiRoot = 'https://hack-api.be-healthy.tech'
$tokenFile = Join-Path (Split-Path -Parent $dataRoot) 'api.token'
$teamToken = [System.IO.File]::ReadAllText($tokenFile).Trim()
if ([string]::IsNullOrWhiteSpace($teamToken)) { throw 'Team API token is empty' }
try {
    Invoke-WebRequest -Uri ($apiRoot + '/case/4/getData?dataset_code=case4-dev-v2') `
        -Headers @{ Authorization = 'Bearer ' + $teamToken; Accept = 'application/json' } `
        -UseBasicParsing -TimeoutSec 120 -MaximumRedirection 0 `
        -OutFile (Join-Path $rawRoot 'case4-dev-v2.json') | Out-Null
    Write-Output 'Case 4 dev bundle downloaded to local data/raw; no secrets printed.'
} catch {
    $httpStatus = $_.Exception.Response.StatusCode
    if ($null -ne $httpStatus) { throw ('Download failed: HTTP ' + [int]$httpStatus + '; details omitted') }
    throw 'Download failed: network or file error; details omitted'
} finally {
    $teamToken = $null
}
