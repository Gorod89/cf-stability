# Reproduces the M1 numbers from the raw data in data/raw: events, splits, statistics,
# IDM calibration (full and with fixed v0, b), persistence baseline.
#   powershell -ExecutionPolicy Bypass -File scripts\run_m1.ps1 [-SkipExtraction] [-Sets openacc,waymo]
param(
    [switch]$SkipExtraction,
    [string[]]$Sets = @("openacc", "follownet_highd", "follownet_ngsim_i80", "follownet_waymo", "waymo", "ngsim_i80", "ngsim_us101")
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$python = Join-Path $repo ".venv\Scripts\python.exe"
# a parameter given as "a,b" on the command line of powershell.exe -File arrives as one string
$Sets = $Sets | ForEach-Object { $_ -split "," } | Where-Object { $_ }

function Invoke-Step([string[]]$Arguments) {
    Write-Output ("`n>>> python " + ($Arguments -join " "))
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "step failed: $($Arguments -join ' ')" }
}

if (-not $SkipExtraction) {
    foreach ($name in $Sets) { Invoke-Step "scripts\extract_events.py", "data=$name" }
}
Invoke-Step (@("scripts\describe_events.py") + $Sets)
foreach ($name in $Sets) {
    Invoke-Step "scripts\calibrate_idm.py", "dataset=$name"
    Invoke-Step "scripts\persistence_baseline.py", "dataset=$name"
}
# heterogeneous IDM of the corridor (M5): v0 and b at their global values, T, s0, a per event
foreach ($name in $Sets | Where-Object { $_ -like "ngsim_*" }) {
    Invoke-Step "scripts\calibrate_idm.py", "dataset=$name", "variant=fixed_v0_b", "global_fit=false", "calibration.fixed={v0: global, b: global}"
}
