# M2 reference runs: every model on one fold of one event set, closed-loop evaluation.
#   powershell -ExecutionPolicy Bypass -File scripts\run_m2.ps1 [-Data follownet_highd] [-Fold 0] [-Seed 0]
# Runs that already have a metrics.json are skipped, so the script can be restarted.
param(
    [string]$Data = "follownet_highd",
    [int]$Fold = 0,
    [int]$Seed = 0,
    [string]$Experiment = "m2",
    [string[]]$Models = @("persistence", "idm", "ovm", "newell", "knn", "mlp", "gru", "lstm", "pidl", "perl", "residual_idm")
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$Models = $Models | ForEach-Object { $_ -split "," } | Where-Object { $_ }

foreach ($model in $Models) {
    $done = Join-Path $repo "runs\$Experiment\$Data\$model\driver_fold${Fold}_seed${Seed}\metrics.json"
    if (Test-Path $done) { Write-Output "`n>>> $model : present, skipped"; continue }
    Write-Output "`n>>> python scripts\train.py data=$Data model=$model fold=$Fold seed=$Seed experiment=$Experiment"
    & $python scripts\train.py "data=$Data" "model=$model" "fold=$Fold" "seed=$Seed" "experiment=$Experiment"
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $model (exit code $LASTEXITCODE)" }
}
