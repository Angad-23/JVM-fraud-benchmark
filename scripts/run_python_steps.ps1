# Run from anywhere:  .\scripts\run_python_steps.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -r python\requirements.txt
Set-Location python\src
python s01_inspect_data.py | Tee-Object -FilePath ..\..\results\tables\data_inspection.txt
python s02_prepare_data.py
python s03_train_sklearn.py
Set-Location $root
Write-Host "Done. Next: docs/03_experiment_protocol.md"