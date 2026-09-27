#!/usr/bin/env bash
# Linux / macOS / WSL. Run from the project root:  bash scripts/run_python_steps.sh
set -euo pipefail
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
cd python/src
python s01_inspect_data.py | tee ../../results/tables/data_inspection.txt
python s02_prepare_data.py
python s03_train_sklearn.py
cd ../..
echo "Done. Next: docs/03_experiment_protocol.md"
