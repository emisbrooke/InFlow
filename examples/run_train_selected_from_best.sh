#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python inflow/run_train_selected_from_best.py \
  --best-csv outputs/model_plots/summary/best_lambda_by_group.csv \
  --select-age 3 \
  --select-gene-type TF \
  --train-ages 3 24 \
  --train-gene-types TF TG \
  --steps 5000 \
  --n-restarts 10 \
  --out-dir outputs/models \
  --verbose
