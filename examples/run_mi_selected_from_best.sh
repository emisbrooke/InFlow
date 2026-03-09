#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python inflow/run_mi_selected_from_best.py \
  --best-csv outputs/model_plots/summary/best_lambda_by_group.csv \
  --age 3 \
  --source-ages 3 24 \
  --gene-type TF \
  --use-age-specific-hparams \
  --target-ages 3 24 \
  --iters 10 \
  --n-tf-samples 30000 \
  --int-burn 200000 \
  --int-save 500 \
  --mc-batch-size 500 \
  --device cuda \
  --out-dir outputs/mi_swaps \
  --verbose
