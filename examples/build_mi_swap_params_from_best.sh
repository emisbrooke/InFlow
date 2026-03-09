#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python inflow/build_mi_swap_params_from_best.py \
  --best-csv outputs/model_plots/summary/best_lambda_by_group.csv \
  --out-file examples/mi_swap_params.txt \
  --age 3 \
  --gene-type TF \
  --ages-csv 3,24 \
  --iters 10 \
  --n-tf-samples 30000 \
  --int-burn 200000 \
  --int-save 500 \
  --mc-batch-size 500
