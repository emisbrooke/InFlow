#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs

python inflow/compare_model_stats.py \
  --models-dir outputs/models \
  --data-dir data \
  --out-csv outputs/model_stats.csv \
  --n-samples 0 \
  --tf-burn 10000 \
  --tf-save-interval 1000 \
  --tf-batch-size 1000 \
  --tf-n-chains 256 \
  --device cuda
