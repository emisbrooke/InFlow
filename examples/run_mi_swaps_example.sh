#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python inflow/run_mi_swaps.py \
  --tissue brain \
  --ages 3 24 \
  --lambda 1.0 \
  --iters 10 \
  --n-tf-samples 30000 \
  --int-burn 200000 \
  --int-save 500 \
  --mc-batch-size 500 \
  --device cuda \
  --out-dir outputs/mi_swaps
