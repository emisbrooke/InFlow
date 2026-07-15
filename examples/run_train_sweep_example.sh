#!/usr/bin/env bash
set -euo pipefail

# Example local sweep runner (non-Slurm).
# Update tissues/ages/lambdas to match your experiment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$REPO_ROOT/data"
OUT_DIR="$REPO_ROOT/outputs/models"

mkdir -p "$OUT_DIR"

# Paper-aligned pattern:
# 1) coarse lambda sweep (integers)
# 2) optional finer sweep around best lambda (e.g. +/- 0.2 by 0.1)

TISSUES=(brain liver)
AGES=(3 24)
GENE_TYPES=(TF TG)

# Coarse sweep
LAMBDAS=(0 1 2 3 4 5)

python inflow/train_sweep.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --tissues "${TISSUES[@]}" \
  --ages "${AGES[@]}" \
  --gene-types "${GENE_TYPES[@]}" \
  --lambdas "${LAMBDAS[@]}" \
  --steps 5000 \
  --optimizer adam \
  --alpha 0.01 \
  --beta1 0.9 \
  --beta2 0.999 \
  --eps 1e-8 \
  --n-restarts 10 \
  --save-dtype float16 \
  --log-int 500

echo "Coarse sweep complete: $OUT_DIR"

# Example fine sweep (uncomment and edit around your selected best lambda)
# FINE_LAMBDAS=(1.8 1.9 2.0 2.1 2.2)
# python inflow/train_sweep.py \
#   --data-dir "$DATA_DIR" \
#   --out-dir "$OUT_DIR" \
#   --tissues "${TISSUES[@]}" \
#   --ages "${AGES[@]}" \
#   --gene-types "${GENE_TYPES[@]}" \
#   --lambdas "${FINE_LAMBDAS[@]}" \
#   --steps 5000 \
#   --optimizer adam \
#   --alpha 0.01 \
#   --beta1 0.9 \
#   --beta2 0.999 \
#   --eps 1e-8 \
#   --n-restarts 10 \
#   --save-dtype float16 \
#   --log-int 500
