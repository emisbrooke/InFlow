#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python inflow/plot_model_stats_summary.py \
  --csv outputs/model_stats.csv \
  --out-dir outputs/model_plots/summary \
  --dpi 160
