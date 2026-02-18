#!/bin/bash

# Usage:
#   sbatch --array=1-$(awk 'NF && $1 !~ /^#/' examples/compare_params.txt | wc -l) \
#     examples/slurm_compare_model_stats_array.sh

#SBATCH --job-name=inflow_stats
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brooke.emison@yale.edu
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=logs/inflow_stats_%A_%a.out

set -euo pipefail

pwd
hostname
date

module reset
module load miniconda
conda activate aging

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs outputs outputs/model_plots

PARAM_FILE="examples/compare_params.txt"
if [[ ! -f "$PARAM_FILE" ]]; then
  echo "Missing parameter file: $PARAM_FILE"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
LINE="$(awk 'NF && $1 !~ /^#/' "$PARAM_FILE" | sed -n "${TASK_ID}p")"
if [[ -z "${LINE}" ]]; then
  echo "No parameter line found for array index ${TASK_ID} in ${PARAM_FILE}"
  exit 1
fi

# Columns:
#   model_path n_samples tf_burn tf_save_interval tf_n_chains tf_batch_size corr_plot_max_pairs
read -r MODEL_PATH N_SAMPLES TF_BURN TF_SAVE TF_CHAINS TF_BATCH CORR_MAX <<< "$LINE"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file does not exist: $MODEL_PATH"
  exit 1
fi

MODEL_BASENAME="$(basename "$MODEL_PATH" .pt)"
OUT_CSV="outputs/model_stats_${MODEL_BASENAME}.csv"
PLOT_DIR="outputs/model_plots"
CHECKPOINT_PATH="outputs/model_stats_${MODEL_BASENAME}.checkpoint.jsonl"

echo "Running task ${TASK_ID}: model=${MODEL_PATH} n_samples=${N_SAMPLES} burn=${TF_BURN} save=${TF_SAVE} chains=${TF_CHAINS} batch=${TF_BATCH}"

python inflow/compare_model_stats.py \
  --model-paths "$MODEL_PATH" \
  --data-dir data \
  --out-csv "$OUT_CSV" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --n-samples "$N_SAMPLES" \
  --tf-burn "$TF_BURN" \
  --tf-save-interval "$TF_SAVE" \
  --tf-n-chains "$TF_CHAINS" \
  --tf-batch-size "$TF_BATCH" \
  --device cuda \
  --plot-dir "$PLOT_DIR" \
  --corr-plot-max-pairs "$CORR_MAX" \
  --skip-existing-plots \
  --verbose

date
