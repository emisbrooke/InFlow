#!/bin/bash

# Usage:
#   sbatch --array=1-$(awk 'NF && $1 !~ /^#/' examples/mi_swap_params.txt | wc -l) \
#     examples/slurm_run_mi_swaps_array.sh

#SBATCH --job-name=inflow_mi_swap
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brooke.emison@yale.edu
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inflow_mi_swap_%A_%a.out

set -euo pipefail

pwd
hostname
date

module reset
module load miniconda
conda activate aging

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs outputs/mi_swaps

PARAM_FILE="examples/mi_swap_params.txt"
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
# tissue lambda iters n_tf_samples int_burn int_save mc_batch_size ages_csv
# ages_csv format: 3,24 or 3,18,24
read -r TISSUE LAM ITERS N_SAMPLES INT_BURN INT_SAVE BATCH_SIZE AGES_CSV <<< "$LINE"

AGES_ARGS=()
IFS=',' read -r -a AGE_ARR <<< "$AGES_CSV"
for a in "${AGE_ARR[@]}"; do
  AGES_ARGS+=("$a")
done

echo "Running task ${TASK_ID}: tissue=${TISSUE} lambda=${LAM} ages=${AGES_CSV}"

python inflow/run_mi_swaps.py \
  --tissue "${TISSUE}" \
  --ages "${AGES_ARGS[@]}" \
  --lambda "${LAM}" \
  --iters "${ITERS}" \
  --n-tf-samples "${N_SAMPLES}" \
  --int-burn "${INT_BURN}" \
  --int-save "${INT_SAVE}" \
  --mc-batch-size "${BATCH_SIZE}" \
  --device cuda \
  --out-dir outputs/mi_swaps

date
