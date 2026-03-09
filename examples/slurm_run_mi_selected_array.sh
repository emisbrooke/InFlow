#!/bin/bash

# Usage:
#   sbatch --array=1-$(awk 'NF && $1 !~ /^#/' examples/mi_selected_params.txt | wc -l) \
#     examples/slurm_run_mi_selected_array.sh
#
#   sbatch --array=1-$(awk 'NF && $1 !~ /^#/' examples/mi_selected_params.txt | wc -l) \
#     examples/slurm_run_mi_selected_array.sh examples/mi_selected_params.txt

#SBATCH --job-name=mi_selected
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brooke.emison@yale.edu
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mi_selected_%A_%a.out

set -eo pipefail

pwd
hostname
date

module reset
module load miniconda
set +u
conda activate aging
set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs outputs/mi_swaps

# Param file precedence: first CLI argument > PARAM_FILE env var > default
PARAM_FILE="${1:-${PARAM_FILE:-examples/mi_selected_params.txt}}"
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
# tissue select_age source_ages_csv gene_type target_ages_csv iters n_tf_samples int_burn int_save mc_batch_size [seed]
read -r TISSUE SELECT_AGE SOURCE_AGES_CSV GENE_TYPE TARGET_AGES_CSV ITERS N_SAMPLES INT_BURN INT_SAVE BATCH_SIZE SEED <<< "$LINE"

if [[ -z "${SEED:-}" ]]; then
  SEED=0
fi

TARGET_AGES_ARGS=()
IFS=',' read -r -a TARGET_AGES_ARR <<< "$TARGET_AGES_CSV"
for a in "${TARGET_AGES_ARR[@]}"; do
  TARGET_AGES_ARGS+=("$a")
done

SOURCE_AGES_ARGS=()
IFS=',' read -r -a SOURCE_AGES_ARR <<< "$SOURCE_AGES_CSV"
for a in "${SOURCE_AGES_ARR[@]}"; do
  SOURCE_AGES_ARGS+=("$a")
done

echo "Running task ${TASK_ID}: tissue=${TISSUE} select_age=${SELECT_AGE} source_ages=${SOURCE_AGES_CSV} gene_type=${GENE_TYPE} targets=${TARGET_AGES_CSV} seed=${SEED}"

AGE_SPECIFIC_FLAG=()
if [[ "${USE_AGE_SPECIFIC_HPARAMS:-1}" == "1" ]]; then
  AGE_SPECIFIC_FLAG=(--use-age-specific-hparams)
fi

python inflow/run_mi_selected_from_best.py \
  --best-csv outputs/model_plots/summary/best_lambda_by_group.csv \
  --age "${SELECT_AGE}" \
  --source-ages "${SOURCE_AGES_ARGS[@]}" \
  --gene-type "${GENE_TYPE}" \
  "${AGE_SPECIFIC_FLAG[@]}" \
  --tissues "${TISSUE}" \
  --target-ages "${TARGET_AGES_ARGS[@]}" \
  --iters "${ITERS}" \
  --n-tf-samples "${N_SAMPLES}" \
  --int-burn "${INT_BURN}" \
  --int-save "${INT_SAVE}" \
  --mc-batch-size "${BATCH_SIZE}" \
  --device cuda \
  --seed "${SEED}" \
  --out-dir outputs/mi_swaps \
  --verbose

date
