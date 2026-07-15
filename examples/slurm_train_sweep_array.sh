#!/bin/bash

# Usage:
#   sbatch --array=1-$(awk 'NF && $1 !~ /^#/' examples/train_params.txt | wc -l) \
#     examples/slurm_train_sweep_array.sh

#SBATCH --job-name=inflow_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brooke.emison@yale.edu
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=logs/inflow_train_%A_%a.out

set -euo pipefail

pwd
hostname
date

module reset
module load miniconda
conda activate aging

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs outputs/models

PARAM_FILE="examples/train_params.txt"
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
#   tissue age gene_type steps n_restarts lambda1 [lambda2 lambda3 ...]
read -r -a FIELDS <<< "$LINE"
if [[ "${#FIELDS[@]}" -lt 6 ]]; then
  echo "Expected at least 6 fields: tissue age gene_type steps n_restarts lambda1 [...], got: $LINE"
  exit 1
fi

TISSUE="${FIELDS[0]}"
AGE="${FIELDS[1]}"
GENE_TYPE="${FIELDS[2]}"
STEPS="${FIELDS[3]}"
N_RESTARTS="${FIELDS[4]}"
LAMBDAS=("${FIELDS[@]:5}")

echo "Running task ${TASK_ID}: tissue=${TISSUE} age=${AGE} gene_type=${GENE_TYPE} steps=${STEPS} n_restarts=${N_RESTARTS} lambdas=${LAMBDAS[*]}"

python inflow/train_sweep.py \
  --data-dir data \
  --out-dir outputs/models \
  --tissues "${TISSUE}" \
  --ages "${AGE}" \
  --gene-types "${GENE_TYPE}" \
  --lambdas "${LAMBDAS[@]}" \
  --steps "${STEPS}" \
  --optimizer adam \
  --alpha 0.01 \
  --beta1 0.9 \
  --beta2 0.999 \
  --eps 1e-8 \
  --n-restarts "${N_RESTARTS}" \
  --save-dtype float16

date
