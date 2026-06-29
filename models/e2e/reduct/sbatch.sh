#!/bin/bash

# ==============================================================================
# SLURM SETUP
# ==============================================================================

#SBATCH --job-name=E2E-Reduct
#SBATCH --output=log/h_%j.log
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p log

echo "Working directory set to: $(pwd)"

source .venv/bin/activate

echo ".venv activated"

export PYTHONPATH="$(pwd):$PYTHONPATH"

# ==============================================================================
# CONFIGURATION & JOB EXECUTION
# ==============================================================================

CONFIG_FILE="./models/e2e/reduct/config.yaml"

echo "Training model using configuration: ${CONFIG_FILE}"

python -m src.train.train --config "${CONFIG_FILE}" --run-name reduct

echo "Job completed successfully."