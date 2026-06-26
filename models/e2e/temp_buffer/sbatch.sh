#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
#SBATCH --job-name=E2E-Temp-Buffer
#SBATCH --output=log/e2e_%j.log
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

# Ensure log directory exists so SBATCH doesn't fail silently
mkdir -p log

echo "Working directory set to: $(pwd)"

# 1. Activate or Create the virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating .venv..."
    ~/Python-3.12.7/python -m venv .venv
    source .venv/bin/activate

    python -m ensurepip --upgrade --default-pip
    python -m pip install --quiet --upgrade pip

    # Install dependencies based on your repo structure
    echo "Installing requirements..."
    if [ -f "src/policy/requirements.txt" ]; then
        python -m pip install --quiet -r src/policy/requirements.txt
    fi
    if [ -f "src/game/requirements.txt" ]; then
        python -m pip install --quiet -r src/game/requirements.txt
    fi
else
    echo "Activating virtual environment from .venv..."
    source .venv/bin/activate
fi

# 2. Crucial: Set PYTHONPATH to the root directory
# This ensures imports like 'from src.policy.algorithms import *' resolve correctly
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ==============================================================================
# CONFIGURATION & JOB EXECUTION
# ==============================================================================

# Construct the config file path automatically
CONFIG_FILE="./models/e2e/temp_buffer/config.yaml"

# Safety check
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file ${CONFIG_FILE} not found!"
    exit 1
fi

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script matching the argparse configuration in policy.py
python -m src.train.train --config "${CONFIG_FILE}" --run-name temp_buffer

echo "Job completed successfully."
