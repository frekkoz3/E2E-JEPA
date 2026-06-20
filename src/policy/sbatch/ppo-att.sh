#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
#SBATCH --job-name=E2E_Policy
#SBATCH --output=log/att_%j.log
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
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
    echo "Virtual environment not found. Creating .venv using Python 3.9..."

    # Explicitly call python3.9 to guarantee the correct version
    python3.9 -m venv .venv
    source .venv/bin/activate

    python -m ensurepip --upgrade --default-pip
    python -m pip install --quiet --upgrade pip

    echo "Injecting and installing core requirements for Python 3.9..."
    # Create an inline requirements list for the packages imported in policy.py
    # Pinned to versions that are highly stable in Python 3.9
    cat <<EOF > inline_requirements.txt
torch>=1.13.0,<2.2.0
numpy>=1.21.0,<2.0.0
PyYAML>=6.0
EOF

    # Install the inline requirements
    python -m pip install --quiet -r inline_requirements.txt

    # Fallback: Also install local requirements if you add them later
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
CONFIG_FILE="src/policy/configs/ppo-att-train.yaml"

# Safety check
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file ${CONFIG_FILE} not found!"
    exit 1
fi

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script matching the argparse configuration in policy.py
python -u src/policy/policy.py --config "${CONFIG_FILE}" --train

echo "Job completed successfully."