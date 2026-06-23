#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
#SBATCH --job-name=PolDQN
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=uts26_tornator
#SBATCH --gres=gpu:1

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module load python/3.11.7
module load cuda/12.6

source $SCRATCH/E2E-JEPA/.venv/bin/activate

mkdir -p log

CONFIG_FILE="./configs/raw-policies/dqn-conv-2d-train.yaml"
python -m src.policy.policy --config "${CONFIG_FILE}" --train
