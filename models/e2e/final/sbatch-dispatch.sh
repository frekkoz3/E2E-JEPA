#!/bin/bash

epochs=$1

# 1. Submit the first job without dependencies
job_id=$(sbatch ./models/e2e/final/sbatch.sh | cut -d ' ' -f 4)
echo "Submitted initial job (Epoch 1): $job_id"

# 2. Loop for the remaining epochs
for (( i=2; i<=epochs; i++ )); do
    job_id=$(sbatch --dependency=afterok:$job_id ./models/e2e/final/sbatch.sh | cut -d ' ' -f 4)
    echo "Submitted dependent job (Epoch $i): $job_id"
done