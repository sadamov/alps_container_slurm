#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=00:05:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

srun --container-writable --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model --config_path ./config.yaml --model hi_lam --graph_name hierarchical --epochs 1
