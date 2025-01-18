#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4

if [ ! -d "logs" ]; then
    mkdir logs
fi
if [ ! -d "wandb" ]; then
    mkdir wandb
fi

srun --container-writable --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml --model hi_lam \
    --graph_name hierarchical --epochs 1 --eval test --n_example_pred 5 --val_steps_to_log 1 3 5 7 9 \
    --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-4x128-01_17_20-9570/min_val_loss.ckpt \
    --hidden_dim 128 --num_nodes 4 --batch_size 2
