#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

ulimit -c 0

if [ ! -d "logs" ]; then
    mkdir logs
fi
if [ ! -d "wandb" ]; then
    mkdir wandb
fi

# Install package on primary node first
srun -N1 -n1 --container-writable \
    --container-mounts=$SCRATCH/container_overlay:/overlay \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m pip install --no-deps -e /users/sadamov/pyprojects/neural-lam

srun --container-writable \
    --container-mounts=$SCRATCH/container_overlay:/overlay \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml --model hi_lam \
    --graph_name hierarchical --epochs 1 --eval test --n_example_pred 5 --val_steps_to_log 1 3 5 7 9 \
    --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-4x128-01_17_20-9570/min_val_loss.ckpt \
    --hidden_dim 128 --num_nodes $SLURM_NNODES --batch_size 1