#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

ulimit -c 0

if [ ! -d "logs" ]; then
    mkdir logs
fi
if [ ! -d "wandb" ]; then
    mkdir wandb
fi

srun \
    --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model \
    --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml \
    --model hi_lam \
    --graph_name hierarchical \
    --epochs 1 \
    --eval test \
    --n_example_pred 1 \
    --val_steps_to_log 1 3 5 7 9 \
    --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-2x256-02_03_19-2187/min_val_loss_unroll1.ckpt \
    --hidden_dim 256 \
    --processor_layers 2 \
    --num_nodes $SLURM_NNODES \
    --batch_size 1
