#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=a-a01
#SBATCH --partition=debug
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

# Clean any existing installations and run training in same container session
srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    bash -c "python -m pip install --force-reinstall --no-deps -e /users/sadamov/pyprojects/neural-lam && \
             python -m neural_lam.train_model --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml --model hi_lam \
             --graph_name hierarchical --epochs 1 --eval test --n_example_pred 1 --val_steps_to_log 1 \
             --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-4x128-01_17_20-9570/min_val_loss.ckpt \
             --hidden_dim 128 --num_nodes $SLURM_NNODES --batch_size 1 --precision bf16"
