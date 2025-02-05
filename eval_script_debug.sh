#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=05:30:00
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

# Clean any existing installations and run training in same container session
srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    bash -c "python -m pip install --force-reinstall --no-deps -e /users/sadamov/pyprojects/neural-lam && \
            python -m neural_lam.train_model \
                --config_path $SCRATCH/pyprojects_data/neural-lam/test_example/cosmo/config.yaml \
                --model hi_lam \
                --graph_name hierarchical \
                --epochs 1 \
                --eval test \
                --n_example_pred 1 \
                --val_steps_to_log 1 \
                --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-2x300-02_04_17-8057/min_val_loss_unroll1.ckpt \
                --hidden_dim 300 \
                --hidden_dim_grid 150 \
                --time_delta_enc_dim 32 \
                --processor_layers 2 \
                --num_nodes $SLURM_NNODES \
                --batch_size 1 \
                --precision bf16"
