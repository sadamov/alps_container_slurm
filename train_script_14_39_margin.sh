#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4

ulimit -c 0

# Final training step
srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model \
    --config_path $SCRATCH/pyprojects_data/neural-lam/config_14_39_margin.yaml \
    --model hi_lam \
    --graph_name hierarchical_14_39_margin \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --batch_size 1 \
    --min_lr 0.001 \
    --epochs 200 \
    --val_interval 10 \
    --val_steps_to_log 1 2 3 4 \
    --ar_steps_eval 4 \
    --precision bf16-mixed \
    --plot_vars "T_2M" \
    --num_workers 12 \
    --num_nodes $SLURM_NNODES
