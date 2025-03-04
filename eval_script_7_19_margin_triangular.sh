#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

ulimit -c 0

srun \
    --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model \
        --model hi_lam \
        --epochs 1 \
        --eval val \
        --n_example_pred 1 \
        --ar_steps_eval 120 \
        --val_steps_to_log 1 12 24 36 48 60 72 84 96 108 120 \
        --hidden_dim 300 \
        --hidden_dim_grid 150 \
        --time_delta_enc_dim 32 \
        --processor_layers 2 \
        --num_nodes $SLURM_NNODES \
        --batch_size 1 \
        --plot_vars "T_2M" "U_10M" \
        --precision bf16-mixed \
        --graph_name triangular_hierarchical \
        --config_path /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/config_7_19_margin.yaml \
        --load 
        #--save_eval_to_zarr_path /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/eval_results/preds_7_19_margin_triangular_era.zarr
