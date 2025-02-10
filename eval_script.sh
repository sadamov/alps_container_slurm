#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

ulimit -c 0

srun \
    --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model \
    --config_path $SCRATCH/pyprojects_data/neural-lam/config_7_19_margin.yaml \
    --model hi_lam \
    --graph_name hierarchical_7_19_margin \
    --epochs 1 \
    --eval test \
    --n_example_pred 1 \
    --val_steps_to_log 1 2 3 4 \
    --ar_steps_eval 4 \
    --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-2x300-02_09_09-9499/last.ckpt \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --num_nodes $SLURM_NNODES \
    --batch_size 1
