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

srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    bash -c " \
            python -m pip install --force-reinstall --no-deps git+https://github.com/joeloskarsson/neural-lam-dev.git@boundary_fc && \
            python -m neural_lam.train_model \
                --config_path /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/config_7_19_margin_interior_ifs.yaml \
                --model hi_lam \
                --graph_name hierarchical_7_19_margin \
                --epochs 1 \
                --eval test \
                --n_example_pred 1 \
                --ar_steps_eval 2 \
                --val_steps_to_log 1 2 \
                --load /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/saved_models/train-hi_lam-2x300-02_21_07-6495/last.ckpt \
                --hidden_dim 300 \
                --hidden_dim_grid 150 \
                --time_delta_enc_dim 32 \
                --processor_layers 2 \
                --num_nodes $SLURM_NNODES \
                --plot_vars "T_2M" \
                --batch_size 1 \
                --precision bf16-mixed \
                --save_eval_to_zarr_path /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/eval_results/train-hi_lam-2x300-02_21_07-6495/test.zarr"
