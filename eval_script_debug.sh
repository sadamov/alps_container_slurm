#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

ulimit -c 0

srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container_debug.toml \
    bash -c " \
            python -m neural_lam.train_model \
                --config_path /users/sadamov/pyprojects/neural-lam/tests/datastore_examples/mdp/danra_100m_winds/config.yaml \
                --model hi_lam \
                --graph_name hierarchical \
                --epochs 1 \
                --eval test \
                --ar_steps_eval 2 \
                --val_steps_to_log 1 2 \
                --num_nodes $SLURM_NNODES \
                --batch_size 1 \
                --num_sanity_steps 0 \
                --save_eval_to_zarr_path /iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/eval_results/test.zarr"
