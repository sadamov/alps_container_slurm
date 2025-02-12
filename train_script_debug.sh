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
    bash -c "python -m pip install --force-reinstall --no-deps -e /users/sadamov/pyprojects/neural-lam && \
             python -m neural_lam.train_model \
            --config_path /users/sadamov/pyprojects/neural-lam/tests/datastore_examples/mdp/danra_100m_winds/config.yaml \
            --model hi_lam \
            --graph_name hierarchical \
            --hidden_dim 16 \
            --hidden_dim_grid 16 \
            --time_delta_enc_dim 16 \
            --processor_layers 2 \
            --batch_size 1 \
            --epochs 1 \
            --ar_steps_train 2 \
            --ar_steps_eval 2 \
            --val_steps_to_log 1 2 \
            --num_nodes $SLURM_NNODES"
