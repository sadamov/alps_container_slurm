#!/bin/bash
#SBATCH --job-name=mllam
#SBATCH --output=logs/mllam_%j.out
#SBATCH --error=logs/mllam_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=a-a01
#SBATCH --partition=normal
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4

if [ ! -d "logs" ]; then
    mkdir logs
fi
if [ ! -d "wandb" ]; then
    mkdir wandb
fi

if [ ! -d "cosmo.datastore.zarr" ]; then
    echo "COSMO Datastore not found, preparing data"
    srun --container-writable --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
        -N1 python -m mllam_data_prep --show-progress $SCRATCH/pyprojects_data/neural-lam/cosmo.datastore.yaml --dask-distributed-local-core-fraction 0.1 &
    wait $!
    if [ $? -ne 0 ]; then
        echo "COSMO data preparation failed"
        exit 1
    fi
fi

if [ ! -d "era5.datastore.zarr" ]; then
    echo "ERA5 Datastore not found, preparing data"
    srun --container-writable --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
        -N1 python -m mllam_data_prep --show-progress $SCRATCH/pyprojects_data/neural-lam/era5.datastore.yaml --dask-distributed-local-core-fraction 0.1 &
    wait $!
    if [ $? -ne 0 ]; then
        echo "ERA5 data preparation failed"
        exit 1
    fi
fi

if [ ! -d "graphs" ]; then
    echo "Graphs not found, building it"
    srun --container-writable --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
        -N1 -n1 python -m neural_lam.build_rectangular_graph --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml \
        --graph_name hierarchical --archetype hierarchical --max_num_levels 3 --mesh_node_distance 0.1 &
    wait $!
    if [ $? -ne 0 ]; then
        echo "Graph preparation failed"
        exit 1
    fi
fi

# For finetuning increase epoch and
# --grad_checkpointing --ar_steps_train 4
# --load $SCRATCH/pyprojects_data/neural-lam/saved_models/train-hi_lam-4x128-01_16_20-7552/min_val_loss.ckpt

# Final training step
srun --container-writable \
    --environment=/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/torch_container.toml \
    python -m neural_lam.train_model \
    --config_path $SCRATCH/pyprojects_data/neural-lam/config.yaml \
    --model hi_lam \
    --graph_name hierarchical \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --batch_size 1 \
    --min_lr 0.001 \
    --epochs 200 \
    --val_interval 10 \
    --val_steps_to_log 1 3 5 7 9 \
    --precision bf16-mixed \
    --plot_vars "T_2M" \
    --num_nodes $SLURM_NNODES
