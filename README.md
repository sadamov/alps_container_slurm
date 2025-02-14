# Neural-LAM Project Setup Guide
### 1. Container Setup

While we can use conda environments for local debugging, we need a containerized environment for HPC job submission with SLURM. We use the following tools:

1. Launch an interactive shell on compute node:
`srun -t 01:00:00 -p normal -A a-a01 --pty bash`
2. Build the container with Podman:
`podman build -t my_pytorch:24.04-py3 -f Dockerfile`
3. Squash the container for HPC use with enroot (adjust PATH if needed):
`enroot import -o $SCRATCH/research.sqsh podman://my_pytorch:24.04-py3`

### 2. Environment Management

We use TOML files to manage container environments. This provides a clean separation between container configuration and runtime parameters. In the `torch_container.toml` file, specify the path to the container image (set with enroot).

### 3. Running Training/Eval Jobs

Launch training jobs using SLURM with container support as seen in the scripts `train_script_*.sh`. For training each experimental setup has its own script referencing the appropriate config file. For evaluation, use the `eval_script.sh` script and adjust it accordingly.

### 4. Monitoring

You will see that job allocation happens instantly, and the jobs will be logged in `logs` and tracked on `wandb`. The connection to the latter is often disrupted, so you need to `wandb sync` manually on job completion.

### 5. Troubleshooting

- The compute node have no internet access, so you need to download any required data beforehand
- NCCL errors can happen, usually resubmission and pray works
- If there are dataloader errors while sanity checking, the number of workers should be reduced (4-8 seems okay)
