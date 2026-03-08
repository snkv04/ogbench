#!/bin/bash

#SBATCH --output=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.err

# Works faster on compute node than GPU node for some reason
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --nodelist=smblade24a2

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /home/svangaru/Desktop/ogbench/
source venv/bin/activate
python -m hierarchical_training_scripts.train_cube_hrl_dqn_profiling \
    --track-with-wandb \
    --save-first-val-episodes-videos=5 \
    --task-id=1 \
    --profiling-start=30000 \
    --profiling-end=31000
