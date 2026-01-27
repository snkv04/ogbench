#!/bin/bash

#SBATCH --output=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=compute
#SBATCH --nodelist=smblade24a3

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /home/svangaru/Desktop/ogbench/
python -m hierarchical_training_scripts.train_cube_hrl_dqn \
    --track-with-wandb \
    --save-first-episode-video \
    --no-noise-initial-state
