#!/bin/bash

#SBATCH --output=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/home/svangaru/Desktop/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2001
#SBATCH --gres=gpu:4

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /home/svangaru/Desktop/ogbench/
python -m hierarchical_training_scripts.train_cube_lowlevel_td3_profiling \
    --compile \
    --cudagraphs \
    --task-id=1 \
    --profiling-start=27000 \
    --profiling-end=28000
