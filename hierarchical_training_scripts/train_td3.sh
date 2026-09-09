#!/bin/bash

#SBATCH --output=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu1907
#SBATCH --gres=gpu:3

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /cs/data/people/svangaru/ogbench/
python -m hierarchical_training_scripts.train_cube_lowlevel_td3 \
    --compile \
    --cudagraphs \
    --task-id=1 \
    --validation-freq=100000 \
    --num-validation-episodes=10 \
    --num-episode-videos=2 \
    --reward-type=sparse_stepwise
