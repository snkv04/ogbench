#!/bin/bash

#SBATCH --output=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2001
#SBATCH --gres=gpu:4

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /cs/data/people/svangaru/ogbench/
source venv/bin/activate
python -m hierarchical_training_scripts.train_antmaze_lowlevel_td3 \
    --compile \
    --cudagraphs \
    --goal-offset-dir=up \
    --goal-offset-dist=1.0 \
    --reward-type=dense \
    --maze-type=arena \
    --max-episode-steps=200 \
    --total-timesteps=2000000 \
    --run-profiling
