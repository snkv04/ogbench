#!/bin/bash

#SBATCH --output=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu1905
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /cs/data/people/svangaru/ogbench/
source venv/bin/activate
python -m hierarchical_training_scripts.train_antmaze_lowlevel_td3 \
    --compile \
    --cudagraphs \
    --fixed-init-ij \
    --reward-type=dense \
    --maze-type=arena \
    --max-episode-steps=500 \
    --total-timesteps=4000000 \
    --run-profiling
