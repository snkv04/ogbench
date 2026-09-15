#!/bin/bash

#SBATCH --output=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu1904
#SBATCH --gres=gpu:8

export MUJOCO_GL=osmesa  # To enable headless rendering

# Run the training script
cd /cs/data/people/svangaru/ogbench/
source venv/bin/activate
python -m hierarchical_training_scripts.train_antmaze_lowlevel_td3_subgoals \
    --compile \
    --cudagraphs \
    --subgoal-selection-radius=2.0 \
    --success-tolerance=1.0 \
    --reward-type=sparse \
    --maze-type=arena \
    --max-episode-steps=200 \
    --total-timesteps=2000000 \
    --run-profiling
