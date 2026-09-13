#!/bin/bash

#SBATCH --output=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.out
#SBATCH --error=/cs/data/people/svangaru/ogbench/slurm_logs/%x_%j.err

#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --partition=compute
#SBATCH --nodelist=typhon
#SBATCH --cpus-per-task=48

export MUJOCO_GL=osmesa  # To enable headless rendering

# full-square:
    # --checkpoint-up=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-08_02-09-40/checkpoints/checkpoint_step1950000.pt \
    # --checkpoint-down=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-09_00-28-52/checkpoints/checkpoint_step1950000.pt \
    # --checkpoint-left=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-09_00-28-56/checkpoints/checkpoint_step1950000.pt \
    # --checkpoint-right=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-10_03-56-57/checkpoints/checkpoint_step1950000.pt \

# half-square:
    # --checkpoint-up=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-17/checkpoints/checkpoint_step1950000.pt \
    # --checkpoint-down=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-47/checkpoints/checkpoint_step1950000.pt \
    # --checkpoint-left=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-36/checkpoints/checkpoint_step1100000.pt \
    # --checkpoint-right=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-23/checkpoints/checkpoint_step1950000.pt \

# Run the training script
cd /cs/data/people/svangaru/ogbench/
source venv/bin/activate
python -m hierarchical_training_scripts.train_antmaze_highlevel_dqn \
    --save-first-val-episodes-videos=4 \
    --no-validation-greedy \
    --checkpoint-up=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-17/checkpoints/checkpoint_step1950000.pt \
    --checkpoint-down=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-47/checkpoints/checkpoint_step1950000.pt \
    --checkpoint-left=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-36/checkpoints/checkpoint_step1100000.pt \
    --checkpoint-right=/cs/data/people/svangaru/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-23/checkpoints/checkpoint_step1950000.pt \
    --termination-time=50 \
    --max-episode-steps=2000 \
    --total-timesteps=2000000 \
    --reward-task-id=1 \
    --goal-radius=1.5 \
    --track-with-wandb \
    --reward-type=sparse \
    --run-profiling