# Training the hierarchical agent with PPO

### Environment setup commands

To set up your environment to align with the output of freezing my environment,
first make sure that the path to the `ogbench` repository on your machine in the
`dependencies/environment.yaml` file is updated to point to the correct location.
Then, run:
```
pip install uv
uv venv
source .venv/bin/activate
uv pip sync dependencies/environment.yaml
```

To install additional dependencies that were not already saved with `uv`:
```
uv pip install torch tyro wandb loguru stable-baselines3 tensordict torchrl flax ml-collections distrax
```

### Training script

This command runs the main PPO training script:
```
.venv/bin/python -m hierarchical_training_scripts.train_cube_hrl_ppo
```
The checkpoints and training metrics are then saved to `.ogbench/ppo_runs`.

As an example with more arguments, to start a short training run, have a window pop up to show the frames in real time, and visualize the training metrics with Weights and Biases, run this:
```
.venv/bin/python -m hierarchical_training_scripts.train_cube_hrl_ppo \
    --total-timesteps 10000 \
    --render-realtime \
    --track-with-wandb
```
Further arguments for customization are visible in the `Args` class definition in
`train_cube_hrl_ppo.py`.

### Inference/evaluation script

This command loads a checkpoint (which, in this example, is at `.ogbench/ppo_runs/cube-single-v0__0__20251224_180454/checkpoint_iter400.pt`) and runs inference, subsequently saving the generated data to `.ogbench/data`:
```
.venv/bin/python -m hierarchical_training_scripts.inference_cube_hrl_ppo \
    --checkpoint-path .ogbench/ppo_runs/cube-single-v0__0__20251224_180454/checkpoint_iter400.pt
```

As an example with more arguments, to run for fewer episodes, render the environment in real time, save a video of the first episode, and use an expert policy instead of a policy trained with PPO, run this:
```
.venv/bin/python -m hierarchical_training_scripts.inference_cube_hrl_ppo \
    --num-episodes 10 \
    --render-realtime \
    --save-first-episode-video \
    --agent-type hierarchical_oracle
```
