"""Trains an RL agent for the cube environment using the hierarchical RL framework and the DQN algorithm."""

import json
import os
import random
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import gymnasium as gym
from loguru import logger as logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

import ogbench.manipspace  # Register environments
from hierarchical_training_scripts.hierarchical_dqn_agent import (
    QNetwork,
    HierarchicalDQNAgent,
)
from ogbench.manipspace.oracles.hierarchical.utils import (
    render_frame_realtime,
    init_realtime_rendering,
    cleanup_realtime_rendering,
    make_manipspace_env,
)
from hierarchical_training_scripts.inference_cube_hrl import task_done

torch.set_float32_matmul_precision("high")


@dataclass
class Args:
    # General setup
    seed: int = 1048596
    torch_deterministic: bool = True
    cuda: bool = True
    track_with_wandb: bool = False
    wandb_project_name: str = "dqn_cube_hierarchical"
    wandb_entity: Optional[str] = None
    log_freq: int = 100

    # Agent-specific arguments
    disable_no_op: bool = False
    no_op_duration: int = 10

    # Training-specific arguments
    env_id: str = "cube-single-v0"
    task_id: int = 0  # Fixed task ID for all episodes (0 = default task)
    total_timesteps: int = 1000000
    learning_rate: float = 1e-3
    num_envs: int = 1
    max_episode_steps: int = 200
    measure_burnin: int = 3
    episode_window_size: int = 10
    
    # DQN-specific arguments
    buffer_size: int = 100000
    gamma: float = 0.98
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 4

    # Saving and loading
    save_dir: str = ".ogbench/dqn_runs"
    checkpoint_freq: int = 10000  # Save every N steps
    save_model: bool = True
    run_name: str = ""
    load_path: str = ""

    # Visualization
    render_realtime: bool = False
    render_delay: float = 0.001


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def save_checkpoint(
    global_step: int,
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: optim.Optimizer,
    args: Args,
    episode_returns: deque,
    episode_successes: deque,
    training_metrics: List[dict],
    save_path: str,
) -> None:
    checkpoint = {
        "global_step": global_step,
        "q_network_state_dict": q_network.state_dict(),
        "target_network_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        # Random states for reproducibility
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # Running averages
        "episode_returns": list(episode_returns),
        "episode_successes": list(episode_successes),
        # Training metrics history
        "training_metrics": training_metrics,
    }

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to: {save_path}")


def run_validation_episodes(
    env,
    agent,
    num_episodes: int,
    max_episode_steps: int,
) -> dict:
    """Run validation episodes and compute metrics.
    
    Args:
        env: The environment to run validation in.
        agent: The hierarchical agent (PPO or DQN) to use for action selection.
        num_episodes: Number of validation episodes to run.
        max_episode_steps: Maximum steps per episode.
    
    Returns:
        Dictionary containing validation metrics:
            - 'success_rate': Success rate (tasks completed at end of episode)
            - 'completion_rate': Completion rate (tasks completed at any point)
            - 'num_episodes': Number of episodes run
    """
    tasks_completed_at_end = 0
    tasks_completed_at_all = 0
    tasks_attempted = 0
    
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Running validation episodes"):
        ob, info = env.reset()
        agent.reset(ob, info)
        
        tasks_attempted += 1
        episode_had_success = False
        
        done = False
        step = 0
        
        while not done:
            # Get action from agent
            action = agent.select_action(ob, info)
            action = np.array(action)
            
            # Step through time
            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Check task completion
            is_task_done = task_done(env, info)
            if done and is_task_done:
                tasks_completed_at_end += 1
            if is_task_done and not episode_had_success:
                tasks_completed_at_all += 1
                episode_had_success = True
            
            ob = next_ob
            step += 1

        assert step == max_episode_steps, "Each episode should last its full length"

    # Compute metrics
    success_rate = tasks_completed_at_end / tasks_attempted if tasks_attempted > 0 else 0.0
    completion_rate = tasks_completed_at_all / tasks_attempted if tasks_attempted > 0 else 0.0
    
    return {
        'success_rate': success_rate,
        'completion_rate': completion_rate,
        'num_episodes': num_episodes,
    }


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
"""Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    
    # Parse arguments
    args = tyro.cli(Args)
    assert args.num_envs == 1, "Only one environment is supported for hierarchical DQN at the moment"
    args.num_episodes = args.total_timesteps // args.max_episode_steps
    if args.total_timesteps % args.max_episode_steps != 0:
        logging.warning(
            f"WARNING: total_timesteps ({args.total_timesteps}) is not divisible by max_episode_steps ({args.max_episode_steps})."
            f"Will instead train for {args.num_episodes * args.max_episode_steps} steps ({args.total_timesteps - (args.num_episodes * args.max_episode_steps)} fewer)."
        )
        args.total_timesteps = args.num_episodes * args.max_episode_steps

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}__{timestamp}" if args.run_name else f"{args.env_id}__{args.seed}__{timestamp}"

    # Optional wandb tracking
    if args.track_with_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
            dir=".ogbench/wandb",
        )

    # Setup save directory
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"Saving to: {save_path}")

    # TRY NOT TO MODIFY: Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Sets device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # Environment setup
    env = make_manipspace_env(args.env_id, args.seed, args.max_episode_steps, args.task_id)
    logging.info(f"Using fixed task_id={args.task_id} for all episodes")

    # Initialize real-time rendering if enabled
    render_window_name = "DQN Training - Real-time Rendering"
    if args.render_realtime:
        init_realtime_rendering(render_window_name)

    # Q-Network and target network
    num_options = 10 if not args.disable_no_op else 9
    obs_dim = HierarchicalDQNAgent.OBS_DIM
    q_network = QNetwork(obs_dim, num_options, hidden_dim=256).to(device)
    target_network = QNetwork(obs_dim, num_options, hidden_dim=256).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Replay buffer for high-level transitions
    # Note: We need a custom observation space for the hierarchical agent
    obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32
    )
    action_space = gym.spaces.Discrete(num_options)
    rb = ReplayBuffer(
        args.buffer_size,
        obs_space,
        action_space,
        device,
        handle_timeout_termination=False,
    )

    # Load from checkpoint if specified
    start_global_step = 0
    episode_returns = deque(maxlen=args.episode_window_size)
    episode_successes = deque(maxlen=args.episode_window_size)
    training_metrics = []
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_path}")
        logging.info(f"Loading checkpoint from: {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        q_network.load_state_dict(checkpoint["q_network_state_dict"])
        target_network.load_state_dict(checkpoint["target_network_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_global_step = checkpoint["global_step"]
        
        # Restore random states for reproducibility
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "np_random_state" in checkpoint:
            np.random.set_state(checkpoint["np_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if "torch_cuda_random_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
        
        # Restore running averages
        if "episode_returns" in checkpoint:
            episode_returns.extend(checkpoint["episode_returns"])
        if "episode_successes" in checkpoint:
            episode_successes.extend(checkpoint["episode_successes"])
        
        # Restore training metrics history
        if "training_metrics" in checkpoint:
            training_metrics = checkpoint["training_metrics"]

        logging.info("Finished loading checkpoint")
        logging.info(f"Resuming from global_step={start_global_step}")

    # Hierarchical agent
    ob, info = env.reset(seed=args.seed)
    agent = HierarchicalDQNAgent(
        env, q_network, device,
        disable_no_op=args.disable_no_op,
        no_op_duration=args.no_op_duration,
    )
    agent.reset(ob, info)

    # Training variables
    start_time = None
    start_burnin_global_step = start_global_step
    episode_return = 0.0
    episode_step_count = 0
    current_hl_info = None
    
    # Initial render
    if args.render_realtime:
        render_frame_realtime(env, render_window_name, args.render_delay)

    # Main training loop
    pbar = tqdm.tqdm(range(start_global_step, start_global_step + args.total_timesteps))
    for global_step in pbar:
        # Start measuring speed after burn-in
        if global_step == args.learning_starts + args.measure_burnin:
            start_time = time.time()
            start_burnin_global_step = global_step

        # Compute epsilon for exploration
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        agent.epsilon = epsilon  # Set epsilon for the agent's select_high_level_action
        
        # Execute action using the hierarchical agent (handles option execution automatically)
        low_level_action = agent.select_action(ob, info)
        
        # Check if a new high-level action was selected
        if agent.was_new_option_selected():
            # Store previous high-level transition if it exists
            if current_hl_info is not None:
                # Add transition to replay buffer
                rb.add(
                    current_hl_info['obs'].cpu().numpy(),
                    current_hl_info['next_obs'].cpu().numpy(),
                    np.array([current_hl_info['action']]),
                    np.array([current_hl_info['accumulated_reward']]),
                    np.array([current_hl_info['done']]),
                    [{}],  # infos
                )

            # Start tracking new option
            obs, action = agent.get_last_transition_info()
            current_hl_info = {
                'obs': obs,
                'action': action,
                'accumulated_reward': 0.0,
                'option_length': 0,
            }
        
        # Step environment
        next_ob, reward, terminated, truncated, next_info = env.step(low_level_action)
        current_hl_info['next_obs'] = agent.get_obs_tensor(next_info)
        done = terminated or truncated
        episode_step_count += 1
        assert terminated == False and (truncated == False or episode_step_count == args.max_episode_steps), "Each episode should last its full length"

        # Accumulate reward for the current option (with forward discounting)
        assert current_hl_info is not None, "Current high-level info not found"
        current_hl_info['accumulated_reward'] += (args.gamma ** current_hl_info['option_length']) * reward
        current_hl_info['option_length'] += 1
        episode_return += reward

        # Render frame if enabled
        if args.render_realtime and agent.active_option is not None:
            render_frame_realtime(
                env, render_window_name, args.render_delay,
                option_idx=agent._options.index(agent.active_option),
                option_text=agent.active_option.name,
            )

        # Check if the episode is done
        if done:
            # Store final transition for this episode
            current_hl_info['done'] = True
            rb.add(
                current_hl_info['obs'].cpu().numpy(),
                current_hl_info['next_obs'].cpu().numpy(),
                np.array([current_hl_info['action']]),
                np.array([current_hl_info['accumulated_reward']]),
                np.array([current_hl_info['done']]),
                [{}],  # infos
            )
            current_hl_info = None

            # Track episode stats
            episode_returns.append(episode_return)
            episode_successes.append(float(next_info['success']))
            episode_return = 0.0
            episode_step_count = 0

            # Reset environment and agent
            ob, info = env.reset()
            agent.reset(ob, info)
        else:
            # Store next_obs for potential transition
            current_hl_info['done'] = False
            ob, info = next_ob, next_info

        # DQN update
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    # Compute option-aware discounting for target
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    # Note: We don't use the done signal here, in order to capture the infinite horizon
                    td_target = data.rewards.flatten() + args.gamma * target_max
                
                # Current Q-values
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        # Logging
        if global_step % args.log_freq == 0 and start_time is not None:
            # Compute metrics
            speed = (global_step - start_burnin_global_step) / (time.time() - start_time)
            avg_return = np.mean(episode_returns) if episode_returns else 0
            avg_success = np.mean(episode_successes) if episode_successes else 0
            desc = f"speed: {speed:4.2f} sps, return: {avg_return:.2f}, success: {avg_success:.2%}"
            pbar.set_description(desc)
            
            # Track metrics
            metrics = {
                "train/global_step": global_step,
                "train/speed": float(speed),
                "train/avg_episode_return": float(avg_return),
                "train/success_rate": float(avg_success),
                "train/epsilon": float(epsilon),
            }
            if global_step > args.learning_starts:
                metrics["train/loss"] = float(loss.item())
            training_metrics.append(metrics)

            # Log training metrics to wandb
            if args.track_with_wandb:
                wandb.log(metrics, step=global_step)

        # Save checkpoint and perform validation
        if args.save_model and global_step % args.checkpoint_freq == 0 and global_step > 0:
            # Save checkpoint
            checkpoint_path = os.path.join(save_path, f"checkpoint_step{global_step}.pt")
            save_checkpoint(
                global_step=global_step,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                args=args,
                episode_returns=episode_returns,
                episode_successes=episode_successes,
                training_metrics=training_metrics,
                save_path=checkpoint_path
            )

            # Run validation episodes
            assert global_step % args.max_episode_steps == 0, "Validation should run right after an episode ends"
            logging.info(f"Running validation with {args.episode_window_size} episodes...")
            q_network.eval()  # Set to evaluation mode
            val_metrics = run_validation_episodes(
                env=env,
                agent=agent,
                num_episodes=args.episode_window_size,
                max_episode_steps=args.max_episode_steps,
            )
            q_network.train()  # Set back to training mode
            
            # Log validation metrics
            logging.info(f"Validation results (step {global_step}):")
            logging.info(f"    success_rate={val_metrics['success_rate']:.2%}")
            logging.info(f"    completion_rate={val_metrics['completion_rate']:.2%}")
            
            # Log validation metrics to wandb
            if args.track_with_wandb:
                wandb.log({
                    "val/global_step": global_step,
                    "val/success_rate": float(val_metrics['success_rate']),
                    "val/completion_rate": float(val_metrics['completion_rate']),
                }, step=global_step)
            
            # Resets training state after validation
            ob, info = env.reset()
            agent.reset(ob, info)
            episode_return = 0.0
            episode_step_count = 0
            current_hl_info = None

    # Cleanup
    env.close()
    if args.render_realtime:
        cleanup_realtime_rendering()
    if args.track_with_wandb:
        wandb.finish()

    # Save final model
    if args.save_model:
        final_model_path = os.path.join(save_path, f"final_model_step{global_step}.pt")
        save_checkpoint(
            global_step=global_step,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            args=args,
            episode_returns=episode_returns,
            episode_successes=episode_successes,
            training_metrics=training_metrics,
            save_path=final_model_path
        )
        logging.info(f"Final model (after {global_step} steps) has been saved to {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(save_path, f"training_metrics_step{global_step}.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    logging.info(f"Saved training metrics to {metrics_path}")

    # Final logging
    logging.info(f"\nTraining complete!")
    logging.info(f"Final average return across episodes: {np.mean(episode_returns):.2f}")
    logging.info(f"Final success rate across episodes: {np.mean(episode_successes):.2%}")
