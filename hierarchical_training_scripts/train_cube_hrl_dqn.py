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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

import ogbench.manipspace  # Register environments
from ogbench.manipspace.oracles.hierarchical.hierarchical_agent import HierarchicalAgent
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
    LiftVerticallyOption,
    NoOpOption,
)
from ogbench.manipspace.oracles.hierarchical.utils import (
    render_frame_realtime,
    init_realtime_rendering,
    cleanup_realtime_rendering,
    make_manipspace_env,
)

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


class QNetwork(nn.Module):
    """Q-Network that maps observations to Q-values for each option."""
    
    def __init__(self, obs_dim: int, num_options: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )
    
    def forward(self, x):
        return self.network(x)


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
    avg_returns: deque,
    avg_successes: deque,
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
        "avg_returns": list(avg_returns),
        "avg_successes": list(avg_successes),
        # Training metrics history
        "training_metrics": training_metrics,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


class HierarchicalDQNAgent(HierarchicalAgent):
    """Hierarchical agent that uses Q-network for high-level option selection.
    
    Extends HierarchicalAgent to use DQN for learning which option to execute,
    while the low-level option execution is handled by pre-defined options.
    
    Attributes:
        OBS_DIM: Dimension of observation features for the policy (14)
            - effector_pos (3)
            - effector_yaw (1)
            - gripper_opening (1)
            - gripper_contact (1)
            - block_pos (3)
            - block_yaw (1)
            - target_pos (3)
            - target_yaw (1)
    """
    
    OBS_DIM = 14  # effector_pos(3) + effector_yaw(1) + gripper(2) + block(4) + target(4)
    
    def __init__(
        self,
        env,
        q_network: QNetwork,
        device: torch.device,
        disable_no_op: bool = False,
        no_op_duration: int = 10,
    ):
        super().__init__(options=[], env=env, min_norm=0.08)
        self.q_network = q_network
        self.device = device
        self.disable_no_op = disable_no_op
        self.no_op_duration = no_op_duration
        self.epsilon = 0.0  # Default to greedy if not set
        
        # Will be initialized in reset()
        self._target_block = None
        self._target_pos = None
        self._target_yaw = None
        self._final_pos = None
        self._final_yaw = None
    
    def reset(self, ob, info):
        """Reset the agent for a new episode/task.
        
        Args:
            ob: Initial observation
            info: Info dict from environment
        """
        super().reset(ob, info)
        
        env = self._env.unwrapped
        
        # Handle both data_collection and task modes
        if env._mode == 'data_collection':
            # In data_collection mode, target info is in privileged keys
            self._target_block = info['privileged/target_block']
            self._target_pos = info['privileged/target_block_pos'].copy()
            self._target_yaw = info['privileged/target_block_yaw'][0]
        else:
            # In task mode, target info is in cur_task_info
            # For cube-single, there's only 1 cube, so target_block = 0
            self._target_block = 0
            self._target_pos = env.cur_task_info['goal_xyzs'][self._target_block].copy()
            self._target_yaw = 0.0  # Task mode uses identity orientation for goals
        
        self._final_pos = np.random.uniform(*env._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)
        self._options = self._create_options()
    
    def _create_options(self) -> List:
        """Create the set of options for cube manipulation.
        
        Returns:
            List of 10 options (or 9, if disable_no_op is True):
                0: no_op - Do nothing for N steps
                1: move_above_block - Move end-effector above the target block
                2: move_to_block - Move end-effector to the target block
                3: grasp_block - Close gripper to grasp the block
                4: lift_after_grasp - Lift the grasped block
                5: move_above_target - Move above the target position
                6: move_to_target - Move to the target position
                7: release - Open gripper to release block
                8: lift_after_release - Lift after releasing
                9: move_to_final - Move to final position
        """
        env = self._env
        target_block = self._target_block
        final_pos = self._final_pos
        final_yaw = self._final_yaw

        def block_above_pos(ob, info):
            return info[f'privileged/block_{target_block}_pos'] + np.array([0, 0, 0.18])

        def block_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            block_yaw_val = info[f'privileged/block_{target_block}_yaw'][0]
            return self._shortest_yaw(effector_yaw, block_yaw_val)

        def block_pos(ob, info):
            return info[f'privileged/block_{target_block}_pos']

        # Use stored target info (works for both data_collection and task modes)
        stored_target_pos = self._target_pos
        stored_target_yaw = self._target_yaw

        def target_above_pos(ob, info):
            return stored_target_pos + np.array([0, 0, 0.18])

        def target_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            return self._shortest_yaw(effector_yaw, stored_target_yaw)

        def target_pos(ob, info):
            return stored_target_pos

        def get_final_pos(ob, info):
            return final_pos

        def get_final_yaw(ob, info):
            return final_yaw

        options = [
            MoveToPositionOption('move_above_block', env, block_above_pos, block_yaw, gripper_state=-1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_block', env, block_pos, block_yaw, gripper_state=-1, min_norm=self._min_norm),
            GraspOption('grasp_block', env, block_pos, block_yaw, min_norm=self._min_norm),
            LiftVerticallyOption('lift_after_grasp', env, block_pos, target_height=0.36, target_yaw_fn=target_yaw, gripper_state=1, min_norm=self._min_norm),
            MoveToPositionOption('move_above_target', env, target_above_pos, target_yaw, gripper_state=1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_target', env, target_pos, target_yaw, gripper_state=1, min_norm=self._min_norm),
            ReleaseOption('release', env),
            LiftVerticallyOption('lift_after_release', env, block_pos, target_height=0.32, target_yaw_fn=get_final_yaw, gripper_state=-1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_final', env, get_final_pos, get_final_yaw, gripper_state=-1, min_norm=self._min_norm),
        ]
        if not self.disable_no_op:
            options.insert(0, NoOpOption('no_op', env, duration=self.no_op_duration))
        return options
    
    @staticmethod
    def _shortest_yaw(current_yaw: float, target_yaw: float) -> float:
        """Compute shortest rotation to reach target yaw."""
        diff = target_yaw - current_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return current_yaw + diff
    
    def get_obs_tensor(self, info) -> torch.Tensor:
        """Extract observation features for the high-level policy.
        
        Args:
            info: Info dict from environment step
            
        Returns:
            Tensor of shape (14,) containing:
                - effector_pos (3)
                - effector_yaw (1)
                - gripper_opening (1)
                - gripper_contact (1)
                - block_pos (3)
                - block_yaw (1)
                - target_pos (3)
                - target_yaw (1)
        """
        # Use stored target info (works for both data_collection and task modes)
        features = np.concatenate([
            info['proprio/effector_pos'],
            np.atleast_1d(info['proprio/effector_yaw']),
            np.atleast_1d(info['proprio/gripper_opening']),
            np.atleast_1d(info['proprio/gripper_contact']),
            info[f'privileged/block_{self._target_block}_pos'],
            np.atleast_1d(info[f'privileged/block_{self._target_block}_yaw']),
            self._target_pos,
            np.atleast_1d(self._target_yaw),
        ])
        return torch.as_tensor(features, dtype=torch.float32, device=self.device)
    
    def select_high_level_action(self, ob, info):
        """Select which option to execute using epsilon-greedy Q-learning.
        
        This method is called by the base HierarchicalAgent when no option is active.
        For training with epsilon-greedy exploration, set self.epsilon before calling.
        
        Args:
            ob: Current observation
            info: Info dict from environment
            
        Returns:
            The selected Option object to execute
        """
        obs = self.get_obs_tensor(info).unsqueeze(0)
        
        if random.random() < self.epsilon:
            # Random exploration
            action_idx = random.randint(0, len(self._options) - 1)
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                q_values = self.q_network(obs)
                action_idx = torch.argmax(q_values, dim=1).item()
        
        # Store for training (observation and action index)
        self.last_obs = obs.squeeze(0)
        self.last_action_idx = action_idx
        
        return self._options[action_idx]
    
    def select_action(self, ob, info):
        """Select action and track when new high-level decisions are made.
        
        This override adds transition tracking for DQN training while using the
        base class's option execution logic.
        """
        # Check if we're about to select a new option
        will_select_new_option = (self._active_option is None or not self._active_option.active)
        
        if will_select_new_option:
            # Signal that a new high-level decision is about to be made
            self._new_option_selected = True
        else:
            self._new_option_selected = False
        
        # Call base class to handle option selection and execution
        action = super().select_action(ob, info)
        
        return action
    
    def get_last_transition_info(self) -> Tuple[torch.Tensor, int]:
        """Get the observation and action index from the last high-level decision.
        
        Returns:
            obs: Observation tensor used for selection
            action_idx: Option index that was selected
        """
        return self.last_obs, self.last_action_idx
    
    def was_new_option_selected(self) -> bool:
        """Check if a new option was just selected in the last select_action call.
        
        Returns:
            True if a new high-level decision was made
        """
        assert hasattr(self, '_new_option_selected'), "New option selected attribute not found"
        return self._new_option_selected


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
        print(
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
    print(f"Saving to: {save_path}")

    # TRY NOT TO MODIFY: Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Sets device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env = make_manipspace_env(args.env_id, args.seed, args.max_episode_steps, args.task_id)
    print(f"Using fixed task_id={args.task_id} for all episodes")

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
    avg_returns = deque(maxlen=100)
    avg_successes = deque(maxlen=100)
    training_metrics = []
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_path}")
        print(f"Loading checkpoint from: {args.load_path}")
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
        if "avg_returns" in checkpoint:
            avg_returns.extend(checkpoint["avg_returns"])
        if "avg_successes" in checkpoint:
            avg_successes.extend(checkpoint["avg_successes"])
        
        # Restore training metrics history
        if "training_metrics" in checkpoint:
            training_metrics = checkpoint["training_metrics"]

        print("Finished loading checkpoint")
        print(f"Resuming from global_step={start_global_step}")

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
            avg_returns.append(episode_return)
            avg_successes.append(float(next_info['success']))
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
        if global_step % 100 == 0 and start_time is not None:
            # Compute metrics
            speed = (global_step - start_burnin_global_step) / (time.time() - start_time)
            avg_ret = np.mean(avg_returns) if avg_returns else 0
            avg_suc = np.mean(avg_successes) if avg_successes else 0
            desc = f"speed: {speed:4.2f} sps, return: {avg_ret:.2f}, success: {avg_suc:.2%}"
            pbar.set_description(desc)
            
            # Track metrics
            metrics = {
                "global_step": global_step,
                "speed": float(speed),
                "avg_episode_return": float(avg_ret),
                "success_rate": float(avg_suc),
                "epsilon": float(epsilon),
            }
            if global_step > args.learning_starts:
                metrics["loss"] = float(loss.item())
            training_metrics.append(metrics)

            # Log metrics to remote server
            if args.track_with_wandb:
                wandb.log(metrics, step=global_step)

        # Save checkpoint
        if args.save_model and global_step % args.checkpoint_freq == 0 and global_step > 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_step{global_step}.pt")
            save_checkpoint(
                global_step=global_step,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                args=args,
                avg_returns=avg_returns,
                avg_successes=avg_successes,
                training_metrics=training_metrics,
                save_path=checkpoint_path
            )

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
            avg_returns=avg_returns,
            avg_successes=avg_successes,
            training_metrics=training_metrics,
            save_path=final_model_path
        )
        print(f"Final model (after {global_step} steps) has been saved to {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(save_path, f"training_metrics_step{global_step}.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    # Final logging
    print(f"\nTraining complete!")
    print(f"Final average return across episodes: {np.mean(avg_returns):.2f}")
    print(f"Final success rate across episodes: {np.mean(avg_successes):.2%}")
