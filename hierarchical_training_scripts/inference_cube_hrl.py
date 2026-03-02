"""Runs inference, and generates a dataset, using a hierarchical RL agent in the cube environment."""

import pathlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, List

import imageio.v2 as imageio
import gymnasium
import numpy as np
import torch
import tyro
from tqdm import trange

import ogbench.manipspace  # noqa
from hierarchical_training_scripts.hierarchical_ppo_agent import (
    PolicyNetwork,
    HierarchicalPPOAgent,
)
from ogbench.manipspace.oracles.hierarchical.cube_hierarchical_oracle import (
    CubeHierarchicalOracle,
)
from ogbench.manipspace.oracles.hierarchical.cube_hierarchical_random import (
    CubeHierarchicalRandom,
)
from hierarchical_training_scripts.train_cube_hrl_ppo import (
    render_frame_realtime,
    init_realtime_rendering,
    cleanup_realtime_rendering,
)
from hierarchical_training_scripts.hierarchical_dqn_agent import (
    QNetwork,
    HierarchicalDQNAgent,
)
from ogbench.manipspace.oracles.hierarchical.utils import (
    add_text_overlay,
    save_episode_video,
)


@dataclass
class Args:
    # Agent type
    agent_type: Literal["hierarchical_ppo", "hierarchical_dqn", "hierarchical_oracle", "hierarchical_random"] = "hierarchical_ppo"
    checkpoint_path: str = ""  # Path to checkpoint file (required for hierarchical_ppo and hierarchical_dqn)
    
    # PPO-specific parameters (ignored for DQN, oracle, and random agents)
    deterministic: bool = False  # If True, use argmax; if False, sample from policy
    temperature: float = 1.0  # Temperature for sampling (higher = more random)

    # DQN-specific parameters (ignored for PPO, oracle, and random agents)
    end_e: float = 0.05  # Final epsilon-greedy exploration rate from training

    # Agent-specific arguments
    disable_no_op: bool = False
    no_op_duration: int = 10

    # Environment
    env_name: str = "cube-single-v0"
    seed: int = 1048596
    task_id: int = 0  # Fixed task ID for all episodes (0 = default task)
    noise_initial_state: bool = True
    reward_is_neg_dist: bool = False
    
    # Dataset generation
    num_episodes: int = 1000
    max_episode_steps: int = 200
    save_path: Optional[str] = None  # If None, auto-generated
    
    # Action noise (optional, for diversity)
    noise: float = 0.0
    
    # Visualization
    save_first_episode_video: bool = False
    render_realtime: bool = False
    render_delay: float = 0.001
    
    # Device
    cuda: bool = True


def task_done(env, info, threshold: float = 0.04) -> bool:
    # In task mode, target_block is always 0 for cube-single
    target_block = 0
    target_pos = env.unwrapped.cur_task_info['goal_xyzs'][target_block]
    block_pos = info[f'privileged/block_{target_block}_pos']
    return np.linalg.norm(target_pos - block_pos) <= threshold


def main():
    args = tyro.cli(Args)
    
    if args.agent_type in ["hierarchical_ppo", "hierarchical_dqn"]:
        assert args.checkpoint_path, f"Must provide --checkpoint_path for {args.agent_type} agent"
    
    # Set save path
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.agent_type == "hierarchical_ppo":
            # PPO uses deterministic flag and temperature
            policy_type = "deterministic" if args.deterministic else f"temp{args.temperature}"
        elif args.agent_type == "hierarchical_dqn":
            policy_type = f"eps{args.end_e}"
        elif args.agent_type == "hierarchical_oracle":
            policy_type = "oracle"
        elif args.agent_type == "hierarchical_random":
            policy_type = "random"
        else:
            raise ValueError(f"Invalid agent type: {args.agent_type}")
        args.save_path = f".ogbench/data/{args.env_name}-{args.agent_type}-task{args.task_id}-{policy_type}-{timestamp}.npz"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.agent_type in ["hierarchical_ppo", "hierarchical_dqn"]:
        print(f"Using device: {device}")
    
    # Load checkpoint (for PPO and DQN agents)
    checkpoint = None
    if args.agent_type == "hierarchical_ppo":
        print(f"Loading PPO checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        print(f"Checkpoint trained for {checkpoint['iteration']} iterations, {checkpoint['global_step']} steps")
    elif args.agent_type == "hierarchical_dqn":
        print(f"Loading DQN checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        print(f"Checkpoint trained for {checkpoint['global_step']} steps")
    
    # Initialize environment in task mode with fixed task
    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode='task',
        reward_task_id=args.task_id,  # Fixed task for all episodes
        max_episode_steps=args.max_episode_steps,
        noise_initial_state=args.noise_initial_state,
        reward_is_neg_dist=args.reward_is_neg_dist,
    )
    print(f"Using fixed task_id={args.task_id} for all episodes")
    print(f"noise_initial_state={args.noise_initial_state}")
    print(f"reward_is_neg_dist={args.reward_is_neg_dist}")
    
    # Initialize agent based on agent_type
    ob, info = env.reset(seed=args.seed)
    if args.agent_type == "hierarchical_ppo":
        # Initialize policy network
        policy_network = PolicyNetwork(
            HierarchicalPPOAgent.OBS_DIM,
            10 if not args.disable_no_op else 9,
            hidden_dim=256,
            device=device,
        )
        policy_network.load_state_dict(checkpoint['model_state_dict'])
        policy_network.eval()
        print("Policy network loaded successfully")
        
        agent = HierarchicalPPOAgent(
            env, policy_network, device,
            deterministic=args.deterministic,
            temperature=args.temperature,
            disable_no_op=args.disable_no_op,
            no_op_duration=args.no_op_duration,
        )
    elif args.agent_type == "hierarchical_dqn":
        # Initialize Q-network
        num_options = 10 if not args.disable_no_op else 9
        obs_dim = HierarchicalDQNAgent.OBS_DIM
        q_network = QNetwork(obs_dim, num_options, hidden_dim=256)
        q_network.to(device)
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        q_network.eval()
        print("Q-network loaded successfully")
        
        agent = HierarchicalDQNAgent(
            env, q_network, device,
            disable_no_op=args.disable_no_op,
            no_op_duration=args.no_op_duration,
        )
        agent.epsilon = args.end_e
    elif args.agent_type == "hierarchical_oracle":
        agent = CubeHierarchicalOracle(
            env=env,
            max_step=args.max_episode_steps,
            no_op_option_prob=0.0,
            suboptimal_option_prob=0.0,
        )
        print("Using CubeHierarchicalOracle agent")
    else:  # hierarchical_random
        agent = CubeHierarchicalRandom(
            env=env,
            max_step=args.max_episode_steps,
        )
        print("Using CubeHierarchicalRandom agent")
    agent.reset(ob, info)
    
    # Collect data
    dataset = defaultdict(list)
    episode_frames = []
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = args.num_episodes
    num_val_episodes = args.num_episodes // 10
    
    # Task mode statistics
    tasks_completed_at_end = 0
    tasks_completed_at_all = 0
    tasks_attempted = 0
    episode_returns = []
    per_task_stats = defaultdict(lambda: {
        'attempted': 0,
        'completed_at_end': 0,
        'completed_at_all': 0,
        'episode_returns': [],
    })
    
    # Initialize window for real-time rendering if enabled
    window_name = 'Learned Policy - Real-time Rendering'
    if args.render_realtime:
        init_realtime_rendering(window_name)
    
    print(f"\nGenerating {num_train_episodes + num_val_episodes} episodes...")
    print(f"  Agent type: {args.agent_type}")
    print(f"  Train: {num_train_episodes}, Val: {num_val_episodes}")
    if args.agent_type == "hierarchical_ppo":
        print(f"  PPO Policy: {'Deterministic (argmax)' if args.deterministic else f'Stochastic (temperature={args.temperature})'}")
    elif args.agent_type == "hierarchical_dqn":
        print(f"  DQN Policy: Epsilon-greedy (epsilon={args.end_e})")
    print(f"  Action noise: {args.noise}")
    
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        ob, info = env.reset()
        agent.reset(ob, info)
        
        # Get current task info
        task_id = env.unwrapped.cur_task_id
        task_name = env.unwrapped.cur_task_info['task_name']
        per_task_stats[task_id]['attempted'] += 1
        tasks_attempted += 1  # Each episode starts with one task
        episode_had_success = False
        episode_return = 0.0
        
        # Track option state
        prev_option_terminated = True
        current_option_idx = None
        current_option_name = None
        
        if ep_idx == 0 and args.save_first_episode_video:
            frame = add_text_overlay(env.render(), current_option_idx, current_option_name)
            episode_frames = [frame]
        
        if args.render_realtime:
            render_frame_realtime(env, window_name, args.render_delay, 
                                  current_option_idx, current_option_name)
        
        done = False
        step = 0
        
        while not done:
            # Get action from learned policy (via hierarchical agent)
            action = agent.select_action(ob, info)
            action = np.array(action)
            
            # Add optional noise
            if args.noise > 0:
                action = action + np.random.normal(0, args.noise, action.shape)
            action = np.clip(action, -1, 1)
            
            next_ob, reward, terminated, truncated, info = env.step(action)
            # active_opt = agent.active_option
            # opt_idx = agent._options.index(active_opt) if active_opt is not None else -1
            # opt_name = active_opt.name if active_opt is not None else "None"
            # gripper_opening = info['proprio/gripper_opening'][0]
            # gripper_contact = info['proprio/gripper_contact'][0]
            # print(f"option_idx={opt_idx}, option={opt_name}")
            # print(f"gripper_opening={gripper_opening}")
            # print(f"gripper_contact={gripper_contact}")
            # print()
            done = terminated or truncated
            episode_return += reward
            # print(f"done = {done}, step = {step}")
            
            # Track option info
            current_active_option = agent.active_option
            assert current_active_option is not None, "Current active option should never be None after a timestep"
            current_option_idx = agent._options.index(current_active_option)
            current_option_name = current_active_option.name
            current_option_initiated = prev_option_terminated
            current_option_terminated = not current_active_option.active
            prev_option_terminated = current_option_terminated
            
            # Handle task completion (check if block is aligned with target)
            # Success = cube is within 4cm of target position
            is_task_done = task_done(env, info)
            if done and is_task_done:
                tasks_completed_at_end += 1
                per_task_stats[task_id]['completed_at_end'] += 1
            if is_task_done and not episode_had_success:
                # Only count the first success (task completed once)
                tasks_completed_at_all += 1
                per_task_stats[task_id]['completed_at_all'] += 1
                episode_had_success = True
            
            # Store data
            dataset['option_indices'].append(current_option_idx)
            dataset['option_names'].append(current_option_name)
            dataset['option_initiated'].append(current_option_initiated)
            dataset['option_terminated'].append(current_option_terminated)
            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            dataset['task_ids'].append(task_id)
            
            if ep_idx == 0 and args.save_first_episode_video:
                frame = add_text_overlay(env.render(), current_option_idx, current_option_name)
                episode_frames.append(frame)
            
            if args.render_realtime:
                render_frame_realtime(env, window_name, args.render_delay,
                                      current_option_idx, current_option_name)
            
            ob = next_ob
            step += 1
        assert step == args.max_episode_steps, "Each episode should last its full length"
        
        # Track episode return
        episode_returns.append(episode_return)
        per_task_stats[task_id]['episode_returns'].append(episode_return)
        
        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step
        
        # Save first episode video
        if (args.save_first_episode_video and args.save_path is not None 
            and ep_idx == 0 and episode_frames):
            save_base = pathlib.Path(args.save_path)
            save_episode_video(
                episode_frames,
                save_dir=save_base.parent.as_posix(),
                filename=f"inference_{save_base.stem}_first_episode",
                fps=30,
            )
    
    # Print statistics
    total_episodes = num_train_episodes + num_val_episodes
    print(f'\n=== Statistics ===')
    policy_str = ""
    if args.agent_type == "hierarchical_ppo":
        # PPO-specific: deterministic vs stochastic with temperature
        policy_str = "(PPO: deterministic)" if args.deterministic else f"(PPO: stochastic, temp={args.temperature})"
    elif args.agent_type == "hierarchical_dqn":
        policy_str = f"(DQN: epsilon-greedy, epsilon={args.end_e})"
    print(f'Agent type: {args.agent_type} {policy_str}')
    if args.checkpoint_path:
        print(f'Checkpoint path: {args.checkpoint_path}')
    print(f'Total steps: {total_steps}')
    print(f'Total episodes: {total_episodes}')
    print(f'Average episode return: {np.mean(episode_returns):.2f}')
    print(f'Success rate (tasks completed at end of episode): {tasks_completed_at_end}/{tasks_attempted} ({100*tasks_completed_at_end/tasks_attempted:.1f}%)')
    print(f'Completion rate (tasks completed at any point in the episode): {tasks_completed_at_all}/{tasks_attempted} ({100*tasks_completed_at_all/tasks_attempted:.1f}%)')
    
    # Per-task breakdown
    print(f'\nPer-task statistics:')
    for task_id in sorted(per_task_stats.keys()):
        stats = per_task_stats[task_id]
        task_name = env.unwrapped.task_infos[task_id - 1]['task_name']
        avg_return = np.mean(stats['episode_returns']) if stats['episode_returns'] else 0.0
        print(f'  Task {task_id} ({task_name}):')
        print(f'    Average return: {avg_return:.2f}')
        print(f'    Completed at end: {stats["completed_at_end"]}/{stats["attempted"]} ({100*stats["completed_at_end"]/stats["attempted"]:.1f}%)')
        print(f'    Completed at any point: {stats["completed_at_all"]}/{stats["attempted"]} ({100*stats["completed_at_all"]/stats["attempted"]:.1f}%)')
    
    # Save dataset
    train_path = args.save_path.replace('.npz', '-train.npz')
    val_path = args.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        elif k in ['option_indices', 'task_ids']:
            dtype = np.int32
        elif k == 'option_names':
            dtype = object
        elif k in ['option_initiated', 'option_terminated']:
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)
    
    for path, ds in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **ds)
        print(f'Saved dataset to: {path}')
    
    # Cleanup
    env.close()
    if args.render_realtime:
        cleanup_realtime_rendering()


if __name__ == '__main__':
    main()
