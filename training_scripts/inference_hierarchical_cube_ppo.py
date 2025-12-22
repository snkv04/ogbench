"""
Runs inference, and generates a dataset, using an agent trained for the cube environment.

The loaded policy can be a hierarchical agent trained with PPO or a rule-based oracle.
"""

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
from ogbench.manipspace.oracles.hierarchical.learned_hierarchical_agent import (
    PolicyNetwork,
    LearnedHierarchicalAgent,
)
from ogbench.manipspace.oracles.hierarchical.cube_hierarchical import (
    CubeHierarchicalOracle,
)
from ogbench.manipspace.oracles.hierarchical.cube_hierarchical_random import (
    CubeHierarchicalRandom,
)
from training_scripts.train_hierarchical_cube_ppo import (
    render_frame_realtime,
    init_realtime_rendering,
    cleanup_realtime_rendering,
    add_text_overlay,
)


@dataclass
class Args:
    # Agent type
    agent_type: Literal["hierarchical_ppo", "hierarchical_oracle", "hierarchical_random"] = "hierarchical_ppo"
    checkpoint_path: str = ""  # Path to checkpoint file (required for hierarchical_ppo)
    deterministic: bool = False  # If True, use argmax instead of sampling
    temperature: float = 1.0  # Temperature for sampling (higher = more random)

    
    # Environment
    env_name: str = "cube-single-v0"
    seed: int = 1048596
    task_id: int = 0  # Fixed task ID for all episodes (0 = default task)
    
    # Dataset generation
    num_episodes: int = 1000
    max_episode_steps: int = 1024
    save_path: Optional[str] = None  # If None, auto-generated
    
    # Action noise (optional, for diversity)
    noise: float = 0.0
    
    # Visualization
    save_first_episode_video: bool = False
    render_realtime: bool = False
    render_delay: float = 0.05
    
    # Device
    cuda: bool = True


def task_done(env, info, threshold: float = 0.04) -> bool:
    # In task mode, target_block is always 0 for cube-single
    target_block = 0
    target_pos = env.unwrapped.cur_task_info['goal_xyzs'][target_block]
    block_pos = info[f'privileged/block_{target_block}_pos']
    return np.linalg.norm(target_pos - block_pos) <= threshold


def save_episode_video(
    frames: List[np.ndarray],
    save_dir: str,
    filename: str,
    fps: int = 30,
) -> str:
    """Save episode frames as a video file.
    
    Args:
        frames: List of RGB frames (numpy arrays).
        save_dir: Directory to save the video in.
        filename: Name of the video file (without extension).
        fps: Frames per second.
    
    Returns:
        Full path to the saved video.
    """
    if not frames:
        return ""
    
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    video_path = save_path / f"{filename}.mp4"
    
    with imageio.get_writer(
        video_path.as_posix(),
        fps=fps,
        codec='libx264',
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    print(f"Saved video to: {video_path.as_posix()}")
    return video_path.as_posix()


def main():
    args = tyro.cli(Args)
    
    if args.agent_type == "hierarchical_ppo":
        assert args.checkpoint_path, "Must provide --checkpoint_path for hierarchical_ppo agent"
    
    # Set save path
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_type = "deterministic" if args.deterministic else f"temp{args.temperature}"
        args.save_path = f".ogbench/data/{args.env_name}-{args.agent_type}-task{args.task_id}-{policy_type}-{timestamp}.npz"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.agent_type == "hierarchical_ppo":
        print(f"Using device: {device}")
    
    # Load checkpoint (only for PPO agent)
    checkpoint = None
    if args.agent_type == "hierarchical_ppo":
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        print(f"Checkpoint trained for {checkpoint['iteration']} iterations, {checkpoint['global_step']} steps")
    
    # Initialize environment in task mode with fixed task
    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode='task',
        reward_task_id=args.task_id,  # Fixed task for all episodes
        max_episode_steps=args.max_episode_steps,
    )
    print(f"Using fixed task_id={args.task_id} for all episodes")
    
    # Initialize agent based on agent_type
    ob, info = env.reset(seed=args.seed)
    if args.agent_type == "hierarchical_ppo":
        # Initialize policy network
        policy_network = PolicyNetwork(
            LearnedHierarchicalAgent.OBS_DIM,
            LearnedHierarchicalAgent.NUM_OPTIONS,
            hidden_dim=256,
            device=device,
        )
        policy_network.load_state_dict(checkpoint['model_state_dict'])
        policy_network.eval()
        print("Policy network loaded successfully")
        
        agent = LearnedHierarchicalAgent(
            env, policy_network, device,
            deterministic=args.deterministic,
            temperature=args.temperature
        )
    elif args.agent_type == "hierarchical_oracle":
        agent = CubeHierarchicalOracle(
            env=env,
            max_step=args.max_episode_steps,
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
    per_task_stats = defaultdict(lambda: {
        'attempted': 0,
        'completed_at_end': 0,
        'completed_at_all': 0,
    })
    
    # Initialize window for real-time rendering if enabled
    window_name = 'Learned Policy - Real-time Rendering'
    if args.render_realtime:
        init_realtime_rendering(window_name)
    
    print(f"\nGenerating {num_train_episodes + num_val_episodes} episodes...")
    print(f"  Agent type: {args.agent_type}")
    print(f"  Train: {num_train_episodes}, Val: {num_val_episodes}")
    if args.agent_type == "hierarchical_ppo":
        print(f"  Deterministic: {args.deterministic}, Temperature: {args.temperature}")
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
            done = terminated or truncated
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
    temperature_str = ""
    if args.agent_type == "hierarchical_ppo":
        temperature_str = f"deterministic" if args.deterministic else f"temperature {args.temperature}"
        temperature_str = f"({temperature_str})"
    print(f'Agent type: {args.agent_type} {temperature_str}')
    print(f'Total steps: {total_steps}')
    print(f'Total episodes: {total_episodes}')
    print(f'Success rate (tasks completed at end of episode): {tasks_completed_at_end}/{tasks_attempted} ({100*tasks_completed_at_end/tasks_attempted:.1f}%)')
    print(f'Completion rate (tasks completed at any point in the episode): {tasks_completed_at_all}/{tasks_attempted} ({100*tasks_completed_at_all/tasks_attempted:.1f}%)')
    
    # Per-task breakdown
    print(f'\nPer-task success rates:')
    for task_id in sorted(per_task_stats.keys()):
        stats = per_task_stats[task_id]
        task_name = env.unwrapped.task_infos[task_id - 1]['task_name']
        rate = 100 * stats['completed_at_end'] / stats['attempted'] if stats['attempted'] > 0 else 0
        print(f'  Task {task_id} ({task_name}):')
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
