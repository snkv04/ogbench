"""Run inference, and generate a dataset, using a trained hierarchical PPO policy."""

import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import gymnasium
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from tqdm import trange

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.hierarchical.learned_hierarchical_agent import (
    PolicyNetwork,
    LearnedHierarchicalAgent,
)


@dataclass
class Args:
    # Model loading
    checkpoint_path: str = ""  # Path to checkpoint file (required)
    
    # Environment
    env_name: str = "cube-single-v0"
    seed: int = 0
    task_id: Optional[int] = None  # Task ID (1-5 for cube-single). None = sample randomly
    terminate_at_goal: bool = False  # End episode when goal reached
    
    # Dataset generation
    num_episodes: int = 1000
    max_episode_steps: int = 1001
    save_path: Optional[str] = None  # If None, auto-generated
    
    # Action noise (optional, for diversity)
    noise: float = 0.0
    
    # Policy behavior
    deterministic: bool = False  # If True, use argmax instead of sampling
    temperature: float = 1.0  # Temperature for sampling (higher = more random)
    
    # Visualization
    save_first_episode_video: bool = False
    render_realtime: bool = False
    render_delay: float = 0.05
    
    # Device
    cuda: bool = True


def render_frame_realtime(env, window_name, delay):
    """Render a frame in real-time."""
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)
    time.sleep(delay)


def main():
    args = tyro.cli(Args)
    
    assert args.checkpoint_path, "Must provide --checkpoint_path"
    
    # Set save path
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_type = "deterministic" if args.deterministic else f"temp{args.temperature}"
        task_str = f"task{args.task_id}" if args.task_id else "alltasks"
        args.save_path = f".ogbench/data/{args.env_name}-{task_str}-{policy_type}-{timestamp}.npz"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint trained for {checkpoint['iteration']} iterations, {checkpoint['global_step']} steps")
    
    # Initialize environment in task mode
    # task_id=None means sample from all tasks, task_id=1-5 means use that specific task
    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=args.terminate_at_goal,
        mode='task',
        reward_task_id=args.task_id,  # None = sample, 1-5 = specific task
        max_episode_steps=args.max_episode_steps,
    )
    
    # Print task info
    if args.task_id:
        print(f"Using fixed task: task{args.task_id}")
    else:
        print(f"Sampling from all {len(env.unwrapped.task_infos)} tasks")
    
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
    
    # Initialize agent
    ob, info = env.reset(seed=args.seed)
    agent = LearnedHierarchicalAgent(
        env, policy_network, device,
        deterministic=args.deterministic,
        temperature=args.temperature
    )
    agent.reset(ob, info)
    
    # Collect data
    dataset = defaultdict(list)
    episode_frames = []
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = args.num_episodes
    num_val_episodes = args.num_episodes // 10
    
    # Task mode statistics
    episodes_succeeded = 0  # Episodes where at least one task succeeded
    episodes_failed = 0  # Episodes with no successes
    tasks_completed = 0  # Total tasks completed across all episodes
    tasks_attempted = 0  # Total tasks attempted across all episodes
    per_task_stats = defaultdict(lambda: {'attempted': 0, 'succeeded': 0})
    
    # Initialize window for real-time rendering if enabled
    window_name = 'Learned Policy - Real-time Rendering'
    if args.render_realtime:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print(f"\nGenerating {num_train_episodes + num_val_episodes} episodes...")
    print(f"  Train: {num_train_episodes}, Val: {num_val_episodes}")
    print(f"  Mode: task, terminate_at_goal: {args.terminate_at_goal}")
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
        
        if ep_idx == 0 and args.save_first_episode_video:
            episode_frames = [env.render()]
        
        if args.render_realtime:
            render_frame_realtime(env, window_name, args.render_delay)
        
        # Track option state
        prev_option_terminated = True
        
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
            
            # Track option info
            current_active_option = agent.active_option
            if current_active_option is not None:
                option_idx = agent._options.index(current_active_option)
                option_name = current_active_option.name
                option_initiated = prev_option_terminated
                option_terminated = not current_active_option.active
                prev_option_terminated = option_terminated
            else:
                option_idx = -1
                option_name = "none"
                option_initiated = False
                option_terminated = False
            
            # Handle task completion (agent.done means task was successful)
            # Success = cube is within 4cm of target position
            did_reset = False
            if agent.done:
                tasks_completed += 1
                per_task_stats[task_id]['succeeded'] += 1
                episode_had_success = True
                
                # Set new target for next task within same episode
                new_task_id = np.random.randint(1, len(env.unwrapped.task_infos) + 1)
                while new_task_id == task_id:
                    new_task_id = np.random.randint(1, len(env.unwrapped.task_infos) + 1)
                ob, info = env.reset(options={'task_id': new_task_id})
                agent.reset(ob, info)
                task_id = new_task_id  # Update current task_id
                per_task_stats[task_id]['attempted'] += 1
                tasks_attempted += 1  # New task started
                option_terminated = True
                prev_option_terminated = True
                did_reset = True
            
            # Store data
            dataset['option_indices'].append(option_idx)
            dataset['option_names'].append(option_name)
            dataset['option_initiated'].append(option_initiated)
            dataset['option_terminated'].append(option_terminated)
            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            dataset['task_ids'].append(task_id)
            
            if ep_idx == 0 and args.save_first_episode_video:
                episode_frames.append(env.render())
            
            if args.render_realtime:
                render_frame_realtime(env, window_name, args.render_delay)
            
            # Only update ob from step result if we didn't reset
            if not did_reset:
                ob = next_ob
            step += 1
        
        # Track episode outcome
        # episodes_succeeded = episodes with at least one task completed
        if episode_had_success:
            episodes_succeeded += 1
        else:
            episodes_failed += 1
        
        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step
        
        # Save first episode video
        if (args.save_first_episode_video and args.save_path is not None 
            and ep_idx == 0 and episode_frames):
            save_base = pathlib.Path(args.save_path)
            video_path = save_base.parent / f'{save_base.stem}_episode0.mp4'
            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(
                video_path.as_posix(),
                fps=30,
                codec='libx264',
                quality=8,
                macro_block_size=None,
            ) as writer:
                for frame in episode_frames:
                    writer.append_data(frame)
            print(f'\nSaved video of first episode to: {video_path.as_posix()}')
    
    # Print statistics
    total_episodes = num_train_episodes + num_val_episodes
    print(f'\n=== Statistics ===')
    print(f'Total steps: {total_steps}')
    print(f'Total episodes: {total_episodes}')
    print(f'Task success rate: {tasks_completed}/{tasks_attempted} ({100*tasks_completed/tasks_attempted:.1f}%)')
    print(f'Episodes with at least one success: {episodes_succeeded}/{total_episodes} ({100*episodes_succeeded/total_episodes:.1f}%)')
    
    # Per-task breakdown
    print(f'\nPer-task success rates:')
    for task_id in sorted(per_task_stats.keys()):
        stats = per_task_stats[task_id]
        task_name = env.unwrapped.task_infos[task_id - 1]['task_name']
        rate = 100 * stats['succeeded'] / stats['attempted'] if stats['attempted'] > 0 else 0
        print(f'  Task {task_id} ({task_name}): {stats["succeeded"]}/{stats["attempted"]} ({rate:.1f}%)')
    
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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
