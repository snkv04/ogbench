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
        args.save_path = f".ogbench/data/{args.env_name}-learned-{policy_type}-{timestamp}.npz"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint trained for {checkpoint['iteration']} iterations, {checkpoint['global_step']} steps")
    
    # Initialize environment
    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=args.max_episode_steps,
    )
    
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
    
    tasks_completed = 0  # Number of tasks where cube reached target
    tasks_attempted = 0  # Total tasks attempted (1 per episode start + 1 per set_new_target)
    episodes_with_success = 0  # Episodes where at least one task was completed
    
    # Initialize window for real-time rendering if enabled
    window_name = 'Learned Policy - Real-time Rendering'
    if args.render_realtime:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print(f"\nGenerating {num_train_episodes + num_val_episodes} episodes...")
    print(f"  Train: {num_train_episodes}, Val: {num_val_episodes}")
    print(f"  Deterministic: {args.deterministic}, Temperature: {args.temperature}")
    print(f"  Action noise: {args.noise}")
    
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        ob, info = env.reset()
        agent.reset(ob, info)
        
        if ep_idx == 0 and args.save_first_episode_video:
            episode_frames = [env.render()]
        
        if args.render_realtime:
            render_frame_realtime(env, window_name, args.render_delay)
        
        # Track option state
        prev_option_terminated = True
        
        done = False
        step = 0
        episode_had_success = False
        tasks_attempted += 1  # Each episode starts with one task
        
        while not done:
            # Get action from learned policy (via hierarchical agent)
            action = agent.select_action(ob, info)
            print(f"active option = {agent.active_option.name if agent.active_option is not None else 'none'}")
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
            # block_pos = info[f'privileged/block_{agent._target_block}_pos']
            # target_pos = info['privileged/target_block_pos']
            # dist = np.linalg.norm(target_pos - block_pos)
            # print(f"Step {step}: cube-target dist = {dist:.4f}m, done = {agent.done}")
            if agent.done:
                tasks_completed += 1
                episode_had_success = True
                
                # Set new target for next task within same episode
                agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=0.0)
                agent.reset(agent_ob, agent_info)
                tasks_attempted += 1  # New task started
                option_terminated = True
                prev_option_terminated = True
            
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
            
            if ep_idx == 0 and args.save_first_episode_video:
                episode_frames.append(env.render())
            
            if args.render_realtime:
                render_frame_realtime(env, window_name, args.render_delay)
            
            ob = next_ob
            step += 1
        
        total_steps += step
        if episode_had_success:
            episodes_with_success += 1
        
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
    print(f'Task success rate: {tasks_completed}/{tasks_attempted} ({100*tasks_completed/tasks_attempted:.1f}%)')
    print(f'  - Tasks completed: {tasks_completed} (cube placed within 4cm of target)')
    print(f'  - Tasks attempted: {tasks_attempted} (includes new targets after each success)')
    print(f'Episodes with at least one success: {episodes_with_success}/{total_episodes} ({100*episodes_with_success/total_episodes:.1f}%)')
    
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
        elif k == 'option_indices':
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
