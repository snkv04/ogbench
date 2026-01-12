"""Trains an RL agent for the cube environment using the hierarchical RL framework and the PPO algorithm."""

import json
import os
import random
import time
from datetime import datetime
# from absl import logging
# logging.set_verbosity(logging.INFO)
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro

import ogbench.manipspace  # Register environments
from ogbench.manipspace.oracles.hierarchical.learned_hierarchical_agent import (
    PolicyNetwork,
    LearnedHierarchicalAgent,
)

torch.set_float32_matmul_precision("high")


@dataclass
class Args:
    seed: int = 1048596
    torch_deterministic: bool = True
    cuda: bool = True
    track_with_wandb: bool = False
    wandb_project_name: str = "ppo_cube_hierarchical"
    wandb_entity: Optional[str] = None

    # Agent-specific arguments
    disable_no_op: bool = False
    no_op_duration: int = 10
    treat_options_as_one_step: bool = False  # If True, option reward = last step reward, GAE uses single-step discounting

    # Algorithm-specific arguments
    env_id: str = "cube-single-v0"
    task_id: int = 0  # Fixed task ID for all episodes (0 = default task)
    total_timesteps: int = 1000000
    learning_rate: float = 1e-3
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.98
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    episodes_per_rollout: int = 2
    backward_discounting: bool = False

    # Saving and loading
    save_dir: str = ".ogbench/ppo_runs"
    checkpoint_freq: int = 100
    save_model: bool = True
    run_name: str = ""
    load_path: str = ""

    # Visualization
    render_realtime: bool = False
    render_delay: float = 0.001

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    max_episode_steps: int = 0


def add_text_overlay(
    frame: np.ndarray,
    option_idx: Optional[int] = None,
    option_text: Optional[str] = None,
    font_scale: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Add HRL option info as text overlay on frame.
    
    Args:
        frame: RGB frame (numpy array).
        option_idx: Option index to display.
        option_text: Option name/description to display.
        font_scale: Font scale for text.
        thickness: Thickness of text.
    
    Returns:
        Frame with text overlay (copy of original).
    """
    if option_idx is None and option_text is None:
        return frame
    
    frame = frame.copy()  # Don't modify original
    
    # Build text string
    if option_idx is not None and option_text is not None:
        text = f"Option {option_idx}: {option_text}"
    elif option_idx is not None:
        text = f"Option {option_idx}"
    else:
        text = option_text
    
    # Draw text with black outline for visibility
    position = (10, 30)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), thickness + 2)  # Black outline
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)  # White text
    
    return frame


def render_frame_realtime(
    env,
    window_name: str,
    delay: float,
    option_idx: Optional[int] = None,
    option_text: Optional[str] = None,
):
    """Render a frame in real-time using OpenCV.
    
    Args:
        env: The gymnasium environment.
        window_name: Name of the OpenCV window.
        delay: Time to sleep after rendering (seconds).
        option_idx: Optional option index to display as overlay.
        option_text: Optional option name to display as overlay.
    """
    frame = env.render()
    frame = add_text_overlay(frame, option_idx, option_text)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)
    time.sleep(delay)


def init_realtime_rendering(window_name: str, width: int = 2000, height: int = 2000):
    """Initialize OpenCV window for real-time rendering.
    
    Args:
        window_name: Name of the OpenCV window.
        width: Window width in pixels.
        height: Window height in pixels.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)


def cleanup_realtime_rendering():
    """Cleanup OpenCV windows."""
    cv2.destroyAllWindows()


def rollout(
    env,
    agent: LearnedHierarchicalAgent,
    ob,
    info,
    num_steps: int,
    gamma: float = 0.98,
    treat_options_as_one_step: bool = False,
    render_realtime: bool = False,
    render_window_name: str = "Training",
    render_delay: float = 0.05,
) -> Tuple[List[dict], List[dict], object, object]:
    """Collect rollout data for high-level transitions."""
    hl_transitions = []
    episode_stats = []
    episode_return = 0.0  # Track cumulative return for current episode
    current_hl_info = None
    
    # Reset agent to clear any active option from previous rollout
    # This ensures each rollout starts with a fresh option selection
    agent.reset(ob, info)
    
    # Initial render
    if render_realtime:
        render_frame_realtime(env, render_window_name, render_delay)

    for step in range(num_steps):
        # Check if we need a new high-level action
        if agent.active_option is None or not agent.active_option.active:
            # Store previous high-level transition
            if current_hl_info is not None:
                hl_transitions.append({
                    'obs': current_hl_info['obs'],
                    'action': current_hl_info['action'],
                    'logprob': current_hl_info['logprob'],
                    'value': current_hl_info['value'],
                    'reward': current_hl_info['accumulated_reward'],
                    'option_length': current_hl_info['option_length'],
                    'start_step': current_hl_info['start_step'],
                    'end_step': step - 1,  # Previous option ended on previous step
                    'done': False,
                })

            # Select new option (agent stores decision in agent.last_decision)
            option = agent.select_high_level_action(ob, info)
            agent._active_option = option
            option.initiate(ob, info)

            current_hl_info = {
                **agent.last_decision,
                'accumulated_reward': 0.0,
                'option_length': 0,
                'start_step': step,
            }

        # Execute low-level action from active option
        low_level_action = agent.active_option.select_action(ob, info)
        agent.active_option.step()  # Increment step counter
        next_ob, reward, terminated, truncated, next_info = env.step(low_level_action)
        # print(f"reward = {reward}")
        done = terminated or truncated
        assert terminated == False and (truncated == False or step % args.max_episode_steps == args.max_episode_steps - 1), "Each episode should last its full length"

        # Render frame if enabled, before the option is set to inactive
        if render_realtime:
            render_frame_realtime(
                env, render_window_name, render_delay,
                option_idx=agent._options.index(agent._active_option),
                option_text=agent._active_option.name,
            )

        # Check if option should terminate
        if agent.active_option.is_terminated(next_ob, next_info):
            agent.active_option.reset()

        # Accumulate reward for the option
        if treat_options_as_one_step:
            # Accumulate non-discounted reward across option steps
            current_hl_info['accumulated_reward'] += reward
        else:
            if args.backward_discounting:
                # Perform backward discounting of reward across option steps
                current_hl_info['accumulated_reward'] = reward + gamma * current_hl_info['accumulated_reward']
            else:
                # Perform forward discounting of reward across option steps
                current_hl_info['accumulated_reward'] += (gamma ** current_hl_info['option_length']) * reward
        current_hl_info['option_length'] += 1
        episode_return += reward

        if done:
            # print(f"\nAt step {step}, episode is done!")
            # print(f"success = {next_info.get('success', False)}")
            # print(f"target_pos = {agent._target_pos}, current_block_pos = {next_info[f'privileged/block_{args.task_id}_pos']}")
            
            # Store final transition for this episode
            if current_hl_info is not None:
                hl_transitions.append({
                    'obs': current_hl_info['obs'],
                    'action': current_hl_info['action'],
                    'logprob': current_hl_info['logprob'],
                    'value': current_hl_info['value'],
                    'reward': current_hl_info['accumulated_reward'],
                    'option_length': current_hl_info['option_length'],
                    'start_step': current_hl_info['start_step'],
                    'end_step': step,  # Current option ended on current step
                    'done': True,
                })
                current_hl_info = None

            episode_stats.append({
                'return': episode_return,
                'success': next_info.get('success', False),
            })
            episode_return = 0.0  # Reset for next episode

            # Reset environment and agent
            ob, info = env.reset()
            agent.reset(ob, info)
            # print(f"env just resetted, target_pos = {agent._target_pos}, initial_block_pos = {info[f'privileged/block_{args.task_id}_pos']}")
        else:
            ob, info = next_ob, next_info

    # Handle remaining transition at end of rollout
    if current_hl_info is not None and current_hl_info['option_length'] > 0:
        hl_transitions.append({
            'obs': current_hl_info['obs'],
            'action': current_hl_info['action'],
            'logprob': current_hl_info['logprob'],
            'value': current_hl_info['value'],
            'reward': current_hl_info['accumulated_reward'],
            'option_length': current_hl_info['option_length'],
            'start_step': current_hl_info['start_step'],
            'end_step': num_steps - 1,  # Current option ended on last step of rollout
            'done': False,
        })

    return hl_transitions, episode_stats, ob, info


def compute_gae(
    transitions: List[dict],
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    treat_options_as_one_step: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns."""
    n = len(transitions)

    obs = torch.stack([t['obs'] for t in transitions])
    actions = torch.stack([t['action'] for t in transitions])
    logprobs = torch.stack([t['logprob'] for t in transitions])
    rewards = torch.tensor([t['reward'] for t in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([t['done'] for t in transitions], dtype=torch.float32, device=device)
    option_lengths = torch.tensor([t['option_length'] for t in transitions], dtype=torch.float32, device=device)
    values = torch.cat([t['value'] for t in transitions])

    advantages = torch.zeros(n, device=device)
    lastgaelam = 0

    for t in reversed(range(n)):        
        if t == n - 1:
            next_exists = 1.0 - dones[t]
            # Uses next_value only if the rollout ended where an episode didn't end
            nextvalue = next_value if not dones[t] else torch.zeros(1, device=device)
        else:
            next_exists = 1.0 - dones[t]
            nextvalue = values[t + 1]
        # print(f"t = {t}, start_step = {transitions[t]['start_step']}, end_step = {transitions[t]['end_step']}, option_length = {option_lengths[t]}, next_exists = {next_exists}, nextvalue = {nextvalue}")

        if treat_options_as_one_step:
            # Use standard single-step discounting (ignore option length)
            gamma_to_k = gamma
        else:
            # Use option-length-based discounting
            gamma_to_k = gamma ** option_lengths[t]
        delta = rewards[t] + gamma_to_k * nextvalue * next_exists - values[t]
        advantages[t] = lastgaelam = delta + gamma_to_k * gae_lambda * next_exists * lastgaelam
        # advantages[t] = lastgaelam = delta + gamma_to_k * (gae_lambda ** option_lengths[t]) * next_exists * lastgaelam

    returns = advantages + values
    return obs, actions, logprobs, values, advantages, returns


def update(
    policy_network: PolicyNetwork,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    args: Args,
) -> dict:
    """Perform PPO update epochs."""
    batch_size = obs.shape[0]
    clipfracs = []
    grad_norms_pre = []
    grad_norms_post = []

    for epoch in range(args.update_epochs):
        b_inds = torch.randperm(batch_size, device=obs.device)

        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            if len(mb_inds) == 0:
                continue

            _, newlogprob, entropy, newvalue = policy_network.get_action_and_value(
                obs[mb_inds], actions[mb_inds]
            )
            logratio = newlogprob - logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_advantages = advantages[mb_inds]
            if args.norm_adv and len(mb_advantages) > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            
            # Compute pre-clipped gradient norm
            total_norm_pre = 0.0
            for p in policy_network.parameters():
                if p.grad is not None:
                    total_norm_pre += p.grad.data.norm(2).item() ** 2
            grad_norms_pre.append(total_norm_pre ** 0.5)
            
            nn.utils.clip_grad_norm_(policy_network.parameters(), args.max_grad_norm)
            
            # Compute post-clipped gradient norm
            total_norm_post = 0.0
            for p in policy_network.parameters():
                if p.grad is not None:
                    total_norm_post += p.grad.data.norm(2).item() ** 2
            grad_norms_post.append(total_norm_post ** 0.5)
            
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    # Compute explained variance: 1 - Var(returns - values) / Var(returns)
    y_pred = values.detach()
    y_true = returns.detach()
    var_y = torch.var(y_true)
    explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else torch.tensor(0.0)

    return {
        'pg_loss': pg_loss.item(),
        'v_loss': v_loss.item(),
        'entropy': entropy_loss.item(),
        'approx_kl': approx_kl.item(),
        'clipfrac': np.mean(clipfracs) if clipfracs else 0,
        'explained_var': explained_var.item(),
        'grad_norm_pre': np.mean(grad_norms_pre) if grad_norms_pre else 0,
        'grad_norm_post': np.mean(grad_norms_post) if grad_norms_post else 0,
    }


def make_env(env_id: str, seed: int, max_episode_steps: int, task_id: int):
    env = gym.make(
        env_id,
        mode='task',
        terminate_at_goal=False,
        max_episode_steps=max_episode_steps,
        reward_task_id=task_id,  # Fixed task for all episodes
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def save_checkpoint(
    iteration: int,
    global_step: int,
    policy_network: PolicyNetwork,
    optimizer: optim.Optimizer,
    args: Args,
    avg_returns: deque,
    avg_successes: deque,
    training_metrics: List[dict],
    save_path: str,
) -> None:
    checkpoint = {
        "iteration": iteration,
        "global_step": global_step,
        "model_state_dict": policy_network.state_dict(),
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "Only one environment is supported for hierarchical PPO at the moment"

    # Compute derived values
    args.max_episode_steps = args.num_steps // args.episodes_per_rollout
    assert args.max_episode_steps > 0, "Cannot have more episodes than steps in each rollout"
    if args.num_steps % args.episodes_per_rollout != 0:
        print(f"WARNING: num_steps ({args.num_steps} steps per rollout) is not divisible by episodes_per_rollout ({args.episodes_per_rollout}). "
              f"Each rollout will have {args.max_episode_steps * args.episodes_per_rollout} steps instead "
              f"({(args.num_steps - args.max_episode_steps * args.episodes_per_rollout)} fewer).")
        args.num_steps = args.max_episode_steps * args.episodes_per_rollout
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}__{timestamp}" if args.run_name else f"{args.env_id}__{args.seed}__{timestamp}"
    assert args.batch_size % args.num_minibatches == 0, "Batch size must be divisible by num_minibatches"
    # if args.batch_size % args.num_minibatches != 0:
    #     print(f"WARNING: batch_size ({args.batch_size}) is not divisible by num_minibatches ({args.num_minibatches}). "
    #           f"Last minibatch will have {args.batch_size % args.minibatch_size} samples instead of {args.minibatch_size}.")
    if args.total_timesteps % args.batch_size != 0:
        actual_timesteps = args.num_iterations * args.batch_size
        print(f"WARNING: total_timesteps ({args.total_timesteps}) is not divisible by batch_size ({args.batch_size}). "
              f"Will train for {actual_timesteps} timesteps instead ({args.total_timesteps - actual_timesteps} fewer).")

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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env = make_env(args.env_id, args.seed, args.max_episode_steps, args.task_id)
    print(f"Using fixed task_id={args.task_id} for all episodes")

    # Initialize real-time rendering if enabled
    render_window_name = "PPO Training - Real-time Rendering"
    if args.render_realtime:
        init_realtime_rendering(render_window_name)

    # Policy network (owned by script, shared with agent)
    policy_network = PolicyNetwork(
        LearnedHierarchicalAgent.OBS_DIM,
        10 if not args.disable_no_op else 9,
        hidden_dim=256,
        device=device,
    )
    optimizer = optim.Adam(policy_network.parameters(), lr=args.learning_rate, eps=1e-5)

    # Load from checkpoint if specified
    start_iteration = 1
    initial_global_step = 0
    global_step = 0
    avg_returns = deque(maxlen=100)
    avg_successes = deque(maxlen=100)
    training_metrics = []
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_path}")
        print(f"Loading checkpoint from: {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        policy_network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint["iteration"] + 1
        initial_global_step = checkpoint["global_step"]
        global_step = checkpoint["global_step"]
        
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
            loaded_avg_returns = checkpoint["avg_returns"]
            avg_returns.extend(loaded_avg_returns)
        if "avg_successes" in checkpoint:
            loaded_avg_successes = checkpoint["avg_successes"]
            avg_successes.extend(loaded_avg_successes)
        
        # Restore training metrics history
        if "training_metrics" in checkpoint:
            training_metrics = checkpoint["training_metrics"]
        
        print(f"Resuming from iteration {start_iteration} (global_step={global_step})")

    # Hierarchical agent (holds reference to policy network)
    ob, info = env.reset(seed=args.seed)
    agent = LearnedHierarchicalAgent(
        env, policy_network, device,
        disable_no_op=args.disable_no_op,
        no_op_duration=args.no_op_duration,
    )
    agent.reset(ob, info)

    # Iterates through rollouts
    pbar = tqdm.tqdm(range(start_iteration, start_iteration + args.num_iterations), desc="Training")
    start_time = time.time()
    for iteration in pbar:
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Collect rollout
        transitions, episode_stats, ob, info = rollout(
            env, agent, ob, info, args.num_steps,
            gamma=args.gamma,
            treat_options_as_one_step=args.treat_options_as_one_step,
            render_realtime=args.render_realtime,
            render_window_name=render_window_name,
            render_delay=args.render_delay,
        )
        global_step += args.num_steps

        # Track episode stats
        for stat in episode_stats:
            avg_returns.append(stat['return'])
            avg_successes.append(float(stat['success']))

        # Skip update if insufficient transitions
        if len(transitions) < 2:
            continue

        # Compute advantages
        with torch.no_grad():
            if not transitions[-1]['done']:
                next_obs = agent.get_obs_tensor(info).unsqueeze(0)
                next_value = policy_network.get_value(next_obs).flatten()
            else:
                next_value = torch.zeros(1, device=device)
        obs, actions, logprobs, values, advantages, returns = compute_gae(
            transitions, next_value, args.gamma, args.gae_lambda, device,
            treat_options_as_one_step=args.treat_options_as_one_step,
        )

        # Update policy
        losses = update(policy_network, optimizer, obs, actions, logprobs, values, advantages, returns, args)

        # Logging
        sps = int((global_step - initial_global_step) / (time.time() - start_time))
        avg_ret = np.mean(avg_returns) if avg_returns else 0
        avg_suc = np.mean(avg_successes) if avg_successes else 0
        print(
            f"Steps per second: {sps}, Return: {avg_ret:.2f}, Success: {avg_suc:.2%}, "
            f"High-level transitions: {len(transitions)}, Loss: {losses['pg_loss']:.4f}"
        )
        
        # Track metrics
        metrics = {
            "iteration": iteration,
            "global_step": global_step,
            "sps": sps,
            "avg_episode_return": float(avg_ret),
            "episode_returns": [stat['return'] for stat in episode_stats],  # Individual episode returns
            "success_rate": float(avg_suc),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "policy_loss": float(losses['pg_loss']),
            "value_loss": float(losses['v_loss']),
            "entropy": float(losses['entropy']),
            "approx_kl": float(losses['approx_kl']),
            "clipfrac": float(losses['clipfrac']),
            "explained_var": float(losses['explained_var']),
            "grad_norm_pre": float(losses['grad_norm_pre']),
            "grad_norm_post": float(losses['grad_norm_post']),
            "high_level_transitions": len(transitions),
        }
        training_metrics.append(metrics)

        # Logging for each training iteration
        if args.track_with_wandb:
            # Log individual episode returns
            for i, stat in enumerate(episode_stats):
                prev_global_step = global_step - args.num_steps
                episode_step = prev_global_step + (i + 1) * args.max_episode_steps
                wandb.log({
                    f"charts/episode_returns": stat['return'],
                }, step=episode_step)

            # Log aggregated metrics
            wandb.log({
                # (Time)steps per second
                "charts/SPS": sps,
                "charts/avg_episode_return": avg_ret,
                "charts/success_rate": avg_suc,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/policy_loss": losses['pg_loss'],
                "losses/value_loss": losses['v_loss'],
                # How much the policy distribution is spread out over options
                "losses/entropy": losses['entropy'],
                # Approximation of the KL divergence between the new and old policy
                "losses/approx_kl": losses['approx_kl'],
                # Proportion of timesteps in the rollout where the policy ratio was clipped
                "losses/clipfrac": losses['clipfrac'],
                # Amount of variance in returns explained by values from critic
                "losses/explained_var": losses['explained_var'],
                "losses/grad_norm_pre": losses['grad_norm_pre'],
                "losses/grad_norm_post": losses['grad_norm_post'],
                "rollout/high_level_transitions": len(transitions),
            }, step=global_step)

        # Save checkpoint
        if args.save_model and iteration % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_iter{iteration}.pt")
            save_checkpoint(
                iteration=iteration,
                global_step=global_step,
                policy_network=policy_network,
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
        final_model_path = os.path.join(save_path, f"final_model_iter{iteration}.pt")
        assert iteration == start_iteration + args.num_iterations - 1, "Iteration count mismatch"
        save_checkpoint(
            iteration=iteration,
            global_step=global_step,
            policy_network=policy_network,
            optimizer=optimizer,
            args=args,
            avg_returns=avg_returns,
            avg_successes=avg_successes,
            training_metrics=training_metrics,
            save_path=final_model_path
        )
        print(f"Final model (after {iteration} iterations) has been saved to {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(save_path, f"training_metrics_iter{iteration}.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    print(f"\nTraining complete!")
    print(f"Final average return: {np.mean(avg_returns):.2f}")
    print(f"Final success rate: {np.mean(avg_successes):.2%}")
