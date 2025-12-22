"""Trains a hierarchical RL agent for the cube environment using PPO."""

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

    # Algorithm specific arguments
    env_id: str = "cube-single-v0"
    task_id: int = 0  # Fixed task ID for all episodes (0 = default task)
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
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

    # Saving
    save_dir: str = ".ogbench/ppo_runs"
    checkpoint_freq: int = 100
    save_model: bool = True

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
                    'done': False,
                })

            # Select new option (agent stores decision in agent.last_decision)
            option = agent.select_high_level_action(ob, info)
            agent._active_option = option
            option.initiate(ob, info)

            current_hl_info = {
                **agent.last_decision,
                'accumulated_reward': 0.0,
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

        current_hl_info['accumulated_reward'] += reward
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
    if current_hl_info is not None and current_hl_info['accumulated_reward'] != 0:
        hl_transitions.append({
            'obs': current_hl_info['obs'],
            'action': current_hl_info['action'],
            'logprob': current_hl_info['logprob'],
            'value': current_hl_info['value'],
            'reward': current_hl_info['accumulated_reward'],
            'done': False,
        })

    return hl_transitions, episode_stats, ob, info


def compute_gae(
    transitions: List[dict],
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns."""
    n = len(transitions)

    obs = torch.stack([t['obs'] for t in transitions])
    actions = torch.stack([t['action'] for t in transitions])
    logprobs = torch.stack([t['logprob'] for t in transitions])
    rewards = torch.tensor([t['reward'] for t in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([t['done'] for t in transitions], dtype=torch.float32, device=device)
    values = torch.cat([t['value'] for t in transitions])

    advantages = torch.zeros(n, device=device)
    lastgaelam = 0

    for t in reversed(range(n)):
        if t == n - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = next_value if not dones[t] else torch.zeros(1, device=device)
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t + 1]

        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

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
            nn.utils.clip_grad_norm_(policy_network.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    return {
        'pg_loss': pg_loss.item(),
        'v_loss': v_loss.item(),
        'entropy': entropy_loss.item(),
        'approx_kl': approx_kl.item(),
        'clipfrac': np.mean(clipfracs) if clipfracs else 0,
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
    run_name = f"{args.env_id}__{args.seed}__{timestamp}"
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
        LearnedHierarchicalAgent.NUM_OPTIONS,
        hidden_dim=256,
        device=device,
    )
    optimizer = optim.Adam(policy_network.parameters(), lr=args.learning_rate, eps=1e-5)

    # Hierarchical agent (holds reference to policy network)
    ob, info = env.reset(seed=args.seed)
    agent = LearnedHierarchicalAgent(env, policy_network, device)
    agent.reset(ob, info)

    # Tracking
    avg_returns = deque(maxlen=100)
    avg_successes = deque(maxlen=100)
    global_step = 0
    start_time = time.time()
    training_metrics = []  # Store metrics for saving

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1), desc="Training")

    for iteration in pbar:
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Collect rollout
        transitions, episode_stats, ob, info = rollout(
            env, agent, ob, info, args.num_steps,
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
            transitions, next_value, args.gamma, args.gae_lambda, device
        )

        # Update policy
        losses = update(policy_network, optimizer, obs, actions, logprobs, values, advantages, returns, args)

        # Logging
        sps = int(global_step / (time.time() - start_time))
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
            "episode_return": float(avg_ret),
            "success_rate": float(avg_suc),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "policy_loss": float(losses['pg_loss']),
            "value_loss": float(losses['v_loss']),
            "entropy": float(losses['entropy']),
            "approx_kl": float(losses['approx_kl']),
            "clipfrac": float(losses['clipfrac']),
            "high_level_transitions": len(transitions),
        }
        training_metrics.append(metrics)

        if args.track_with_wandb:
            wandb.log({
                "charts/SPS": sps,
                "charts/episode_return": avg_ret,
                "charts/success_rate": avg_suc,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/policy_loss": losses['pg_loss'],
                "losses/value_loss": losses['v_loss'],
                "losses/entropy": losses['entropy'],
                "losses/approx_kl": losses['approx_kl'],
                "losses/clipfrac": losses['clipfrac'],
                "rollout/high_level_transitions": len(transitions),
            }, step=global_step)

        # Save checkpoint
        if args.save_model and iteration % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_iter{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": policy_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

    # Cleanup
    env.close()
    if args.render_realtime:
        cleanup_realtime_rendering()
    if args.track_with_wandb:
        wandb.finish()

    # Save final model
    if args.save_model:
        final_model_path = os.path.join(save_path, f"final_model_iter{iteration}.pt")
        assert iteration == args.num_iterations, "Iteration count does not match num_iterations"
        torch.save({
            "iteration": iteration,
            "global_step": global_step,
            "model_state_dict": policy_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, final_model_path)
        print(f"Saved final model (after {iteration} iterations) to {final_model_path}")

    # Save training metrics
    metrics_path = os.path.join(save_path, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    print(f"\nTraining complete!")
    print(f"Final average return: {np.mean(avg_returns):.2f}")
    print(f"Final success rate: {np.mean(avg_successes):.2%}")
