"""Train high-level DQN over four TD3 directional options for ant maze."""

from __future__ import annotations

import json
import os

import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger as logging
from stable_baselines3.common.buffers import ReplayBuffer
import tqdm
import tyro
from torch.nn.utils import clip_grad_norm_

from hierarchical_training_scripts.hierarchical_antmaze_dqn_agent import (
    HierarchicalAntMazeDQNAgent,
    MoveInDirectionOption,
    NoOpOption,
)
from hierarchical_training_scripts.hierarchical_dqn_agent import QNetwork
from hierarchical_training_scripts.train_antmaze_lowlevel_td3 import run_maze_validation_episodes
from hierarchical_training_scripts.train_cube_hrl_dqn import (
    _log_param_counts,
    _prof_checkpoint,
    _save_profiling_bar_graph,
    _save_profiling_json,
    save_checkpoint,
)
from hierarchical_training_scripts.train_cube_lowlevel_td3 import Actor
from ogbench.locomaze.maze import make_maze_env
from ogbench.manipspace.oracles.hierarchical.option import Option

torch.set_float32_matmul_precision("high")


@dataclass
class Args:
    seeds: List[int] = field(default_factory=lambda: list(range(42, 47)))
    """List of random seeds to run sequentially. Each seed is a separate WandB run under the same group."""
    torch_deterministic: bool = True
    cuda: bool = True
    track_with_wandb: bool = False
    wandb_project_name: str = "dqn_antmaze_hierarchical"
    wandb_entity: Optional[str] = None
    log_freq: int = 100

    maze_type: str = "arena"
    max_episode_steps: int = 1000
    add_noise_to_init: bool = False
    add_noise_to_goal: bool = False
    reward_task_id: int = 0
    """Fixed task ID for MazeEnv. 0 uses the maze-type default task; positive int selects that task (1-indexed)."""
    goal_radius: Optional[float] = None
    """Success radius for goal reaching. If None, uses MazeEnv default (0.5 for ant)."""
    concatenate_only_xy: bool = True

    checkpoint_up: str = ""
    checkpoint_down: str = ""
    checkpoint_left: str = ""
    checkpoint_right: str = ""
    termination_time: int = 50
    """Max low-level env steps per option before forced termination."""

    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-3
    measure_burnin: int = 3
    episode_window_size: int = 100

    buffer_size: int = 100_000
    gamma: float = 0.98
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 4
    max_grad_norm: float = 1.0
    q_hidden_dim: int = 256

    reward_type: str = "sparse"
    """reward type: 'sparse' (binary success signal from MazeEnv), 'dense' (negative euclidean distance per timestep), or 'dense_on_options' (zero per-timestep reward; option transition reward is dense reward at the last timestep of the option)"""
    discount_inside_option: bool = True
    """If True, rewards within an option are discounted by gamma^t. If False, rewards are summed without discounting."""

    discretize_hl_obs: bool = True
    """If True, XY dims of the HL observation are replaced with per-axis bins + scalar displacement from bin center."""
    use_one_hot: bool = False
    """If True (and discretize_hl_obs=True), encode each axis as a one-hot vector + displacement (2*(num_bins+1) dims). If False, encode as [bin_idx, displacement] per axis (4 dims total)."""
    num_bins: int = 6
    """Number of bins per axis when discretize_hl_obs=True."""

    save_dir: str = ".ogbench/dqn_antmaze_hl_runs"
    checkpoint_freq: int = 100_000
    save_model: bool = True
    run_name: str = ""
    load_path: str = ""
    save_first_val_episodes_videos: int = 0

    validation_greedy: bool = True
    """If True, high-level DQN uses epsilon=0 (greedy discrete actions) during validation. If False, keeps the current training epsilon (same epsilon-greedy as the training step)."""

    run_profiling: bool = False
    profiling_start: int = 0


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def maze_task_done(env: gym.Env, info: dict) -> bool:
    """Whether the maze navigation task is successful (same signal as training metrics)."""
    return float(info.get("success", 0.0)) >= 1.0


def _dense_reward_from_env(env: gym.Env) -> float:
    """Negative Euclidean distance from agent xy to goal xy at the current env state."""
    agent_xy = env.unwrapped.get_xy()
    goal_xy = env.unwrapped.cur_goal_xy
    return -float(np.linalg.norm(agent_xy - goal_xy))


class AntMazeHLRewardWrapper(gym.Wrapper):
    """Applies sparse or dense reward shaping at the high level.

    sparse:          pass through the reward from MazeEnv unchanged.
    dense:           replace reward with negative Euclidean distance from agent xy to goal xy.
    dense_on_options: return 0 per timestep; option transition reward is set to the dense reward
                     at the last timestep of the option (handled in the training loop).
    """

    def __init__(self, env: gym.Env, reward_type: str = "sparse"):
        super().__init__(env)
        assert reward_type in ("sparse", "dense", "dense_on_options"), (
            f"reward_type must be 'sparse', 'dense', or 'dense_on_options', got '{reward_type}'"
        )
        self._reward_type = reward_type

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._reward_type == "dense":
            reward = _dense_reward_from_env(self)
        elif self._reward_type == "dense_on_options":
            reward = 0.0
        return obs, reward, terminated, truncated, info


def make_antmaze_hrl_env(args: Args) -> gym.Env:
    base_env = make_maze_env(
        "ant",
        "maze",
        maze_type=args.maze_type,
        terminate_at_goal=False,
        add_noise_to_init=args.add_noise_to_init,
        add_noise_to_goal=args.add_noise_to_goal,
        reward_task_id=args.reward_task_id,
        goal_radius=args.goal_radius,
    )
    base_env = AntMazeHLRewardWrapper(base_env, reward_type=args.reward_type)
    return gym.wrappers.TimeLimit(base_env, max_episode_steps=args.max_episode_steps)


def build_move_options(
    env: gym.Env,
    device: torch.device,
    checkpoint_paths: List[str],
    names: List[str],
    args: Args,
) -> List[Option]:
    n_act = int(np.prod(env.action_space.shape))
    base_dim = int(np.prod(env.observation_space.shape))
    prefix_dim = 2 if args.concatenate_only_xy else base_dim
    td3_obs_dim = prefix_dim + base_dim
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    no_op = NoOpOption(
        name="no_op",
        env=env,
        termination_time=args.termination_time,
        action_low=action_low,
        action_high=action_high,
    )
    options: List[Option] = [no_op]
    for name, ckpt in zip(names, checkpoint_paths):
        if not ckpt:
            raise ValueError(f"Missing checkpoint path for option '{name}'")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found for '{name}': {ckpt}")
        actor = Actor(env=env, n_obs=td3_obs_dim, n_act=n_act, device=device, exploration_noise=0.0)
        opt = MoveInDirectionOption(
            name=name,
            env=env,
            actor=actor,
            device=device,
            checkpoint_path=ckpt,
            termination_time=args.termination_time,
            concatenate_only_xy=args.concatenate_only_xy,
            action_low=action_low,
            action_high=action_high,
        )
        opt.load_actor_weights()
        options.append(opt)
    return options


def _log_maze_reset(env: gym.Env, tag: str = "") -> None:
    unwrapped = env.unwrapped
    init_xy = unwrapped.get_xy()
    goal_xy = unwrapped.cur_goal_xy
    prefix = f"[{tag}] " if tag else ""
    logging.info(f"{prefix}init_xy={init_xy}  goal_xy={goal_xy}")


def placeholder_noop_configure_env_for_hrl() -> None:
    """Reserved for future env-specific wiring (e.g. reward shaping flags). No-op."""
    return


def train_one_seed(
    seed: int,
    args: Args,
    device: torch.device,
    run_name: str,
    save_path: str
) -> None:
    seed_save_path = os.path.join(save_path, f"seed_{seed}") if len(args.seeds) > 1 else save_path
    os.makedirs(seed_save_path, exist_ok=True)
    seed_run_name = f"{run_name}_seed{seed}" if len(args.seeds) > 1 else run_name
    logging.info(f"Starting seed={seed}")

    if args.track_with_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=seed_run_name,
            group=run_name,
            config={**vars(args), "seed": seed},
            save_code=True,
            dir=".ogbench/wandb",
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    env = make_antmaze_hrl_env(args)
    env.action_space.seed(seed)
    placeholder_noop_configure_env_for_hrl()

    ckpt_names = ["up", "down", "left", "right"]
    ckpt_paths = [args.checkpoint_up, args.checkpoint_down, args.checkpoint_left, args.checkpoint_right]
    options = build_move_options(env, device, ckpt_paths, ckpt_names, args)
    num_options = len(options)  # 5: noop + up + down + left + right

    base_obs_dim = int(np.prod(env.observation_space.shape))
    if args.discretize_hl_obs:
        xy_enc_dim = 2 * (args.num_bins + 1) if args.use_one_hot else 4
        hl_obs_dim = xy_enc_dim + (base_obs_dim - 2)
        mode = f"one-hot ({args.num_bins} bins/axis + displacement)" if args.use_one_hot else "bin-idx + displacement (4 dims)"
        logging.info(f"High-level Q input dim={hl_obs_dim} (discretized XY: {mode})")
    else:
        hl_obs_dim = base_obs_dim
        logging.info(f"High-level Q input dim={hl_obs_dim} (same as env observation space)")

    q_network = QNetwork(hl_obs_dim, num_options, hidden_dim=args.q_hidden_dim).to(device)
    target_network = QNetwork(hl_obs_dim, num_options, hidden_dim=args.q_hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    _log_param_counts("Q-network", q_network)
    _log_param_counts("Target network", target_network)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(hl_obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Discrete(num_options)
    rb = ReplayBuffer(
        args.buffer_size,
        obs_space,
        action_space,
        device,
        handle_timeout_termination=False,
    )

    start_global_step = 0
    episode_returns = deque(maxlen=args.episode_window_size)
    episode_successes = deque(maxlen=args.episode_window_size)
    episode_completions = deque(maxlen=args.episode_window_size)
    training_metrics: List[dict] = []
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_path}")
        logging.info(f"Loading checkpoint from: {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        q_network.load_state_dict(checkpoint["q_network_state_dict"])
        target_network.load_state_dict(checkpoint["target_network_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_global_step = checkpoint["global_step"]
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "np_random_state" in checkpoint:
            np.random.set_state(checkpoint["np_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if "torch_cuda_random_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
        if "episode_returns" in checkpoint:
            episode_returns.extend(checkpoint["episode_returns"])
        if "episode_successes" in checkpoint:
            episode_successes.extend(checkpoint["episode_successes"])
        if "episode_completions" in checkpoint:
            episode_completions.extend(checkpoint["episode_completions"])
        if "training_metrics" in checkpoint:
            training_metrics = checkpoint["training_metrics"]
        logging.info(f"Resuming from global_step={start_global_step}")

    ob, info = env.reset(seed=seed)
    _log_maze_reset(env, tag="initial reset")
    unwrapped = env.unwrapped
    half_cell = unwrapped._maze_unit / 2
    nrows, ncols = unwrapped.maze_map.shape
    xy_low = np.array([
        1 * unwrapped._maze_unit - unwrapped._offset_x - half_cell,
        1 * unwrapped._maze_unit - unwrapped._offset_y - half_cell,
    ], dtype=np.float32)
    xy_high = np.array([
        (ncols - 2) * unwrapped._maze_unit - unwrapped._offset_x + half_cell,
        (nrows - 2) * unwrapped._maze_unit - unwrapped._offset_y + half_cell,
    ], dtype=np.float32)
    logging.info(f"Maze XY bounds: low={xy_low} high={xy_high}")

    agent = HierarchicalAntMazeDQNAgent(
        env, q_network, device, options=options,
        discretize_hl_obs=args.discretize_hl_obs,
        num_bins=args.num_bins,
        use_one_hot=args.use_one_hot,
        xy_low=xy_low,
        xy_high=xy_high,
    )
    agent.reset(ob, info)
    assert len(agent._options) == num_options

    start_time = None
    start_burnin_global_step = None
    episode_return = 0.0
    episode_step_count = 0
    episode_had_completion = False
    current_hl_info = None
    last_loss = torch.tensor(0.0, device=device)

    pbar = tqdm.tqdm(range(start_global_step, start_global_step + args.total_timesteps))
    for global_step in pbar:
        if args.run_profiling and global_step == args.profiling_start:
            args.last_time = time.time()
            args.profiling_dict = {
                "calculate_epsilon": 0.0,
                "select_agent_action": 0.0,
                "add_transition_to_rb": 0.0,
                "step_env": 0.0,
                "reset_env_on_ep_end": 0.0,
                "update_dqn_params": 0.0,
                "log_to_wandb": 0.0,
                "save_checkpoint": 0.0,
                "run_validation": 0.0,
            }
        if global_step == args.learning_starts + args.measure_burnin:
            start_time = time.time()
            start_burnin_global_step = global_step

        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        agent.epsilon = epsilon
        _prof_checkpoint(args, global_step, "calculate_epsilon")

        low_level_action = agent.select_action(ob, info)
        _prof_checkpoint(args, global_step, "select_agent_action")

        if agent.was_new_option_selected():
            if current_hl_info is not None:
                if args.reward_type == "dense_on_options":
                    opt_reward = _dense_reward_from_env(env)
                    current_hl_info["accumulated_reward"] = opt_reward
                    episode_return += opt_reward
                # logging.info(
                #     f"[option done] step={global_step} option={agent._options[current_hl_info['action']].name!r}"
                #     f" length={current_hl_info['option_length']} reward={current_hl_info['accumulated_reward']:.4f}"
                # )
                rb.add(
                    current_hl_info["obs"].cpu().numpy(),
                    current_hl_info["next_obs"].cpu().numpy(),
                    np.array([current_hl_info["action"]]),
                    np.array([current_hl_info["accumulated_reward"]]),
                    np.array([current_hl_info["done"]]),
                    [{}],
                )
            obs, action = agent.get_last_transition_info()
            current_hl_info = {
                "obs": obs,
                "action": action,
                "accumulated_reward": 0.0,
                "option_length": 0,
            }
        _prof_checkpoint(args, global_step, "add_transition_to_rb")

        next_ob, reward, terminated, truncated, next_info = env.step(low_level_action)
        _prof_checkpoint(args, global_step, "step_env")
        assert current_hl_info is not None
        current_hl_info["next_obs"] = agent.get_obs_tensor(next_ob, next_info)
        done = terminated or truncated
        episode_step_count += 1

        # reward will be 0 if using dense_on_options
        discount = (args.gamma ** current_hl_info["option_length"]) if args.discount_inside_option else 1.0
        current_hl_info["accumulated_reward"] += discount * reward
        current_hl_info["option_length"] += 1
        episode_return += reward

        if not episode_had_completion and maze_task_done(env, next_info):
            episode_had_completion = True

        if done:
            if args.reward_type == "dense_on_options":
                opt_reward = _dense_reward_from_env(env)
                current_hl_info["accumulated_reward"] = opt_reward
                episode_return += opt_reward
            current_hl_info["done"] = True
            # logging.info(
            #     f"[option done/ep end] step={global_step} option={agent._options[current_hl_info['action']].name!r}"
            #     f" length={current_hl_info['option_length']} reward={current_hl_info['accumulated_reward']:.4f}"
            # )
            rb.add(
                current_hl_info["obs"].cpu().numpy(),
                current_hl_info["next_obs"].cpu().numpy(),
                np.array([current_hl_info["action"]]),
                np.array([current_hl_info["accumulated_reward"]]),
                np.array([current_hl_info["done"]]),
                [{}],
            )
            current_hl_info = None

            episode_returns.append(episode_return)
            episode_successes.append(float(maze_task_done(env, next_info)))
            episode_completions.append(float(episode_had_completion))
            episode_return = 0.0
            episode_step_count = 0
            episode_had_completion = False

            ob, info = env.reset()
            _log_maze_reset(env, tag=f"episode reset step={global_step}")
            agent.reset(ob, info)
        else:
            current_hl_info["done"] = False
            ob, info = next_ob, next_info
        _prof_checkpoint(args, global_step, "reset_env_on_ep_end")

        if global_step >= args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                last_loss = loss.detach()
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.copy_(args.tau * qp.data + (1.0 - args.tau) * tp.data)
        _prof_checkpoint(args, global_step, "update_dqn_params")

        if global_step % args.log_freq == 0 and start_time is not None:
            speed = (global_step - start_burnin_global_step) / (time.time() - start_time)
            avg_return = np.mean(episode_returns) if episode_returns else 0
            avg_success = np.mean(episode_successes) if episode_successes else 0
            avg_completion = np.mean(episode_completions) if episode_completions else 0
            pbar.set_description(
                f"sps: {speed:4.2f}, avg_ret: {avg_return:.2f}, avg_suc: {avg_success:.2%}, avg_cmpl: {avg_completion:.2%}"
            )
            metrics = {
                "train/global_step": global_step,
                "train/speed": float(speed),
                "train/avg_episode_return": float(avg_return),
                "train/success_rate": float(avg_success),
                "train/completion_rate": float(avg_completion),
                "train/epsilon": float(epsilon),
            }
            if global_step >= args.learning_starts:
                metrics["train/loss"] = float(last_loss.item())
            training_metrics.append(metrics)
            if args.track_with_wandb:
                import wandb

                wandb.log(metrics, step=global_step)
        _prof_checkpoint(args, global_step, "log_to_wandb")

        if args.save_model and global_step % args.checkpoint_freq == 0 and global_step >= args.learning_starts:
            checkpoint_path = os.path.join(seed_save_path, f"checkpoint_step{global_step}.pt")
            save_checkpoint(
                global_step=global_step,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                args=args,
                episode_returns=episode_returns,
                episode_successes=episode_successes,
                episode_completions=episode_completions,
                training_metrics=training_metrics,
                save_path=checkpoint_path,
            )
            _prof_checkpoint(args, global_step, "save_checkpoint")

            logging.info(
                f"Running validation with 10 episodes "
                f"(validation_greedy={args.validation_greedy})..."
            )
            val_eps = agent.epsilon
            if args.validation_greedy:
                agent.epsilon = 0.0
            q_network.eval()
            val_metrics = run_maze_validation_episodes(
                env=env,
                agent=agent,
                num_episodes=10,
                num_episode_videos=args.save_first_val_episodes_videos,
                save_dir=seed_save_path,
                video_prefix=f"validation_step{global_step}",
            )
            q_network.train()
            agent.epsilon = val_eps

            logging.info(
                f"Validation step {global_step}: success_rate={val_metrics['success_rate']:.2%}, "
                f"completion_rate={val_metrics['completion_rate']:.2%}, "
                f"avg_return={val_metrics['avg_episode_return']:.2f}"
            )
            if args.track_with_wandb:
                import wandb

                wandb.log(
                    {
                        "val/global_step": global_step,
                        "val/success_rate": float(val_metrics["success_rate"]),
                        "val/completion_rate": float(val_metrics["completion_rate"]),
                        "val/avg_episode_return": float(val_metrics["avg_episode_return"]),
                    },
                    step=global_step,
                )

            ob, info = env.reset()
            _log_maze_reset(env, tag=f"post-validation reset step={global_step}")
            agent.reset(ob, info)
            episode_return = 0.0
            episode_step_count = 0
            episode_had_completion = False
            current_hl_info = None
            _prof_checkpoint(args, global_step, "run_validation")

    if args.run_profiling:
        _save_profiling_json(
            args.profiling_dict, seed_save_path, args.profiling_start, args.total_timesteps, "DQN_antmaze_hl"
        )
        _save_profiling_bar_graph(
            args.profiling_dict, seed_save_path, args.profiling_start, args.total_timesteps, "DQN_antmaze_hl"
        )

    env.close()
    if args.track_with_wandb:
        import wandb

        wandb.finish()

    if args.save_model:
        final_model_path = os.path.join(seed_save_path, f"final_model_step{global_step}.pt")
        save_checkpoint(
            global_step=global_step,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            args=args,
            episode_returns=episode_returns,
            episode_successes=episode_successes,
            episode_completions=episode_completions,
            training_metrics=training_metrics,
            save_path=final_model_path,
        )
        logging.info(f"Final model saved to {final_model_path}")

    metrics_path = os.path.join(seed_save_path, f"training_metrics_step{global_step}.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    logging.info(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    # Imports sb3 dependency
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError(
            'Install stable_baselines3>=2.0, e.g. poetry run pip install "stable_baselines3==2.0.0a1"'
        )

    # Processes args
    args = tyro.cli(Args)
    num_episodes = args.total_timesteps // args.max_episode_steps
    if args.total_timesteps % args.max_episode_steps != 0:
        logging.warning(
            f"total_timesteps ({args.total_timesteps}) is not divisible by max_episode_steps ({args.max_episode_steps}); "
            f"training for {num_episodes * args.max_episode_steps} env steps instead."
        )
        args.total_timesteps = num_episodes * args.max_episode_steps

    # Makes run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}__{timestamp}" if args.run_name else f"antmaze_hl__{args.maze_type}__{timestamp}"

    # Makes save path from run name
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"Saving to: {save_path}")

    # Gets device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Running {len(args.seeds)} seed(s): {args.seeds}")

    # Runs each seed
    for seed in args.seeds:
        train_one_seed(seed, args, device, run_name, save_path)

    logging.info("Training complete.")
