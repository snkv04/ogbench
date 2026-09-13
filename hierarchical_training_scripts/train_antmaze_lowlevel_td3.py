import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

from datetime import datetime
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from loguru import logger as logging
import numpy as np
from PIL import Image
import torch
import tqdm
import tyro
import wandb
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from ogbench.locomaze.maze import make_maze_env
from ogbench.manipspace.oracles.hierarchical.utils import add_text_overlay, save_episode_video
from hierarchical_training_scripts.td3_common import (
    apply_compile_and_cudagraphs,
    build_td3_networks,
    compute_exploration_noise,
    make_update_fns,
    save_td3_checkpoint,
)
from hierarchical_training_scripts.train_cube_hrl_dqn import (
    _prof_checkpoint,
    _save_profiling_bar_graph,
    _save_profiling_json,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Environment
    maze_type: str = "arena"
    """maze type: one of 'arena', 'medium', 'large', 'giant', 'teleport'"""
    max_episode_steps: int = 1000
    """maximum steps per episode"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    start_exploration_noise: float = 0.8
    """the scale of exploration noise at learning_starts"""
    end_exploration_noise: float = 0.8
    """the scale of exploration noise at total_timesteps / 2 (held constant thereafter)"""
    learning_starts: int = 25_000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    measure_burnin: int = 3
    """number of burn-in iterations for speed measure"""

    compile: bool = False
    """whether to use torch.compile"""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile"""

    # Validation
    validation_freq: int = 50_000
    """how often (in env steps) to run validation"""
    num_validation_episodes: int = 10
    """number of episodes to run validation for"""
    num_episode_videos: int = 2
    """number of episode videos to save"""
    save_every_k_training_episodes: int = 0
    """save a training video every k episodes (0 to disable)"""
    episode_window_len: int = 20
    """number of recent episodes to average over for training metrics"""

    reward_type: str = "sparse"
    """reward type: 'sparse' (binary success signal) or 'dense' (negative euclidean distance to goal)"""
    fixed_init_ij: bool = False
    """if True, always start from init_ij=(3,2) with no init noise; if False, randomly sample from free cells"""
    avoid_walls: bool = False
    """if True, only sample init cells where the adjacent cell in ``goal_offset_dir`` (toward the goal) is also free"""
    concatenate_only_xy: bool = True
    """if True, prepend only init xy (obs size +2); if False, prepend full init obs (obs size *2)"""
    goal_offset_dir: str = "up"
    """direction of goal offset relative to init: one of 'up', 'down', 'left', 'right'"""
    goal_offset_dist: float = 1.0
    """distance of goal offset relative to init, in maze unit blocks"""

    # Profiling
    run_profiling: bool = False
    profiling_start: int = 0


def _is_goal_xy_reachable(maze_env, goal_xy) -> bool:
    """Return True if goal_xy maps to a free (non-wall) cell in the maze grid."""
    i, j = maze_env.xy_to_ij(goal_xy)
    maze_map = maze_env.maze_map
    return 0 <= i < maze_map.shape[0] and 0 <= j < maze_map.shape[1] and maze_map[i, j] == 0


# World goal offset in (dx, dy) units of maze_unit; must match MazeEnv ij convention (ij_to_xy).
_DIR_TO_XY_OFFSET = {"up": (0.0, 1.0), "down": (0.0, -1.0), "right": (1.0, 0.0), "left": (-1.0, 0.0)}
# Neighbor of init (i,j) toward the goal: one step in grid along the same direction as ``_DIR_TO_XY_OFFSET``.
_DIR_TO_NEIGHBOR_DELTA_IJ = {"up": (1, 0), "down": (-1, 0), "right": (0, 1), "left": (0, -1)}


class RandomInitGoalEnv(gym.Wrapper):
    """Wrapper that randomly samples init and goal positions from free maze cells on each reset.

    Prepends the true (post-noise) init xy (or full init obs) to every observation,
    expanding the observation space from (29,) to (31,) or (58,) respectively.
    """

    def __init__(
        self,
        env,
        goal_offset_dir: str,
        goal_offset_dist: float,
        fixed_init_ij: bool = False,
        avoid_walls: bool = False,
        concatenate_only_xy: bool = True,
        reward_type: str = "sparse",
        augment_observations: bool = True,
    ):
        super().__init__(env)
        assert reward_type in ("sparse", "dense"), f"reward_type must be 'sparse' or 'dense', got '{reward_type}'"
        if goal_offset_dir not in _DIR_TO_XY_OFFSET:
            raise ValueError(f"goal_offset_dir must be one of {list(_DIR_TO_XY_OFFSET)}, got '{goal_offset_dir}'")
        dx, dy = _DIR_TO_XY_OFFSET[goal_offset_dir]
        goal_x_offset = dx * goal_offset_dist
        goal_y_offset = dy * goal_offset_dist

        maze_map = env.unwrapped.maze_map
        rows, cols = np.where(maze_map == 0)
        all_free = set(zip(rows.tolist(), cols.tolist()))
        if avoid_walls:
            di, dj = _DIR_TO_NEIGHBOR_DELTA_IJ[goal_offset_dir]
            self._free_cells = [
                (i, j) for (i, j) in all_free
                if (i + di, j + dj) in all_free
            ]
        else:
            self._free_cells = list(all_free)
        self._fixed_init_ij = fixed_init_ij
        self._concatenate_only_xy = concatenate_only_xy
        self._goal_xy_offset = (goal_x_offset, goal_y_offset)
        self._reward_type = reward_type
        self._augment_observations = augment_observations
        logging.info(f"self._free_cells = {self._free_cells}")

        base_shape = env.observation_space.shape  # (29,)
        prefix_size = 2 if concatenate_only_xy else base_shape[0]
        self._episode_init_prefix = np.zeros(prefix_size, dtype=np.float64)
        if augment_observations:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(base_shape[0] + prefix_size,),
                dtype=env.observation_space.dtype,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=base_shape,
                dtype=env.observation_space.dtype,
            )

    def _augment_obs(self, obs):
        if not self._augment_observations:
            return obs
        return np.concatenate([self._episode_init_prefix, obs])

    def reset(self, *, seed=None, options=None, **kwargs):
        unwrapped = self.unwrapped
        init_ij = (1, 1) if self._fixed_init_ij else self._free_cells[np.random.randint(len(self._free_cells))]
        task_info = dict(
            init_ij=init_ij,
            # init_xy=unwrapped.ij_to_xy(init_ij),
            goal_xy_offset=self._goal_xy_offset,
        )
        options = options or {}
        options["task_info"] = task_info
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        # Record true (post-noise) init xy — set_xy has already been called inside MazeEnv.reset()
        if self._concatenate_only_xy:
            self._episode_init_prefix = unwrapped.get_xy().astype(np.float64)
        else:
            self._episode_init_prefix = obs.astype(np.float64)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._reward_type == "dense":
            agent_xy = self.unwrapped.get_xy()
            goal_xy = self.unwrapped.cur_goal_xy
            reward = -float(np.linalg.norm(agent_xy - goal_xy))
        return self._augment_obs(obs), reward, terminated, truncated, info


class AntMazeTD3Agent:
    """Thin wrapper around a TD3 actor for use in validation."""

    def __init__(self, actor, device, action_low: float, action_high: float):
        self.actor = actor
        self.device = device
        self.action_low = action_low
        self.action_high = action_high
        self.active_option = None  # satisfies run_validation_episodes video-overlay check

    def reset(self, ob, info):
        pass

    def select_action(self, ob, info):
        obs_tensor = torch.as_tensor(ob.reshape(1, -1), device=self.device, dtype=torch.float)
        with torch.no_grad():
            return self.actor(obs_tensor).clamp(self.action_low, self.action_high)[0].cpu().numpy()


def _maze_validation_frame(env: gym.Env, agent) -> np.ndarray:
    """Render one RGB frame; if ``agent`` is hierarchical, overlay active option index and name."""
    frame = env.render()
    if not (hasattr(agent, "active_option") and hasattr(agent, "_options")):
        return frame
    current_option_idx = None
    current_option_name = None
    ao = agent.active_option
    if ao is not None:
        current_option_idx = agent._options.index(ao)
        current_option_name = ao.name
    return add_text_overlay(frame, task_name=None, option_idx=current_option_idx, option_text=current_option_name)


def run_maze_validation_episodes(
    env,
    agent,
    num_episodes: int,
    # max_episode_steps: int,
    num_episode_videos: int = 0,
    save_dir: Optional[str] = None,
    video_prefix: str = "validation",
) -> dict:
    """Run validation episodes and compute maze-specific metrics.

    Success is determined by info['success'] == 1.0, which MazeEnv sets when
    the agent is within goal_tol of the goal.

    When saving videos, if ``agent`` has ``active_option`` and ``_options`` (hierarchical agent),
    each frame is passed through ``add_text_overlay`` with option index and name only.
    """
    assert num_episode_videos <= num_episodes, "Cannot make more videos than episodes"

    tasks_completed_at_end = 0.0
    tasks_completed_at_all = 0.0
    episode_returns = []
    filtered_tasks_completed_at_end = 0.0
    filtered_tasks_completed_at_all = 0.0
    filtered_episode_returns = []
    filtered_episode_count = 0

    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Running validation episodes"):
        ob, info = env.reset()
        agent.reset(ob, info)

        goal_reachable = _is_goal_xy_reachable(env.unwrapped, env.unwrapped.cur_goal_xy)
        if goal_reachable:
            filtered_episode_count += 1

        episode_had_success = False
        episode_return = 0.0
        save_this_ep = ep_idx < num_episode_videos
        episode_frames = [_maze_validation_frame(env, agent)] if save_this_ep else []

        done = False
        step = 0
        while not done:
            action = agent.select_action(ob, info)
            ob, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            episode_return += float(reward)

            success = info.get("success", 0.0) == 1.0
            if success and not episode_had_success:
                tasks_completed_at_all += 1
                if goal_reachable:
                    filtered_tasks_completed_at_all += 1
                episode_had_success = True
            if done and success:
                tasks_completed_at_end += 1
                if goal_reachable:
                    filtered_tasks_completed_at_end += 1

            if save_this_ep:
                episode_frames.append(_maze_validation_frame(env, agent))

        logging.info(f"Episode {ep_idx} terminated after {step} steps")
        episode_returns.append(episode_return)
        if goal_reachable:
            filtered_episode_returns.append(episode_return)

        if save_this_ep and save_dir is not None and episode_frames:
            save_episode_video(
                episode_frames,
                save_dir=save_dir,
                filename=f"{video_prefix}_episode{ep_idx}",
                fps=30,
            )

    n = num_episodes if num_episodes > 0 else 1
    fn = filtered_episode_count if filtered_episode_count > 0 else 1
    return {
        "success_rate": tasks_completed_at_end / n,
        "completion_rate": tasks_completed_at_all / n,
        "avg_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "num_episodes": num_episodes,
        "filtered_success_rate": filtered_tasks_completed_at_end / fn,
        "filtered_completion_rate": filtered_tasks_completed_at_all / fn,
        "filtered_avg_episode_return": float(np.mean(filtered_episode_returns)) if filtered_episode_returns else 0.0,
        "num_filtered_episodes": filtered_episode_count,
    }


def _log_reset(env, episode_idx: int, save_dir: str) -> None:
    """Log true (post-noise) init/goal xy and save a frame after env.reset()."""
    unwrapped = env.unwrapped
    true_init_xy = unwrapped.get_xy()
    true_goal_xy = unwrapped.cur_goal_xy
    task_info = unwrapped.cur_task_info or {}
    init_ij = task_info.get("init_ij")
    # logging.info(
    #     f"[ep {episode_idx}] init_ij={init_ij}  true_init_xy={true_init_xy}  true_goal_xy={true_goal_xy}"
    # )
    frame = env.render()
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(frame).save(os.path.join(save_dir, f"reset_ep{episode_idx:06d}.png"))


class EpisodeStatsTracker:
    """Rolling window of recent-episode training stats (returns/success), overall and goal-reachable-filtered."""

    def __init__(self, window_len: int):
        self.max_ep_ret = -float("inf")
        self.avg_returns = deque(maxlen=window_len)
        self.tasks_completed_at_end = deque(maxlen=window_len)
        self.tasks_completed_at_all = deque(maxlen=window_len)
        self.filtered_avg_returns = deque(maxlen=window_len)
        self.filtered_tasks_completed_at_end = deque(maxlen=window_len)
        self.filtered_tasks_completed_at_all = deque(maxlen=window_len)

    def record_episode(self, episode_return: float, success: bool, episode_had_success: bool, goal_reachable: bool) -> None:
        self.max_ep_ret = max(self.max_ep_ret, episode_return)
        self.avg_returns.append(episode_return)
        self.tasks_completed_at_end.append(float(success))
        self.tasks_completed_at_all.append(float(episode_had_success))
        if goal_reachable:
            self.filtered_tasks_completed_at_end.append(float(success))
            self.filtered_tasks_completed_at_all.append(float(episode_had_success))
            self.filtered_avg_returns.append(episode_return)

    def describe(self, global_step: int) -> str:
        return f"global_step={global_step}, episodic_return={torch.tensor(self.avg_returns).mean(): 4.2f} (max={self.max_ep_ret: 4.2f})"

    def wandb_logs(self) -> dict:
        def mean_or(dq, default=0.0):
            return float(np.mean(dq)) if dq else default

        return {
            "episode_return": mean_or(self.avg_returns),
            "success_rate": mean_or(self.tasks_completed_at_end),
            "completion_rate": mean_or(self.tasks_completed_at_all),
            "filtered_episode_return": mean_or(self.filtered_avg_returns),
            "filtered_success_rate": mean_or(self.filtered_tasks_completed_at_end),
            "filtered_completion_rate": mean_or(self.filtered_tasks_completed_at_all),
        }


def setup_experiment(args: Args) -> str:
    """Init wandb, seed all RNGs, and return the run name."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"antmaze-{args.maze_type}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}__{timestamp}"
    logging.info(f"run_name = {run_name}")

    wandb.init(
        project="td3_antmaze",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    return run_name


def build_env(args: Args):
    """Build the RandomInitGoalEnv-wrapped ant maze env and return it with action/obs metadata."""
    base_env = make_maze_env(
        "ant", "maze", maze_type=args.maze_type, terminate_at_goal=False,
        add_noise_to_init=not args.fixed_init_ij, add_noise_to_goal=False,
    )
    env = RandomInitGoalEnv(
        base_env,
        goal_offset_dir=args.goal_offset_dir,
        goal_offset_dist=args.goal_offset_dist,
        fixed_init_ij=args.fixed_init_ij,
        avoid_walls=args.avoid_walls,
        concatenate_only_xy=args.concatenate_only_xy,
        reward_type=args.reward_type,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    env.action_space.seed(args.seed)

    n_act = math.prod(env.action_space.shape)
    n_obs = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    action_low, action_high = float(env.action_space.low[0]), float(env.action_space.high[0])
    logging.info(f"action space: {env.action_space}")
    logging.info(f"observation space: {env.observation_space}")

    return env, n_obs, n_act, action_low, action_high


def _reset_episode(env, run_name: str, episode_idx: int, device, seed: Optional[int] = None):
    """Reset the env, log the reset frame, and return the initial obs tensor plus goal-reachability."""
    obs_raw, _ = env.reset(seed=seed)
    reset_frame_dir = os.path.join(".ogbench", "td3_runs", run_name, "reset_frames")
    _log_reset(env, episode_idx, reset_frame_dir)
    obs = torch.as_tensor(obs_raw.reshape(1, -1), device=device, dtype=torch.float)
    goal_reachable = _is_goal_xy_reachable(env.unwrapped, env.unwrapped.cur_goal_xy)
    return obs, goal_reachable


def run_periodic_validation(args: Args, env, val_agent, actor, qnet, run_name: str, global_step: int, num_recent_episodes: int) -> None:
    logging.info(f"Running validation with {num_recent_episodes} recent-episode window at step {global_step}...")

    actor.eval()
    qnet.eval()

    val_metrics = run_maze_validation_episodes(
        env=env,
        agent=val_agent,
        num_episodes=args.num_validation_episodes,
        num_episode_videos=args.num_episode_videos,
        save_dir=os.path.join(".ogbench", "td3_runs", run_name, "validation"),
        video_prefix=f"validation_step{global_step}",
    )

    actor.train()
    qnet.train()

    logging.info(f"Validation results (step {global_step}):")
    logging.info(f"    success_rate={val_metrics['success_rate']:.2%}")
    logging.info(f"    completion_rate={val_metrics['completion_rate']:.2%}")
    logging.info(f"    avg_episode_return={val_metrics['avg_episode_return']:.2f}")
    logging.info(f"    filtered_success_rate={val_metrics['filtered_success_rate']:.2%}  (n={val_metrics['num_filtered_episodes']})")
    logging.info(f"    filtered_completion_rate={val_metrics['filtered_completion_rate']:.2%}")
    logging.info(f"    filtered_avg_episode_return={val_metrics['filtered_avg_episode_return']:.2f}")

    wandb.log(
        {
            "val/success_rate": float(val_metrics["success_rate"]),
            "val/completion_rate": float(val_metrics["completion_rate"]),
            "val/avg_episode_return": float(val_metrics["avg_episode_return"]),
            "val/num_episodes": float(val_metrics["num_episodes"]),
            "val/filtered_success_rate": float(val_metrics["filtered_success_rate"]),
            "val/filtered_completion_rate": float(val_metrics["filtered_completion_rate"]),
            "val/filtered_avg_episode_return": float(val_metrics["filtered_avg_episode_return"]),
            "val/num_filtered_episodes": float(val_metrics["num_filtered_episodes"]),
        },
        step=global_step,
    )


def train(args: Args, env, n_obs: int, n_act: int, action_low: float, action_high: float, run_name: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Device: {device}")

    networks, policy = build_td3_networks(
        env=env,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        exploration_noise=args.start_exploration_noise,
        learning_rate=args.learning_rate,
        cudagraphs=args.cudagraphs,
        compile=args.compile,
    )
    actor, qnet = networks.actor, networks.qnet
    update_main, update_pol = make_update_fns(
        networks, gamma=args.gamma, policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        action_low=action_low, action_high=action_high,
    )
    policy, update_main, update_pol = apply_compile_and_cudagraphs(
        policy, update_main, update_pol, compile=args.compile, cudagraphs=args.cudagraphs
    )

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    val_agent = AntMazeTD3Agent(actor=actor, device=device, action_low=action_low, action_high=action_high)

    episode_idx = 0
    obs, episode_goal_reachable = _reset_episode(env, run_name, episode_idx, device, seed=args.seed)

    stats = EpisodeStatsTracker(window_len=args.episode_window_len)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    desc = ""
    episode_return = 0.0
    episode_had_success = False
    save_this_ep = False
    episode_frames = []

    for global_step in pbar:
        if args.run_profiling and global_step == args.profiling_start:
            args.last_time = time.time()
            args.profiling_dict = {
                "select_agent_action": 0.0,
                "step_env": 0.0,
                "render_env_for_train_vids": 0.0,
                "add_transition_to_rb": 0.0,
                "update_td3_critic": 0.0,
                "update_td3_actor": 0.0,
                "log_to_wandb": 0.0,
                "reset_env_at_ep_end": 0.0,
                "run_validation": 0.0,
                "save_td3_checkpoint": 0.0,
            }
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            current_exploration_noise = torch.tensor(
                compute_exploration_noise(
                    global_step, args.learning_starts, args.total_timesteps,
                    args.start_exploration_noise, args.end_exploration_noise,
                ),
                device=device, dtype=torch.float,
            )
            action_tensor = policy(obs=obs, exploration_noise=current_exploration_noise).clamp(action_low, action_high)
            action = action_tensor[0].cpu().numpy()
        _prof_checkpoint(args, global_step, "select_agent_action")

        next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = torch.as_tensor(next_obs_raw.reshape(1, -1), device=device, dtype=torch.float)
        _prof_checkpoint(args, global_step, "step_env")

        if save_this_ep:
            episode_frames.append(env.render())
        _prof_checkpoint(args, global_step, "render_env_for_train_vids")

        rewards = torch.as_tensor([[env_reward]], device=device, dtype=torch.float)
        episode_return += float(env_reward)
        success = info.get("success", 0.0) == 1.0
        if success and not episode_had_success:
            episode_had_success = True

        actions_tensor = torch.as_tensor(action, device=device, dtype=torch.float).unsqueeze(0)
        terminations = torch.as_tensor([terminated], device=device, dtype=torch.bool)
        transition = TensorDict(
            observations=obs,
            next_observations=next_obs,
            actions=actions_tensor,
            rewards=rewards,
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device,
        )

        obs = next_obs
        data = extend_and_sample(transition)
        _prof_checkpoint(args, global_step, "add_transition_to_rb")

        if global_step > args.learning_starts:
            out_main = update_main(data)
            _prof_checkpoint(args, global_step, "update_td3_critic")
            if global_step % args.policy_frequency == 0:
                out_main.update(update_pol(data))

                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                networks.qnet_target_params.lerp_(networks.qnet_params.data, args.tau)
                networks.target_actor_params.lerp_(networks.actor_params.data, args.tau)
            _prof_checkpoint(args, global_step, "update_td3_actor")

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        **stats.wandb_logs(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                wandb.log({"speed": speed, **logs}, step=global_step)
            _prof_checkpoint(args, global_step, "log_to_wandb")

        if done:
            stats.record_episode(episode_return, success, episode_had_success, episode_goal_reachable)
            desc = stats.describe(global_step)

            if save_this_ep:
                assert episode_frames, "Episode frames should not be empty"
                logging.info(f"Episode {episode_idx} is saving {len(episode_frames)} frames")
                save_episode_video(
                    episode_frames,
                    save_dir=os.path.join(".ogbench", "td3_runs", run_name, "training_videos"),
                    filename=f"training_step{global_step}_ep{episode_idx}",
                    fps=30,
                )

            episode_idx += 1
            obs, episode_goal_reachable = _reset_episode(env, run_name, episode_idx, device)
            episode_return = 0.0
            episode_had_success = False

            save_this_ep = (
                args.save_every_k_training_episodes > 0
                and episode_idx % args.save_every_k_training_episodes == 0
            )
            episode_frames = [env.render()] if save_this_ep else []

            _prof_checkpoint(args, global_step, "reset_env_at_ep_end")

        if global_step > args.learning_starts and global_step % args.validation_freq == 0:
            run_periodic_validation(args, env, val_agent, actor, qnet, run_name, global_step, len(stats.avg_returns))
            _prof_checkpoint(args, global_step, "run_validation")

            checkpoint_path = os.path.join(".ogbench", "td3_runs", run_name, "checkpoints", f"checkpoint_step{global_step}.pt")
            save_td3_checkpoint(
                global_step=global_step,
                actor=actor,
                target_actor_params=networks.target_actor_params,
                qnet_params=networks.qnet_params,
                qnet_target_params=networks.qnet_target_params,
                actor_optimizer=networks.actor_optimizer,
                q_optimizer=networks.q_optimizer,
                save_path=checkpoint_path,
            )
            _prof_checkpoint(args, global_step, "save_td3_checkpoint")

    if args.run_profiling:
        save_path = os.path.join(".ogbench", "td3_profiling", run_name)
        os.makedirs(save_path, exist_ok=True)
        _save_profiling_json(args.profiling_dict, save_path, args.profiling_start, args.total_timesteps, "TD3")
        _save_profiling_bar_graph(args.profiling_dict, save_path, args.profiling_start, args.total_timesteps, "TD3")

    env.close()


def main() -> None:
    args = tyro.cli(Args)
    run_name = setup_experiment(args)
    env, n_obs, n_act, action_low, action_high = build_env(args)
    train(args, env, n_obs, n_act, action_low, action_high, run_name)


if __name__ == "__main__":
    main()

