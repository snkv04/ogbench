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
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

from ogbench.locomaze.maze import make_maze_env
from ogbench.manipspace.oracles.hierarchical.utils import save_episode_video
from hierarchical_training_scripts.train_cube_lowlevel_td3 import Actor, QNetwork
from hierarchical_training_scripts.train_cube_hrl_dqn import (
    _log_param_counts,
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
    maze_type: str = "medium"
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
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
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
    save_every_k_training_episodes: int = 50
    """save a training video every k episodes (0 to disable)"""
    episode_window_len: int = 20
    """number of recent episodes to average over for training metrics"""

    fixed_init_ij: bool = False
    """if True, always start from init_ij=(3,2) with no init noise; if False, randomly sample from free cells"""
    avoid_walls: bool = False
    """if True, only sample init cells where the cell above, (i+1, j), is also free"""
    concatenate_only_xy: bool = True
    """if True, prepend only init xy (obs size +2); if False, prepend full init obs (obs size *2)"""

    # Profiling
    run_profiling: bool = False
    profiling_start: int = 0
    profiling_end: int = 1_000_000


def _is_goal_xy_reachable(maze_env, goal_xy) -> bool:
    """Return True if goal_xy maps to a free (non-wall) cell in the maze grid."""
    i, j = maze_env.xy_to_ij(goal_xy)
    maze_map = maze_env.maze_map
    return 0 <= i < maze_map.shape[0] and 0 <= j < maze_map.shape[1] and maze_map[i, j] == 0


class RandomInitGoalEnv(gym.Wrapper):
    """Wrapper that randomly samples init and goal positions from free maze cells on each reset.

    Prepends the true (post-noise) init xy (or full init obs) to every observation,
    expanding the observation space from (29,) to (31,) or (58,) respectively.
    """

    def __init__(self, env, fixed_init_ij: bool = False, avoid_walls: bool = False, concatenate_only_xy: bool = True):
        super().__init__(env)
        maze_map = env.unwrapped.maze_map
        rows, cols = np.where(maze_map == 0)
        all_free = set(zip(rows.tolist(), cols.tolist()))
        self._free_cells = [
            (i, j) for (i, j) in all_free
            if not avoid_walls or (i + 1, j) in all_free
        ]
        self._fixed_init_ij = fixed_init_ij
        self._concatenate_only_xy = concatenate_only_xy
        logging.info(f"self._free_cells = {self._free_cells}")

        base_shape = env.observation_space.shape  # (29,)
        prefix_size = 2 if concatenate_only_xy else base_shape[0]
        self._episode_init_prefix = np.zeros(prefix_size, dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_shape[0] + prefix_size,),
            dtype=env.observation_space.dtype,
        )

    def _augment_obs(self, obs):
        return np.concatenate([self._episode_init_prefix, obs])

    def reset(self, *, seed=None, options=None, **kwargs):
        unwrapped = self.unwrapped
        init_ij = (3, 2) if self._fixed_init_ij else self._free_cells[np.random.randint(len(self._free_cells))]
        task_info = dict(
            init_ij=init_ij,
            # init_xy=unwrapped.ij_to_xy(init_ij),
            goal_xy_offset=(0, 0.25),
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
    """
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
        episode_frames = [env.render()] if save_this_ep else []

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
                episode_frames.append(env.render())

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
    logging.info(
        f"[ep {episode_idx}] init_ij={init_ij}  true_init_xy={true_init_xy}  true_goal_xy={true_goal_xy}"
    )
    frame = env.render()
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(frame).save(os.path.join(save_dir, f"reset_ep{episode_idx:06d}.png"))


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"antmaze-{args.maze_type}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}__{timestamp}"

    wandb.init(
        project="td3_antmaze",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup: MazeEnv wrapped to randomly sample init/goal on each reset.
    base_env = make_maze_env("ant", "maze", maze_type=args.maze_type, terminate_at_goal=False, add_noise_to_init=not args.fixed_init_ij, add_noise_to_goal=False)
    env = RandomInitGoalEnv(
        base_env,
        fixed_init_ij=args.fixed_init_ij,
        avoid_walls=args.avoid_walls,
        concatenate_only_xy=args.concatenate_only_xy
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    env.action_space.seed(args.seed)

    n_act = math.prod(env.action_space.shape)
    n_obs = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    action_low, action_high = float(env.action_space.low[0]), float(env.action_space.high[0])
    logging.info(f"action space: {env.action_space}")
    logging.info(f"observation space: {env.observation_space}")

    actor = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    actor_detach = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    def get_params_qnet():
        qf1 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)

        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        qnet = QNetwork(n_obs=n_obs, n_act=n_act, device="meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target_params, qnet

    def get_params_actor(actor):
        target_actor = Actor(env=env, device="meta", n_act=n_act, n_obs=n_obs)
        actor_params = from_module(actor).data
        target_actor_params = actor_params.clone()
        target_actor_params.to_module(target_actor)
        return actor_params, target_actor_params, target_actor

    qnet_params, qnet_target_params, qnet = get_params_qnet()
    actor_params, target_actor_params, target_actor = get_params_actor(actor)

    _log_param_counts("Actor", actor)
    _log_param_counts("Q-network (per copy)", qnet)

    q_optimizer = optim.Adam(
        qnet_params.values(include_nested=True, leaves_only=True),
        lr=args.learning_rate,
        capturable=args.cudagraphs and not args.compile,
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=args.learning_rate, capturable=args.cudagraphs and not args.compile
    )

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    action_scale = target_actor.action_scale

    def update_main(data):
        observations = data["observations"]
        next_observations = data["next_observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(policy_noise).clamp(-noise_clip, noise_clip).mul(action_scale)

        next_state_actions = (target_actor(next_observations) + clipped_noise).clamp(action_low, action_high)

        qf_next_target = torch.vmap(batched_qf, (0, None, None))(qnet_target_params, next_observations, next_state_actions)
        min_qf_next_target = qf_next_target.min(0).values
        next_q_value = rewards.flatten() + args.gamma * min_qf_next_target.flatten()

        qf_loss = torch.vmap(batched_qf, (0, None, None, None))(qnet_params, observations, actions, next_q_value)
        qf_loss = qf_loss.sum(0)

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data):
        actor_optimizer.zero_grad()
        with qnet_params.data[0].to_module(qnet):
            actor_loss = -qnet(data["observations"], actor(data["observations"])).mean()

        actor_loss.backward()
        actor_optimizer.step()
        return TensorDict(actor_loss=actor_loss.detach())

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[], warmup=5)
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[], warmup=5)
        policy = CudaGraphModule(policy)

    val_agent = AntMazeTD3Agent(actor=actor, device=device, action_low=action_low, action_high=action_high)

    reset_frame_dir = os.path.join(".ogbench", "td3_runs", run_name, "reset_frames")

    # TRY NOT TO MODIFY: start the game
    obs_raw, _ = env.reset(seed=args.seed)
    episode_idx = 0
    _log_reset(env, episode_idx, reset_frame_dir)
    obs = torch.as_tensor(obs_raw.reshape(1, -1), device=device, dtype=torch.float)

    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=args.episode_window_len)
    train_tasks_completed_at_end = deque(maxlen=args.episode_window_len)
    train_tasks_completed_at_all = deque(maxlen=args.episode_window_len)
    train_filtered_avg_returns = deque(maxlen=args.episode_window_len)
    train_filtered_tasks_completed_at_end = deque(maxlen=args.episode_window_len)
    train_filtered_tasks_completed_at_all = deque(maxlen=args.episode_window_len)
    desc = ""
    episode_return = 0.0
    episode_had_success = False
    episode_goal_reachable = _is_goal_xy_reachable(env.unwrapped, env.unwrapped.cur_goal_xy)
    save_this_ep = False
    episode_frames = []

    # Main training loop
    for global_step in pbar:
        if args.run_profiling:
            if global_step >= args.profiling_end:
                break
            if global_step == args.profiling_start:
                args.last_time = time.time()
                args.profiling_dict = {
                    "select_agent_action": 0.0,
                    "step_env": 0.0,
                    "add_transition_to_rb": 0.0,
                    "update_td3_params": 0.0,
                    "log_to_wandb": 0.0,
                    "reset_env": 0.0,
                }
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action_tensor = policy(obs=obs).clamp(action_low, action_high)
            action = action_tensor[0].cpu().numpy()
        _prof_checkpoint(args, global_step, "select_agent_action")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = torch.as_tensor(next_obs_raw.reshape(1, -1), device=device, dtype=torch.float)
        _prof_checkpoint(args, global_step, "step_env")

        if save_this_ep:
            episode_frames.append(env.render())

        rewards = torch.as_tensor([[env_reward]], device=device, dtype=torch.float)
        episode_return += float(env_reward)
        success = info.get("success", 0.0) == 1.0
        if success and not episode_had_success:
            episode_had_success = True

        # TRY NOT TO MODIFY: save data to replay buffer
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

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)
        _prof_checkpoint(args, global_step, "add_transition_to_rb")

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:
                out_main.update(update_pol(data))

                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target_params.lerp_(qnet_params.data, args.tau)
                target_actor_params.lerp_(actor_params.data, args.tau)
            _prof_checkpoint(args, global_step, "update_td3_params")

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": float(np.mean(avg_returns)) if avg_returns else 0.0,
                        "success_rate": float(np.mean(train_tasks_completed_at_end)) if train_tasks_completed_at_end else 0.0,
                        "completion_rate": float(np.mean(train_tasks_completed_at_all)) if train_tasks_completed_at_all else 0.0,
                        "filtered_episode_return": float(np.mean(train_filtered_avg_returns)) if train_filtered_avg_returns else 0.0,
                        "filtered_success_rate": float(np.mean(train_filtered_tasks_completed_at_end)) if train_filtered_tasks_completed_at_end else 0.0,
                        "filtered_completion_rate": float(np.mean(train_filtered_tasks_completed_at_all)) if train_filtered_tasks_completed_at_all else 0.0,
                        "actor_loss": out_main["actor_loss"].mean(),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )
            _prof_checkpoint(args, global_step, "log_to_wandb")

        # Handle end of episode
        if done:
            max_ep_ret = max(max_ep_ret, episode_return)
            avg_returns.append(episode_return)
            train_tasks_completed_at_end.append(float(success))
            train_tasks_completed_at_all.append(float(episode_had_success))
            if episode_goal_reachable:
                train_filtered_tasks_completed_at_end.append(float(success))
                train_filtered_tasks_completed_at_all.append(float(episode_had_success))
                train_filtered_avg_returns.append(episode_return)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            )

            if save_this_ep:
                assert episode_frames, "Episode frames should not be empty"
                logging.info(f"Episode {episode_idx} is saving {len(episode_frames)} frames")
                save_episode_video(
                    episode_frames,
                    save_dir=os.path.join(".ogbench", "td3_runs", run_name, "training_videos"),
                    filename=f"training_step{global_step}_ep{episode_idx}",
                    fps=30,
                )

            obs_raw, _ = env.reset()
            episode_idx += 1
            _log_reset(env, episode_idx, reset_frame_dir)
            obs = torch.as_tensor(obs_raw.reshape(1, -1), device=device, dtype=torch.float)
            episode_return = 0.0
            episode_had_success = False
            episode_goal_reachable = _is_goal_xy_reachable(env.unwrapped, env.unwrapped.cur_goal_xy)
            logging.info(f"Is goal {env.unwrapped.cur_goal_xy} reachable? {episode_goal_reachable}")

            save_this_ep = (
                args.save_every_k_training_episodes > 0
                and episode_idx % args.save_every_k_training_episodes == 0
            )
            if save_this_ep:
                episode_frames = [env.render()]
            else:
                episode_frames = []

            _prof_checkpoint(args, global_step, "reset_env")

        # Optional validation at fixed step intervals
        if global_step > args.learning_starts and global_step % args.validation_freq == 0:
            logging.info(f"Running validation with {len(avg_returns)} recent-episode window at step {global_step}...")

            actor.eval()
            qnet.eval()

            val_metrics = run_maze_validation_episodes(
                env=env,
                agent=val_agent,
                num_episodes=args.num_validation_episodes,
                # max_episode_steps=args.max_episode_steps,
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

    # Profiling output
    if args.run_profiling:
        save_path = os.path.join(".ogbench", "td3_profiling", run_name)
        os.makedirs(save_path, exist_ok=True)
        _save_profiling_json(
            args.profiling_dict, save_path, args.profiling_start, args.profiling_end, "TD3"
        )
        _save_profiling_bar_graph(
            args.profiling_dict, save_path, args.profiling_start, args.profiling_end, "TD3"
        )

    env.close()
