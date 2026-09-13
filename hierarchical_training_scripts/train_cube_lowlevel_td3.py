# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

from datetime import datetime
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
from loguru import logger as logging
import numpy as np
import torch
import tqdm
import tyro
import wandb
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

import ogbench.manipspace  # Register environments
from hierarchical_training_scripts.td3_common import (
    apply_compile_and_cudagraphs,
    build_td3_networks,
    make_update_fns,
)
from hierarchical_training_scripts.train_cube_hrl_dqn import (
    _prof_checkpoint,
    _save_profiling_bar_graph,
    _save_profiling_json,
    run_validation_episodes,
)
from ogbench.manipspace.oracles.hierarchical.hierarchical_agent import HierarchicalAgent
from ogbench.manipspace.oracles.hierarchical.utils import make_cube_env
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
    LiftVerticallyOption,
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "cube-single-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    max_episode_steps: int = 200
    """maximum steps per episode (used by make_cube_env)"""
    task_id: Optional[int] = None
    """fixed task for all episodes (int), 0 = default task, None = random"""
    noise_initial_state: bool = True
    """whether to add noise to initial cube state (make_cube_env)"""
    reward_is_neg_dist: bool = False
    """use negative distance as reward (make_cube_env)"""
    """total timesteps of the experiments"""
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
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    reward_option: str = "move_above_block"
    """name of option whose sparse reward to use"""
    reward_type: str = 'dense'
    """reward type to use"""

    # Validation
    validation_freq: int = 50_000
    """how often (in env steps) to run validation"""
    num_validation_episodes: int = 100
    """number of episodes to run validation for"""
    num_episode_videos: int = 5
    """number of episode videos to save"""

    # Profiling
    run_profiling: bool = False
    profiling_start: int = 0


# Size-14 observation (same structure as HierarchicalDQNAgent.get_obs_tensor)
OBS_DIM = 14


class LowLevelTD3Agent(HierarchicalAgent):
    """HierarchicalAgent-compatible wrapper around the TD3 policy.

    This agent exposes only primitive low-level actions (no options) so it can be
    used with `run_validation_episodes`, which expects a hierarchical-style agent
    interface (reset/select_action/active_option/_options).
    """

    def __init__(self, env, actor, device, action_low: float, action_high: float):
        # Initialize base HierarchicalAgent with no options
        super().__init__(options=None, env=env)

        # TD3-specific components
        self.env = env
        self.actor = actor
        self.device = device
        self.action_low = action_low
        self.action_high = action_high

        # Episode-specific target info
        self._target_block = 0
        self._target_pos = None
        self._target_yaw = 0.0

    def reset(self, ob, info):
        """Called at the start of each validation episode."""
        super().reset(ob, info)
        self._target_block, self._target_pos, self._target_yaw = _get_target_from_info(self.env, info)

    def select_action(self, ob, info):
        """Select a deterministic TD3 action given env info."""
        obs_tensor = torch.as_tensor(
            _info_to_obs_14(info, self._target_block, self._target_pos, self._target_yaw).reshape(1, -1),
            device=self.device,
            dtype=torch.float,
        )
        with torch.no_grad():
            action_tensor = self.actor(obs_tensor).clamp(self.action_low, self.action_high)
        return action_tensor[0].cpu().numpy()


# This is called every time the env is reset, so it gets the new target correctly
def _get_target_from_info(env, info: dict) -> Tuple[int, np.ndarray, float]:
    """Get (target_block, target_pos, target_yaw) for the current episode from env and info."""
    unwrapped = env.unwrapped
    if getattr(unwrapped, "_mode", None) == "data_collection":
        target_block = info["privileged/target_block"]
        target_pos = info["privileged/target_block_pos"].copy()
        target_yaw = float(info["privileged/target_block_yaw"][0])
    else:
        target_block = 0
        target_pos = unwrapped.cur_task_info["goal_xyzs"][target_block].copy()
        target_yaw = 0.0
    return target_block, target_pos, target_yaw


def _info_to_obs_14(
    info: dict,
    target_block: int,
    target_pos: np.ndarray,
    target_yaw: float,
) -> np.ndarray:
    """Build the 14-dim observation from info and episode target (effector, gripper, block, target)."""
    return np.concatenate([
        # End-effector
        info["proprio/effector_pos"],
        np.atleast_1d(info["proprio/effector_yaw"]),
        np.atleast_1d(info["proprio/gripper_opening"]),
        np.atleast_1d(info["proprio/gripper_contact"]),
        
        # Block
        info[f"privileged/block_{target_block}_pos"],
        np.atleast_1d(info[f"privileged/block_{target_block}_yaw"]),

        # Target
        target_pos,
        np.atleast_1d(np.float64(target_yaw)),
    ]).astype(np.float32)


# TODO: Modify to create options on every reset, to deal with the case that multiple tasks/goals are used
def _create_cube_options(env, reset_info, args):
    """Create the 9 cube manipulation options used for sparse rewards.

    Returns a dict mapping option name to option instance.
    """
    base_env = env.unwrapped

    # Determine target block and target pose, mirroring HierarchicalDQNAgent.reset
    if getattr(base_env, "_mode", None) == "data_collection":
        target_block = reset_info["privileged/target_block"]
        stored_target_pos = reset_info["privileged/target_block_pos"].copy()
        stored_target_yaw = reset_info["privileged/target_block_yaw"][0]
    else:
        # In task mode, cube-single has a single cube with goal in cur_task_info
        target_block = 0
        stored_target_pos = base_env.cur_task_info["goal_xyzs"][target_block].copy()
        stored_target_yaw = 0.0  # Task mode uses identity orientation for goals

    # Sample final arm pose the same way as HierarchicalDQNAgent
    final_pos = np.random.uniform(*base_env._arm_sampling_bounds)
    final_yaw = np.random.uniform(-np.pi, np.pi)

    min_norm = 0.08

    # Helper functions for block position and orientation
    def block_above_pos(ob, info):
        return info[f"privileged/block_{target_block}_pos"] + np.array([0, 0, 0.18])

    def block_yaw(ob, info):
        effector_yaw = info["proprio/effector_yaw"][0]
        block_yaw_val = info[f"privileged/block_{target_block}_yaw"][0]
        # Use shortest rotation as in HierarchicalDQNAgent._shortest_yaw
        diff = block_yaw_val - effector_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return effector_yaw + diff

    def block_pos(ob, info):
        return info[f"privileged/block_{target_block}_pos"]

    def target_above_pos(ob, info):
        return stored_target_pos + np.array([0, 0, 0.18])

    def target_yaw(ob, info):
        effector_yaw = info["proprio/effector_yaw"][0]
        # Shortest rotation to stored_target_yaw
        diff = stored_target_yaw - effector_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return effector_yaw + diff

    def target_pos(ob, info):
        return stored_target_pos

    def get_final_pos(ob, info):
        return final_pos

    def get_final_yaw(ob, info):
        return final_yaw

    options = [
        MoveToPositionOption(
            "move_above_block",
            env,
            block_above_pos,
            block_yaw,
            gripper_state=-1,
            min_norm=min_norm,
            reward_type=args.reward_type,
            use_position=True,
            use_yaw=True,
            use_gripper=True,
        ),
        MoveToPositionOption(
            "move_to_block",
            env,
            block_pos,
            block_yaw,
            gripper_state=-1,
            min_norm=min_norm,
            reward_type=args.reward_type,
            use_position=True,
            use_yaw=True,
            use_gripper=True,
        ),
        GraspOption("grasp_block", env),
        LiftVerticallyOption(
            "lift_after_grasp",
            env,
            block_pos,
            target_height=0.36,
            target_yaw_fn=target_yaw,
            gripper_state=1,
            min_norm=min_norm,
            reward_type=args.reward_type,
        ),
        MoveToPositionOption(
            "move_above_target",
            env,
            target_above_pos,
            target_yaw,
            gripper_state=1,
            min_norm=min_norm,
            reward_type=args.reward_type,
            use_position=True,
            use_yaw=False,
            use_gripper=True,
        ),
        MoveToPositionOption(
            "move_to_target",
            env,
            target_pos,
            target_yaw,
            gripper_state=1,
            min_norm=min_norm,
            reward_type=args.reward_type,
            use_position=True,
            use_yaw=False,
            use_gripper=True,
        ),
        ReleaseOption("release", env),
        LiftVerticallyOption(
            "lift_after_release",
            env,
            block_pos,
            target_height=0.32,
            target_yaw_fn=target_yaw,
            gripper_state=-1,
            min_norm=min_norm,
            reward_type=args.reward_type,
        ),
        MoveToPositionOption(
            "move_to_final",
            env,
            get_final_pos,
            get_final_yaw,
            gripper_state=-1,
            min_norm=min_norm,
            reward_type=args.reward_type,
            use_position=True,
            use_yaw=False,
            use_gripper=False,
        ),
    ]

    return {opt.name: opt for opt in options}


def setup_experiment(args: Args) -> str:
    """Init wandb, seed all RNGs, and return the run name."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}__{timestamp}"

    wandb.init(
        project="td3_continuous_action",
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
    """Create the (non-vectorized) cube env and return it with action/obs metadata."""
    env = make_cube_env(
        args.env_id,
        args.seed,
        args.max_episode_steps,
        args.task_id,
        noise_initial_state=args.noise_initial_state,
        reward_is_neg_dist=args.reward_is_neg_dist,
    )
    env.action_space.seed(args.seed)

    n_act = math.prod(env.action_space.shape)
    n_obs = OBS_DIM
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    action_low, action_high = float(env.action_space.low[0]), float(env.action_space.high[0])
    logging.info(f"action space: {env.action_space}")
    logging.info(f"observation space (logical): shape=({OBS_DIM},)")

    return env, n_obs, n_act, action_low, action_high


def _reset_episode(env, args: Args, device):
    """Reset the env, sample a fresh target + option set, and return the initial obs tensor."""
    obs_raw, reset_info = env.reset()
    target_block, target_pos, target_yaw = _get_target_from_info(env, reset_info)
    obs = torch.as_tensor(
        _info_to_obs_14(reset_info, target_block, target_pos, target_yaw).reshape(1, -1),
        device=device,
        dtype=torch.float,
    )
    cube_options = _create_cube_options(env, reset_info, args)
    return obs, target_block, target_pos, target_yaw, cube_options


def train(args: Args, env, n_obs: int, n_act: int, action_low: float, action_high: float, run_name: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    networks, policy = build_td3_networks(
        env=env,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        exploration_noise=args.exploration_noise,
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

    # Validation agent (uses deterministic actor)
    val_agent = LowLevelTD3Agent(env=env, actor=actor, device=device, action_low=action_low, action_high=action_high)

    obs_raw, reset_info = env.reset(seed=args.seed)
    target_block, target_pos, target_yaw = _get_target_from_info(env, reset_info)
    obs = torch.as_tensor(
        _info_to_obs_14(reset_info, target_block, target_pos, target_yaw).reshape(1, -1),
        device=device,
        dtype=torch.float,
    )
    cube_options = _create_cube_options(env, reset_info, args)
    if args.reward_option not in cube_options:
        raise ValueError(f"Unknown reward_option '{args.reward_option}'. Available: {list(cube_options.keys())}")

    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    episode_return = 0.0

    for global_step in pbar:
        if args.run_profiling and global_step == args.profiling_start:
            args.last_time = time.time()
            args.profiling_dict = {
                "select_agent_action": 0.0,
                "step_env": 0.0,
                "compute_option_reward": 0.0,
                "add_transition_to_rb": 0.0,
                "update_td3_params": 0.0,
                "log_to_wandb": 0.0,
                "reset_env": 0.0,
            }
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action_tensor = policy(obs=obs).clamp(action_low, action_high)
            action = action_tensor[0].cpu().numpy()
        _prof_checkpoint(args, global_step, "select_agent_action")

        next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = torch.as_tensor(
            _info_to_obs_14(info, target_block, target_pos, target_yaw).reshape(1, -1),
            device=device,
            dtype=torch.float,
        )
        _prof_checkpoint(args, global_step, "step_env")

        # Replace environment rewards with sparse option-based reward from selected option
        single_next_obs = next_obs[0]
        reward_value = float(cube_options[args.reward_option].calculate_reward(single_next_obs, info))
        rewards = torch.as_tensor([[reward_value]], device=device, dtype=torch.float)
        episode_return += reward_value
        _prof_checkpoint(args, global_step, "compute_option_reward")

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
            if global_step % args.policy_frequency == 0:
                out_main.update(update_pol(data))

                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                networks.qnet_target_params.lerp_(networks.qnet_params.data, args.tau)
                networks.target_actor_params.lerp_(networks.actor_params.data, args.tau)
            _prof_checkpoint(args, global_step, "update_td3_params")

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                wandb.log({"speed": speed, **logs}, step=global_step)
            _prof_checkpoint(args, global_step, "log_to_wandb")

        if done:
            max_ep_ret = max(max_ep_ret, episode_return)
            avg_returns.append(episode_return)
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

            obs, target_block, target_pos, target_yaw, cube_options = _reset_episode(env, args, device)
            episode_return = 0.0
            _prof_checkpoint(args, global_step, "reset_env")

        if global_step > args.learning_starts and global_step % args.validation_freq == 0:
            logging.info(f"Running validation with  {len(avg_returns)} recent-episode window at step {global_step}...")

            actor.eval()
            qnet.eval()

            val_metrics = run_validation_episodes(
                env=env,
                agent=val_agent,
                num_episodes=args.num_validation_episodes,
                max_episode_steps=args.max_episode_steps,
                option=cube_options[args.reward_option],
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
            logging.info(f"    avg_episode_option_return={val_metrics['avg_episode_option_return']:.2f}")

            wandb.log(
                {
                    "val/success_rate": float(val_metrics["success_rate"]),
                    "val/completion_rate": float(val_metrics["completion_rate"]),
                    "val/avg_episode_return": float(val_metrics["avg_episode_return"]),
                    "val/avg_episode_option_return": float(val_metrics["avg_episode_option_return"]),
                },
                step=global_step,
            )

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
