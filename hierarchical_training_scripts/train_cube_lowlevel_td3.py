# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

from datetime import datetime
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
from loguru import logger as logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

import ogbench.manipspace  # Register environments
from hierarchical_training_scripts.train_cube_hrl_dqn import (
    _log_param_counts,
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
    profiling_end: int = 1000000


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, env, exploration_noise=1, device=None, h_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.fc_mu = nn.Linear(h_dim, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer("exploration_noise", torch.as_tensor(exploration_noise, device=device))

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = self.fc_mu(obs).tanh()
        return obs * self.action_scale + self.action_bias

    def explore(self, obs, exploration_noise=None):
        act = self(obs)
        noise = self.exploration_noise if exploration_noise is None else exploration_noise
        return act + torch.randn_like(act).mul(self.action_scale * noise)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, device=None, h_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}__{timestamp}"

    wandb.init(
        project="td3_continuous_action",
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

    # env setup: single non-vectorized env created via make_cube_env (no extra wrappers)
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

    actor = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    actor_detach = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    def get_params_qnet():
        qf1 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)

        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        # discard params of net
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
        # Note: We don't use the done signal here, in order to capture the infinite horizon
        next_q_value = rewards.flatten() + args.gamma * min_qf_next_target.flatten()

        qf_loss = torch.vmap(batched_qf, (0, None, None, None))(qnet_params, observations, actions, next_q_value)
        qf_loss = qf_loss.sum(0)

        # optimize the model
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
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[], warmup=5)
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[], warmup=5)
        policy = CudaGraphModule(policy)

    # Validation agent (uses deterministic actor)
    val_agent = LowLevelTD3Agent(env=env, actor=actor, device=device, action_low=action_low, action_high=action_high)

    # TRY NOT TO MODIFY: start the game
    obs_raw, reset_info = env.reset(seed=args.seed)
    target_block, target_pos, target_yaw = _get_target_from_info(env, reset_info)
    obs = torch.as_tensor(
        _info_to_obs_14(reset_info, target_block, target_pos, target_yaw).reshape(1, -1),
        device=device,
        dtype=torch.float,
    )
    cube_options = _create_cube_options(env, reset_info, args)
    if args.reward_option not in cube_options:
        raise ValueError(
            f"Unknown reward_option '{args.reward_option}'. "
            f"Available: {list(cube_options.keys())}"
        )
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    episode_return = 0.0

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
                    "compute_option_reward": 0.0,
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
            # Sample random exploratory action from env.action_space
            action = env.action_space.sample()
        else:
            # Policy outputs a batch of size 1; take the single action and clamp
            action_tensor = policy(obs=obs).clamp(action_low, action_high)
            action = action_tensor[0].cpu().numpy()
        _prof_checkpoint(args, global_step, "select_agent_action")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = torch.as_tensor(
            _info_to_obs_14(info, target_block, target_pos, target_yaw).reshape(1, -1),
            device=device,
            dtype=torch.float,
        )
        _prof_checkpoint(args, global_step, "step_env")
        # logging.info(f"global_step = {global_step}")
        # logging.info(f"action = {action}")
        # logging.info(f"next_obs_raw = {next_obs_raw}")
        # logging.info(f"next_obs = {next_obs}")
        # logging.info(f"terminated = {terminated}")
        # logging.info(f"truncated = {truncated}")
        # logging.info(f"info = {info}")
        # logging.info("")

        # Replace environment rewards with sparse option-based reward from selected option
        single_next_obs = next_obs[0]
        reward_value = float(cube_options[args.reward_option].calculate_reward(single_next_obs, info))
        rewards = torch.as_tensor([[reward_value]], device=device, dtype=torch.float)
        episode_return += reward_value
        _prof_checkpoint(args, global_step, "compute_option_reward")
        # logging.info(f"env_reward = {env_reward}")
        # logging.info(f"reward_value = {reward_value}")
        # logging.info(f"rewards = {rewards}")

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

                # update the target networks
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target_params.lerp_(qnet_params.data, args.tau)
                target_actor_params.lerp_(actor_params.data, args.tau)
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
            # Handle episode-level logging
            max_ep_ret = max(max_ep_ret, episode_return)
            avg_returns.append(episode_return)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            )

            # Reset environment
            obs_raw, reset_info = env.reset()
            target_block, target_pos, target_yaw = _get_target_from_info(env, reset_info)
            obs = torch.as_tensor(
                _info_to_obs_14(reset_info, target_block, target_pos, target_yaw).reshape(1, -1),
                device=device,
                dtype=torch.float,
            )
            cube_options = _create_cube_options(env, reset_info, args)
            episode_return = 0.0
            _prof_checkpoint(args, global_step, "reset_env")

        # Optional validation at fixed step intervals
        if global_step > args.learning_starts and global_step % args.validation_freq == 0:
            logging.info(f"Running validation with  {len(avg_returns)} recent-episode window at step {global_step}...")

            # Switch to eval mode
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

            # Switch back to train mode
            actor.train()
            qnet.train()

            # Log validation metrics
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
