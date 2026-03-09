# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

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
)
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

    # Profiling
    run_profiling: bool = False
    profiling_start: int = 0
    profiling_end: int = 1000000


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    task_id: Optional[int],
    noise_initial_state: bool,
    reward_is_neg_dist: bool,
):
    """Factory that returns a thunk creating one cube env instance."""

    def thunk():
        # Create ManipSpace cube env with task-mode configuration
        env = make_cube_env(
            env_id,
            seed,
            max_episode_steps,
            task_id,
            noise_initial_state=noise_initial_state,
            reward_is_neg_dist=reward_is_neg_dist,
        )
        # Optional video recording for the first environment
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # Episode statistics for logging
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, env, exploration_noise=1, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mu = nn.Linear(256, n_act, device=device)
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

    def explore(self, obs):
        act = self(obs)
        return act + torch.randn_like(act).mul(self.action_scale * self.exploration_noise)


def _vector_infos_to_single(envs, infos):
    """Convert vectorized infos from a SyncVectorEnv to a single-env info dict."""
    # Handle list/tuple of dicts (Gym reset with multiple envs) and dict of batched arrays.
    if isinstance(infos, (list, tuple)):
        if len(infos) == envs.num_envs:
            return infos[0]
    if isinstance(infos, dict):
        single = {}
        for k, v in infos.items():
            if isinstance(v, np.ndarray) and v.shape[0] == envs.num_envs:
                single[k] = v[0]
            else:
                single[k] = v
        return single
    return infos


# TODO: Modify to create options on every reset, to deal with the case that multiple tasks/goals are used
def _create_cube_options(envs, reset_info):
    """Create the 9 cube manipulation options used for sparse rewards.

    Returns a dict mapping option name to option instance.
    """
    # We assume a single environment in the vectorized wrapper.
    env = envs.envs[0]
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
        ),
        MoveToPositionOption(
            "move_above_target",
            env,
            target_above_pos,
            target_yaw,
            gripper_state=1,
            min_norm=min_norm,
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
        ),
        MoveToPositionOption(
            "move_to_final",
            env,
            get_final_pos,
            get_final_yaw,
            gripper_state=-1,
            min_norm=min_norm,
            use_position=True,
            use_yaw=False,
            use_gripper=False,
        ),
    ]

    return {opt.name: opt for opt in options}


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                0,
                args.capture_video,
                run_name,
                args.max_episode_steps,
                args.task_id,
                args.noise_initial_state,
                args.reward_is_neg_dist,
            )
        ]
    )
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    action_low, action_high = float(envs.single_action_space.low[0]), float(envs.single_action_space.high[0])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    logging.info(f"action space: {envs.single_action_space}")
    logging.info(f"observation space: {envs.single_observation_space}")

    actor = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    actor_detach = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
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
        target_actor = Actor(env=envs, device="meta", n_act=n_act, n_obs=n_obs)
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

    envs.single_observation_space.dtype = np.float32
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

    # TRY NOT TO MODIFY: start the game
    obs, reset_infos = envs.reset(seed=args.seed)
    reset_info = _vector_infos_to_single(envs, reset_infos)
    cube_options = _create_cube_options(envs, reset_info)
    if args.reward_option not in cube_options:
        raise ValueError(
            f"Unknown reward_option '{args.reward_option}'. "
            f"Available: {list(cube_options.keys())}"
        )
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""

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
                }
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = policy(obs=obs)
            actions = actions.clamp(action_low, action_high).cpu().numpy()
        _prof_checkpoint(args, global_step, "select_agent_action")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        _prof_checkpoint(args, global_step, "step_env")
        # logging.info(f"global_step = {global_step}")
        # logging.info(f"actions = {actions}")
        # logging.info(f"next_obs = {next_obs}")
        # logging.info(f"rewards = {rewards}")
        # logging.info(f"terminations = {terminations}")
        # logging.info(f"truncations = {truncations}")
        # logging.info(f"infos = {infos}")
        # logging.info("")

        # Replace environment rewards with sparse option-based reward from selected option
        single_next_obs = next_obs[0]
        single_info = _vector_infos_to_single(envs, infos)
        reward_value = float(cube_options[args.reward_option].calculate_reward(single_next_obs, single_info))
        rewards = [reward_value]
        _prof_checkpoint(args, global_step, "compute_option_reward")
        # logging.info(f"reward_value from option {args.reward_option} = {reward_value}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # TODO: Modify to ensure correct handling of end of episode
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"].reshape(()))
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            )

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        # TODO: Modify to ensure correct handling of end of episode
        if "final_observation" in infos:
            real_next_obs[truncations] = torch.as_tensor(
                np.asarray(list(infos["final_observation"][truncations]), dtype=np.float32), device=device, dtype=torch.float
            )
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
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

    envs.close()
