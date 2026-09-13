"""Shared TD3 building blocks used by train_antmaze_lowlevel_td3.py and train_cube_lowlevel_td3.py."""
import os
from dataclasses import dataclass

from loguru import logger as logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule


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


def _log_param_counts(name: str, model: nn.Module) -> None:
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"{name} parameter count: {num_params:,}")


@dataclass
class TD3Networks:
    """Bundle of actor/critic modules, functional params, and optimizers for TD3.

    Does not include the exploration policy: that gets wrapped by ``apply_compile_and_cudagraphs``
    after construction, so callers should thread the value returned by ``build_td3_networks``
    (and then by ``apply_compile_and_cudagraphs``) through as a local rather than reaching into
    this bundle, to avoid ending up with a stale, unwrapped copy.
    """

    actor: Actor
    qnet: QNetwork
    qnet_params: TensorDict
    qnet_target_params: TensorDict
    actor_params: TensorDict
    target_actor_params: TensorDict
    target_actor: Actor
    q_optimizer: optim.Optimizer
    actor_optimizer: optim.Optimizer
    action_scale: torch.Tensor


def build_td3_networks(
    env,
    n_obs: int,
    n_act: int,
    device,
    exploration_noise: float,
    learning_rate: float,
    cudagraphs: bool,
    compile: bool,
):
    """Construct actor/critic networks, functional param copies, and their optimizers.

    Returns ``(networks, policy)`` — ``policy`` is the (uncompiled) exploration policy, kept
    separate from ``TD3Networks`` since ``apply_compile_and_cudagraphs`` replaces it with a
    wrapped version that callers must thread through explicitly.
    """
    actor = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=exploration_noise)
    actor_detach = Actor(env=env, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=exploration_noise)
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qf1 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
    qf2 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
    qnet_params = from_modules(qf1, qf2, as_module=True)
    qnet_target_params = qnet_params.data.clone()
    qnet = QNetwork(n_obs=n_obs, n_act=n_act, device="meta")
    qnet_params.to_module(qnet)

    target_actor = Actor(env=env, device="meta", n_act=n_act, n_obs=n_obs)
    actor_params = from_module(actor).data
    target_actor_params = actor_params.clone()
    target_actor_params.to_module(target_actor)

    _log_param_counts("Actor", actor)
    _log_param_counts("Q-network (per copy)", qnet)

    q_optimizer = optim.Adam(
        qnet_params.values(include_nested=True, leaves_only=True),
        lr=learning_rate,
        capturable=cudagraphs and not compile,
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate, capturable=cudagraphs and not compile)

    networks = TD3Networks(
        actor=actor,
        qnet=qnet,
        qnet_params=qnet_params,
        qnet_target_params=qnet_target_params,
        actor_params=actor_params,
        target_actor_params=target_actor_params,
        target_actor=target_actor,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        action_scale=target_actor.action_scale,
    )
    return networks, policy


def make_update_fns(networks: TD3Networks, gamma: float, policy_noise: float, noise_clip: float, action_low: float, action_high: float):
    """Return (update_main, update_pol) closures for one TD3 critic/actor update step."""
    qnet = networks.qnet
    qnet_params = networks.qnet_params
    qnet_target_params = networks.qnet_target_params
    target_actor = networks.target_actor
    actor = networks.actor
    action_scale = networks.action_scale
    q_optimizer = networks.q_optimizer
    actor_optimizer = networks.actor_optimizer

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                return F.mse_loss(vals.view(-1), next_q_value)
            return vals

    def update_main(data):
        observations = data["observations"]
        next_observations = data["next_observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(policy_noise).clamp(-noise_clip, noise_clip).mul(action_scale)

        next_state_actions = (target_actor(next_observations) + clipped_noise).clamp(action_low, action_high)

        qf_next_target = torch.vmap(batched_qf, (0, None, None))(qnet_target_params, next_observations, next_state_actions)
        min_qf_next_target = qf_next_target.min(0).values
        next_q_value = rewards.flatten() + gamma * min_qf_next_target.flatten()

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

    return update_main, update_pol


def apply_compile_and_cudagraphs(policy, update_main, update_pol, compile: bool, cudagraphs: bool):
    """Optionally wrap policy/update functions with torch.compile and/or CudaGraphModule."""
    if compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[], warmup=5)
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[], warmup=5)
        policy = CudaGraphModule(policy)

    return policy, update_main, update_pol


def compute_exploration_noise(
    global_step: int,
    learning_starts: int,
    total_timesteps: int,
    start_noise: float,
    end_noise: float,
) -> float:
    """Linearly anneal exploration noise from ``start_noise`` to ``end_noise`` by ``total_timesteps / 2``.

    Passing ``start_noise == end_noise`` yields constant noise.
    """
    anneal_frac = min(1.0, (global_step - learning_starts) / (total_timesteps / 2 - learning_starts))
    return start_noise + anneal_frac * (end_noise - start_noise)


def save_td3_checkpoint(
    global_step: int,
    actor,
    target_actor_params,
    qnet_params,
    qnet_target_params,
    actor_optimizer,
    q_optimizer,
    save_path: str,
) -> None:
    checkpoint = {
        "global_step": global_step,
        "actor_state_dict": actor.state_dict(),
        "target_actor_params": target_actor_params,
        "qnet_params": qnet_params,
        "qnet_target_params": qnet_target_params,
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to: {save_path}")
