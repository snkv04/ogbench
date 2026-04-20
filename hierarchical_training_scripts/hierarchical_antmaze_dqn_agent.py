"""High-level DQN agent and TD3-backed options for ant maze hierarchical RL."""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger as logging

from ogbench.manipspace.oracles.hierarchical.hierarchical_agent import HierarchicalAgent
from ogbench.manipspace.oracles.hierarchical.option import Option

from hierarchical_training_scripts.hierarchical_dqn_agent import QNetwork
from hierarchical_training_scripts.train_cube_lowlevel_td3 import Actor


class MoveInDirectionOption(Option):
    """Runs a frozen TD3 actor with a fixed prefix (option-start xy or full obs) + current base obs."""

    def __init__(
        self,
        name: str,
        env,
        actor: Actor,
        device: torch.device,
        checkpoint_path: str,
        termination_time: int,
        concatenate_only_xy: bool,
        action_low: float,
        action_high: float,
    ):
        super().__init__(name, env)
        self._actor = actor
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._termination_time = int(termination_time)
        self._concatenate_only_xy = concatenate_only_xy
        self._action_low = action_low
        self._action_high = action_high

        base_dim = int(np.prod(env.observation_space.shape))
        self._base_obs_dim = base_dim
        prefix_dim = 2 if concatenate_only_xy else base_dim
        self._prefix_dim = prefix_dim
        self._td3_obs_dim = prefix_dim + base_dim

        self._prefix: np.ndarray | None = None

    def load_actor_weights(self) -> None:
        """Load actor weights from ``checkpoint_path`` (call once after construction)."""
        try:
            ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
        except TypeError:
            logging.info(f"Failed to load from path {self._checkpoint_path} with weights_only=False, reloading.")
            ckpt = torch.load(self._checkpoint_path, map_location=self._device)
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._actor.eval()
        for p in self._actor.parameters():
            p.requires_grad = False

    def can_initiate(self, ob, info) -> bool:
        return True

    def initiate(self, ob, info):
        super().initiate(ob, info)
        ob = np.asarray(ob, dtype=np.float64).ravel()
        if ob.shape[0] != self._base_obs_dim:
            raise ValueError(
                f"MoveInDirectionOption.initiate: expected base obs dim {self._base_obs_dim}, got {ob.shape[0]}"
            )
        if self._concatenate_only_xy:
            xy = self._env.unwrapped.get_xy()
            self._prefix = np.asarray(xy, dtype=np.float64).ravel()[:2].astype(np.float64)
        else:
            self._prefix = ob.copy().astype(np.float64)

    def _build_td3_obs(self, ob: np.ndarray) -> torch.Tensor:
        ob = np.asarray(ob, dtype=np.float64).ravel()
        assert self._prefix is not None
        full = np.concatenate([self._prefix, ob]).astype(np.float32)
        return torch.as_tensor(full, device=self._device).unsqueeze(0)

    def select_action(self, ob, info):
        obs_t = self._build_td3_obs(ob)
        with torch.no_grad():
            a = self._actor(obs_t).clamp(self._action_low, self._action_high)
        out = a[0].cpu().numpy()
        # logging.info(
        #     f"[antmaze LL] option={self.name!r} obs_shape={tuple(obs_t.shape)} action_shape={out.shape}"
        # )
        return out

    def is_terminated(self, ob, info) -> bool:
        return self._step >= self._termination_time

    def reset(self):
        super().reset()
        self._prefix = None


class NoOpOption(Option):
    """Option that sends zero torques for ``termination_time`` steps."""

    def __init__(self, name: str, env, termination_time: int, action_low: float, action_high: float):
        super().__init__(name, env)
        self._termination_time = int(termination_time)
        self._n_act = int(np.prod(env.action_space.shape))
        self._zero_action = np.zeros(self._n_act, dtype=np.float32)

    def can_initiate(self, ob, info) -> bool:
        return True

    def initiate(self, ob, info):
        super().initiate(ob, info)

    def select_action(self, ob, info):
        return self._zero_action.copy()

    def is_terminated(self, ob, info) -> bool:
        return self._step >= self._termination_time

    def reset(self):
        super().reset()


class HierarchicalAntMazeDQNAgent(HierarchicalAgent):
    """Epsilon-greedy DQN over a fixed list of ``MoveInDirectionOption`` instances."""

    def __init__(
        self,
        env,
        q_network: QNetwork,
        device: torch.device,
        options: List[MoveInDirectionOption],
    ):
        super().__init__(options=options, env=env, min_norm=0.08)
        self.q_network = q_network
        self.device = device
        self.epsilon = 0.0
        self._base_obs_dim = int(np.prod(env.observation_space.shape))

    def reset(self, ob, info):
        super().reset(ob, info)

    def get_obs_tensor(self, ob: np.ndarray, info=None) -> torch.Tensor:
        """High-level Q input: same vector the env returns (e.g. shape (29,)) at option boundaries."""
        ob = np.asarray(ob, dtype=np.float32).ravel()
        if ob.shape[0] != self._base_obs_dim:
            raise ValueError(
                f"get_obs_tensor: expected env obs dim {self._base_obs_dim}, got {ob.shape[0]}"
            )
        return torch.as_tensor(ob, dtype=torch.float32, device=self.device)

    def select_high_level_action(self, ob, info):
        obs = self.get_obs_tensor(ob, info).unsqueeze(0)
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self._options) - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(obs)
                action_idx = torch.argmax(q_values, dim=1).item()
        self.last_obs = obs.squeeze(0)
        self.last_action_idx = action_idx
        chosen = self._options[action_idx]
        # logging.info(
        #     f"[antmaze HL] obs_shape={tuple(obs.shape)} action_index={action_idx} "
        #     f"action_shape={np.asarray(action_idx).shape} option={chosen.name!r}"
        # )
        return chosen

    def select_action(self, ob, info):
        will_select_new_option = self._active_option is None or not self._active_option.active
        self._new_option_selected = bool(will_select_new_option)
        return super().select_action(ob, info)

    def get_last_transition_info(self) -> Tuple[torch.Tensor, int]:
        return self.last_obs, self.last_action_idx

    def was_new_option_selected(self) -> bool:
        assert hasattr(self, "_new_option_selected")
        return self._new_option_selected
