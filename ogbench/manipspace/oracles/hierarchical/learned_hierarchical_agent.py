"""Learned hierarchical agent for cube manipulation using a neural network policy."""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ogbench.manipspace.oracles.hierarchical.hierarchical_agent import HierarchicalAgent
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
    LiftVerticallyOption,
    NoOpOption,
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """Actor-critic network for high-level option selection."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256, device=None):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim, device=device)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, num_actions, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1, device=device), std=1.0)

    def get_value(self, obs):
        return self.critic(self.network(obs))

    def get_action_and_value(self, obs, action=None, deterministic=False, temperature=1.0):
        """Get action and value from the policy.
        
        Args:
            obs: Observation tensor
            action: Optional action to evaluate (if None, samples new action)
            deterministic: If True, use argmax instead of sampling
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            action, log_prob, entropy, value
        """
        hidden = self.network(obs)
        logits = self.actor(hidden) / temperature  # If deterministic, dividing by temperature doesn't change action
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
                # print(f"Deterministic action: {action}")
                # print(f"Logits: {logits}")
                # print(f"Probs: {probs}")
            else:
                action = probs.sample()
                # print(f"Sampled action: {action}")
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class LearnedHierarchicalAgent(HierarchicalAgent):
    """Hierarchical agent with a learned high-level policy for option selection.
    
    This agent uses a neural network (PolicyNetwork) to select which option to execute,
    rather than following a deterministic oracle. The low-level option execution is still
    handled by the pre-defined options (MoveToPosition, Grasp, etc.).
    
    Attributes:
        OBS_DIM: Dimension of observation features for the policy (14)
            - effector_pos (3)
            - effector_yaw (1)
            - gripper_opening (1)
            - gripper_contact (1)
            - block_pos (3)
            - block_yaw (1)
            - target_pos (3)
            - target_yaw (1)
    """

    OBS_DIM = 14  # effector_pos(3) + effector_yaw(1) + gripper(2) + block(4) + target(4)

    def __init__(self, env, policy_network: PolicyNetwork, device: torch.device,
                 deterministic: bool = False, temperature: float = 1.0,
                 disable_no_op: bool = False, no_op_duration: int = 10):
        """Initialize the learned hierarchical agent.
        
        Args:
            env: The gymnasium environment
            policy_network: Neural network for option selection
            device: Torch device (cpu or cuda)
            deterministic: If True, use argmax for option selection
            temperature: Temperature for sampling options (higher = more random)
            disable_no_op: If True, disable the no-op option (index 0)
            no_op_duration: Duration of the no-op option
        """
        super().__init__(options=[], env=env)
        self.policy_network = policy_network
        self.device = device
        self.deterministic = deterministic
        self.temperature = temperature
        self._target_block = None
        self._final_pos = None
        self._final_yaw = None
        self.last_decision = None  # Stores {obs, action, logprob, value} for PPO
        self.disable_no_op = disable_no_op
        self.no_op_duration = no_op_duration

    def reset(self, ob, info):
        """Reset the agent for a new episode/task.
        
        Args:
            ob: Initial observation
            info: Info dict from environment
        """
        super().reset(ob, info)
        self._done = False  # Reset done flag for new task
        
        env = self._env.unwrapped
        
        # Handle both data_collection and task modes
        if env._mode == 'data_collection':
            # In data_collection mode, target info is in privileged keys
            self._target_block = info['privileged/target_block']
            self._target_pos = info['privileged/target_block_pos'].copy()
            self._target_yaw = info['privileged/target_block_yaw'][0]
        else:
            # In task mode, target info is in cur_task_info
            # For cube-single, there's only 1 cube, so target_block = 0
            self._target_block = 0
            self._target_pos = env.cur_task_info['goal_xyzs'][self._target_block].copy()
            self._target_yaw = 0.0  # Task mode uses identity orientation for goals
        
        self._final_pos = np.random.uniform(*env._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)
        self._options = self._create_options()
        self.last_decision = None

    def _create_options(self) -> List:
        """Create the set of options for cube manipulation.
        
        Returns:
            List of 10 options (or 9, if disable_no_op is True):
                0: no_op - Do nothing for 10 steps
                1: move_above_block - Move end-effector above the target block
                2: move_to_block - Move end-effector to the target block
                3: grasp_block - Close gripper to grasp the block
                4: lift_after_grasp - Lift the grasped block
                5: move_above_target - Move above the target position
                6: move_to_target - Move to the target position
                7: release - Open gripper to release block
                8: lift_after_release - Lift after releasing
                9: move_to_final - Move to final position
        """
        env = self._env
        target_block = self._target_block
        final_pos = self._final_pos
        final_yaw = self._final_yaw

        def block_above_pos(ob, info):
            return info[f'privileged/block_{target_block}_pos'] + np.array([0, 0, 0.18])

        def block_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            block_yaw_val = info[f'privileged/block_{target_block}_yaw'][0]
            return self._shortest_yaw(effector_yaw, block_yaw_val)

        def block_pos(ob, info):
            return info[f'privileged/block_{target_block}_pos']

        # Use stored target info (works for both data_collection and task modes)
        stored_target_pos = self._target_pos
        stored_target_yaw = self._target_yaw

        def target_above_pos(ob, info):
            return stored_target_pos + np.array([0, 0, 0.18])

        def target_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            return self._shortest_yaw(effector_yaw, stored_target_yaw)

        def target_pos(ob, info):
            return stored_target_pos

        def get_final_pos(ob, info):
            return final_pos

        def get_final_yaw(ob, info):
            return final_yaw

        options = [
            MoveToPositionOption('move_above_block', env, block_above_pos, block_yaw, gripper_state=-1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_block', env, block_pos, block_yaw, gripper_state=-1, min_norm=self._min_norm),
            GraspOption('grasp_block', env, block_pos, block_yaw, min_norm=self._min_norm),
            LiftVerticallyOption('lift_after_grasp', env, block_pos, target_height=0.36, target_yaw_fn=target_yaw, gripper_state=1, min_norm=self._min_norm),
            MoveToPositionOption('move_above_target', env, target_above_pos, target_yaw, gripper_state=1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_target', env, target_pos, target_yaw, gripper_state=1, min_norm=self._min_norm),
            ReleaseOption('release', env),
            LiftVerticallyOption('lift_after_release', env, block_pos, target_height=0.32, target_yaw_fn=get_final_yaw, gripper_state=-1, min_norm=self._min_norm),
            MoveToPositionOption('move_to_final', env, get_final_pos, get_final_yaw, gripper_state=-1, min_norm=self._min_norm),
        ]
        if not self.disable_no_op:
            options.insert(0, NoOpOption('no_op', env, duration=self.no_op_duration))
        return options

    @staticmethod
    def _shortest_yaw(current_yaw: float, target_yaw: float) -> float:
        """Compute shortest rotation to reach target yaw."""
        diff = target_yaw - current_yaw
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return current_yaw + diff

    def get_obs_tensor(self, info) -> torch.Tensor:
        """Extract observation features for the high-level policy.
        
        Args:
            info: Info dict from environment step
            
        Returns:
            Tensor of shape (14,) containing:
                - effector_pos (3)
                - effector_yaw (1)
                - gripper_opening (1)
                - gripper_contact (1)
                - block_pos (3)
                - block_yaw (1)
                - target_pos (3)
                - target_yaw (1)
        """
        # Use stored target info (works for both data_collection and task modes)
        features = np.concatenate([
            info['proprio/effector_pos'],
            np.atleast_1d(info['proprio/effector_yaw']),
            np.atleast_1d(info['proprio/gripper_opening']),
            np.atleast_1d(info['proprio/gripper_contact']),
            info[f'privileged/block_{self._target_block}_pos'],
            np.atleast_1d(info[f'privileged/block_{self._target_block}_yaw']),
            self._target_pos,
            np.atleast_1d(self._target_yaw),
        ])
        return torch.as_tensor(features, dtype=torch.float32, device=self.device)

    def select_high_level_action(self, ob, info):
        """Select which option to execute using the learned policy.
        
        This method is called by the base HierarchicalAgent when no option is active
        or the current option has terminated, but it can also be called manually to
        select an action.
        
        Args:
            ob: Current observation
            info: Info dict from environment
            
        Returns:
            The selected Option object to execute
        """        
        obs_tensor = self.get_obs_tensor(info).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = self.policy_network.get_action_and_value(
                obs_tensor,
                deterministic=self.deterministic,
                temperature=self.temperature
            )

        # Store decision for PPO training to collect
        self.last_decision = {
            'obs': obs_tensor.squeeze(0),
            'action': action,
            'logprob': logprob,
            'value': value.flatten(),
        }

        # Clamp action index to valid range (in case policy outputs invalid index)
        option_idx = min(action.item(), len(self._options) - 1)
        return self._options[option_idx]

    def select_action(self, ob, info):
        """Select action (handles options automatically)."""
        action = super().select_action(ob, info)
        
        # Check if task is complete (cube aligned with target)
        # Using 4cm threshold, same as environment's success threshold
        block_pos = info[f'privileged/block_{self._target_block}_pos']
        target_aligned = np.linalg.norm(self._target_pos - block_pos) <= 0.04
        self._done = target_aligned
            
        return action
