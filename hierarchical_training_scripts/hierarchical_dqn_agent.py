import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import random
from ogbench.manipspace.oracles.hierarchical.hierarchical_agent import HierarchicalAgent
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
    LiftVerticallyOption,
    NoOpOption,
)


class QNetwork(nn.Module):
    """Q-Network that maps observations to Q-values for each option."""
    
    def __init__(self, obs_dim: int, num_options: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )
    
    def forward(self, x):
        return self.network(x)


class HierarchicalDQNAgent(HierarchicalAgent):
    """Hierarchical agent that uses Q-network for high-level option selection.
    
    Extends HierarchicalAgent to use DQN for learning which option to execute,
    while the low-level option execution is handled by pre-defined options.
    
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
    
    def __init__(
        self,
        env,
        q_network: QNetwork,
        device: torch.device,
        disable_no_op: bool = False,
        no_op_duration: int = 10,
    ):
        super().__init__(options=[], env=env, min_norm=0.08)
        self.q_network = q_network
        self.device = device
        self.disable_no_op = disable_no_op
        self.no_op_duration = no_op_duration
        self.epsilon = 0.0  # Default to greedy if not set
        
        # Will be initialized in reset()
        self._target_block = None
        self._target_pos = None
        self._target_yaw = None
        self._final_pos = None
        self._final_yaw = None
    
    def reset(self, ob, info):
        """Reset the agent for a new episode/task.
        
        Args:
            ob: Initial observation
            info: Info dict from environment
        """
        super().reset(ob, info)
        
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
    
    def _create_options(self) -> List:
        """Create the set of options for cube manipulation.
        
        Returns:
            List of 10 options (or 9, if disable_no_op is True):
                0: no_op - Do nothing for N steps
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
        """Select which option to execute using epsilon-greedy Q-learning.
        
        This method is called by the base HierarchicalAgent when no option is active.
        For training with epsilon-greedy exploration, set self.epsilon before calling.
        
        Args:
            ob: Current observation
            info: Info dict from environment
            
        Returns:
            The selected Option object to execute
        """
        obs = self.get_obs_tensor(info).unsqueeze(0)
        
        if random.random() < self.epsilon:
            # Random exploration
            action_idx = random.randint(0, len(self._options) - 1)
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                q_values = self.q_network(obs)
                action_idx = torch.argmax(q_values, dim=1).item()
        
        # Store for training (observation and action index)
        self.last_obs = obs.squeeze(0)
        self.last_action_idx = action_idx
        
        return self._options[action_idx]
    
    def select_action(self, ob, info):
        """Select action and track when new high-level decisions are made.
        
        This override adds transition tracking for DQN training while using the
        base class's option execution logic.
        """
        # Check if we're about to select a new option
        will_select_new_option = (self._active_option is None or not self._active_option.active)
        
        if will_select_new_option:
            # Signal that a new high-level decision is about to be made
            self._new_option_selected = True
        else:
            self._new_option_selected = False
        
        # Call base class to handle option selection and execution
        action = super().select_action(ob, info)
        
        return action
    
    def get_last_transition_info(self) -> Tuple[torch.Tensor, int]:
        """Get the observation and action index from the last high-level decision.
        
        Returns:
            obs: Observation tensor used for selection
            action_idx: Option index that was selected
        """
        return self.last_obs, self.last_action_idx
    
    def was_new_option_selected(self) -> bool:
        """Check if a new option was just selected in the last select_action call.
        
        Returns:
            True if a new high-level decision was made
        """
        assert hasattr(self, '_new_option_selected'), "New option selected attribute not found"
        return self._new_option_selected
