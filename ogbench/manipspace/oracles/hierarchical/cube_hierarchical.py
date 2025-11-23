from absl import logging
import numpy as np
import random

from ogbench.manipspace.oracles.hierarchical.hierarchical_oracle import HierarchicalOracle
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
    LiftVerticallyOption,
    NoOpOption,
)


class CubeHierarchicalOracle(HierarchicalOracle):
    """Hierarchical oracle for cube manipulation using options.
    
    This demonstrates how to use the hierarchical framework. The high-level
    policy selects options that encapsulate common manipulation behaviors.
    """
    
    def __init__(self, max_step=200, no_op_option_prob=0.05, suboptimal_option_prob=0.05, *args, **kwargs):
        """Initialize the hierarchical cube oracle.
        
        Args:
            max_step: Maximum number of steps per episode
            no_op_option_prob: Probability of selecting the no-op option instead of the normal option
            suboptimal_option_prob: Probability of selecting a random suboptimal option
        """
        super().__init__(options=[], *args, **kwargs)
        self._max_step = max_step
        self._step = 0
        self._no_op_option_prob = no_op_option_prob
        self._suboptimal_option_prob = suboptimal_option_prob
        
        # These will be set in reset()
        self._target_block = None
        self._final_pos = None
        self._final_yaw = None
        
    def reset(self, ob, info):
        """Reset the oracle and create options based on current task."""
        super().reset(ob, info)
        self._done = False
        self._step = 0
        self._max_step = 200
        
        # Extract task information
        self._target_block = info['privileged/target_block']
        self._final_pos = np.random.uniform(*self._env.unwrapped._arm_sampling_bounds)
        self._final_yaw = np.random.uniform(-np.pi, np.pi)
        
        # Create options for this task
        self._options = []
        
        # Option 0: No-op option (always at index 0)
        self._options.append(
            NoOpOption(
                'no_op',
                self._env,
                duration=10,
            )
        )
        
        # Option 1: Move above block
        def block_above_pos(ob, info):
            block_pos = info[f'privileged/block_{self._target_block}_pos']
            return block_pos + np.array([0, 0, 0.18])
        
        def block_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            block_yaw = info[f'privileged/block_{self._target_block}_yaw'][0]
            # Use shortest_yaw from parent class
            return self.shortest_yaw(effector_yaw, block_yaw)
        
        self._options.append(
            MoveToPositionOption(
                'move_above_block',
                self._env,
                block_above_pos,
                block_yaw,
                gripper_state=-1,
                min_norm=self._min_norm,
            )
        )
        
        # Option 2: Move to block
        def block_pos(ob, info):
            return info[f'privileged/block_{self._target_block}_pos']
        
        self._options.append(
            MoveToPositionOption(
                'move_to_block',
                self._env,
                block_pos,
                block_yaw,
                gripper_state=-1,
                min_norm=self._min_norm,
            )
        )
        
        # Option 3: Grasp block
        self._options.append(
            GraspOption(
                'grasp_block',
                self._env,
                block_pos,
                block_yaw,
                min_norm=self._min_norm,
            )
        )
        
        # Option 4: Move in the air (lift vertically after grasping)
        def block_base_pos(ob, info):
            # Keep x,y at block position, lift z
            return info[f'privileged/block_{self._target_block}_pos']
        
        def target_yaw_for_lift(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            block_yaw = info[f'privileged/block_{self._target_block}_yaw'][0]
            target_yaw = info['privileged/target_block_yaw'][0]
            # Rotate from block_yaw to target_yaw
            return self.shortest_yaw(effector_yaw, target_yaw)
        
        self._options.append(
            LiftVerticallyOption(
                'lift_after_grasp',
                self._env,
                block_base_pos,
                target_height=0.36,  # block_above_offset[2] * 2 = 0.18 * 2
                target_yaw_fn=target_yaw_for_lift,
                gripper_state=1,  # Keep gripper closed
                min_norm=self._min_norm,
            )
        )
        
        # Option 5: Move above target
        def target_above_pos(ob, info):
            target_pos = info['privileged/target_block_pos']
            return target_pos + np.array([0, 0, 0.18])
        
        def target_yaw(ob, info):
            effector_yaw = info['proprio/effector_yaw'][0]
            target_yaw = info['privileged/target_block_yaw'][0]
            return self.shortest_yaw(effector_yaw, target_yaw)
        
        self._options.append(
            MoveToPositionOption(
                'move_above_target',
                self._env,
                target_above_pos,
                target_yaw,
                gripper_state=1,  # Keep gripper closed
                min_norm=self._min_norm,
            )
        )
        
        # Option 6: Move to target
        def target_pos(ob, info):
            return info['privileged/target_block_pos']
        
        self._options.append(
            MoveToPositionOption(
                'move_to_target',
                self._env,
                target_pos,
                target_yaw,
                gripper_state=1,  # Keep gripper closed
                min_norm=self._min_norm,
            )
        )
        
        # Option 7: Release
        self._options.append(ReleaseOption('release', self._env))
        
        # Option 8: Move in the air (lift vertically after releasing)
        def block_base_pos_after_release(ob, info):
            # Keep x,y at block position, lift z
            return info[f'privileged/block_{self._target_block}_pos']
        
        def final_yaw_for_lift(ob, info):
            return self._final_yaw
        
        self._options.append(
            LiftVerticallyOption(
                'lift_after_release',
                self._env,
                block_base_pos_after_release,
                target_height=0.32,  # above_threshold * 2 = 0.16 * 2
                target_yaw_fn=final_yaw_for_lift,
                gripper_state=-1,  # Keep gripper open
                min_norm=self._min_norm,
            )
        )
        
        # Option 9: Move to final position
        def final_pos(ob, info):
            return self._final_pos
        
        def final_yaw(ob, info):
            return self._final_yaw
        
        self._options.append(
            MoveToPositionOption(
                'move_to_final',
                self._env,
                final_pos,
                final_yaw,
                gripper_state=-1,
                min_norm=self._min_norm,
            )
        )
        
    def select_high_level_action(self, ob, info):
        """Select high-level action (option or primitive).
        
        This implements a simple rule-based high-level policy with stochasticity.
        With probability no_op_option_prob, selects the no-op option.
        With probability suboptimal_option_prob, selects a random suboptimal option.
        Otherwise, selects the optimal option based on the current state.
        
        In a learned hierarchical RL system, this would be replaced with a learned policy.
        """
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        gripper_open = info['proprio/gripper_contact'] < 0.1
        
        block_pos = info[f'privileged/block_{self._target_block}_pos']
        target_pos = info['privileged/target_block_pos']
        
        # Constants from cube_markov.py
        above_threshold = 0.16
        
        # Check current state
        above = effector_pos[2] > above_threshold
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04
        
        # High-level policy: select appropriate option based on state
        # Note: No-op is at index 0, normal options are at indices 1-9
        if not target_pos_aligned:
            # Phases 1-6: Pick up and move block to target
            if not xy_aligned:
                # Phase 1: Move above the block
                normal_option = self._options[1]
            elif not pos_aligned:
                # Phase 2: Move to the block
                normal_option = self._options[2]
            elif pos_aligned and not gripper_closed:
                # Phase 3: Grasp block
                normal_option = self._options[3]
            elif pos_aligned and gripper_closed and not above and not target_xy_aligned:
                # Phase 4: Move in the air after grasping (lift vertically)
                normal_option = self._options[4]
            elif pos_aligned and gripper_closed and above and not target_xy_aligned:
                # Phase 5: Move above the target
                normal_option = self._options[5]
            else:
                # Phase 6: Move to the target
                normal_option = self._options[6]
        else:
            # Phases 7-9: Block is at target, release and move away
            if not gripper_open:
                # Phase 7: Release
                normal_option = self._options[7]
            elif gripper_open and not above:
                # Phase 8: Move in the air after releasing (lift vertically)
                normal_option = self._options[8]
            else:
                # Phase 9: Move to the final position
                if final_pos_aligned:
                    self._done = True
                normal_option = self._options[9]
        
        # Apply stochasticity: with some probability, replace with no-op or suboptimal option
        rand_val = random.random()
        no_op_option_idx = 0  # No-op is always at index 0
        if rand_val < self._no_op_option_prob:
            return self._options[no_op_option_idx]
        elif rand_val < (self._no_op_option_prob + self._suboptimal_option_prob):
            # Filter to only options that can be initiated (excluding the normal one and no-op)
            normal_option_idx = self._options.index(normal_option)
            available_options = [
                (i, opt) for i, opt in enumerate(self._options)
                if i != normal_option_idx and i != no_op_option_idx and opt.can_initiate(ob, info)
            ]
            
            if len(available_options) > 0:
                # Randomly select from available options that can be initiated
                selected_idx, _ = random.choice(available_options)
                return self._options[selected_idx]
            else:
                # No valid suboptimal options available, fall back to normal option
                return normal_option
        else:
            return normal_option
        
    def select_action(self, ob, info):
        """Select action (handles options automatically)."""
        action = super().select_action(ob, info)
        
        self._step += 1
        if self._step >= self._max_step:
            self._done = True
            
        return action

