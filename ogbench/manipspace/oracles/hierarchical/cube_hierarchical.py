import numpy as np

from ogbench.manipspace.oracles.hierarchical.hierarchical_oracle import HierarchicalOracle
from ogbench.manipspace.oracles.hierarchical.cube_options import (
    MoveToPositionOption,
    GraspOption,
    ReleaseOption,
)


class CubeHierarchicalOracle(HierarchicalOracle):
    """Hierarchical oracle for cube manipulation using options.
    
    This demonstrates how to use the hierarchical framework. The high-level
    policy selects options that encapsulate common manipulation behaviors.
    """
    
    def __init__(self, max_step=200, *args, **kwargs):
        """Initialize the hierarchical cube oracle."""
        super().__init__(options=[], *args, **kwargs)
        self._max_step = max_step
        self._step = 0
        
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
        
        # Option 4: Move above target
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
        
        # Option 5: Move to target
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
        
        # Option 6: Release
        self._options.append(ReleaseOption('release', self._env))
        
        # Option 7: Move to final position
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
        
        This implements a simple rule-based high-level policy. In a learned
        hierarchical RL system, this would be replaced with a learned policy.
        """
        effector_pos = info['proprio/effector_pos']
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        gripper_open = info['proprio/gripper_contact'] < 0.1
        
        block_pos = info[f'privileged/block_{self._target_block}_pos']
        target_pos = info['privileged/target_block_pos']
        
        # Check current state
        xy_aligned = np.linalg.norm(block_pos[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= 0.02
        target_pos_aligned = np.linalg.norm(target_pos - block_pos) <= 0.02
        final_pos_aligned = np.linalg.norm(self._final_pos - effector_pos) <= 0.04
        
        # High-level policy: select appropriate option based on state
        if not target_pos_aligned:
            # Need to pick up and move the block
            if not xy_aligned:
                return self._options[0]  # move_above_block
            elif not pos_aligned:
                return self._options[1]  # move_to_block
            elif not gripper_closed:
                return self._options[2]  # grasp_block
            else:
                # Block is grasped, move to target
                above_threshold = effector_pos[2] > 0.16
                target_xy_aligned = np.linalg.norm(target_pos[:2] - block_pos[:2]) <= 0.04
                
                if not above_threshold or not target_xy_aligned:
                    return self._options[3]  # move_above_target
                else:
                    return self._options[4]  # move_to_target
        else:
            # Block is at target, release and move away
            if not gripper_open:
                return self._options[5]  # release
            elif not final_pos_aligned:
                return self._options[6]  # move_to_final
            else:
                # Task complete
                self._done = True
                # Return a no-op action
                return np.zeros(5)
        
    def select_action(self, ob, info):
        """Select action (handles options automatically)."""
        action = super().select_action(ob, info)
        
        self._step += 1
        if self._step >= self._max_step:
            self._done = True
            
        return action

