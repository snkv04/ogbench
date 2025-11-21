import numpy as np

from ogbench.manipspace.oracles.hierarchical.option import Option
from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class MoveToPositionOption(Option):
    """Option to move the gripper to a target position.
    
    This option encapsulates the "move to position" behavior, which
    can be reused for moving to the block, target, or final position.
    """
    
    def __init__(self, name, env, target_pos_fn, target_yaw_fn=None, 
                 gripper_state=-1, min_norm=0.4, gain_pos=5, gain_yaw=3,
                 termination_threshold=0.04):
        """Initialize the move-to-position option.
        
        Args:
            name: Option name
            env: Environment instance
            target_pos_fn: Function (ob, info) -> np.ndarray that returns target position
            target_yaw_fn: Optional function (ob, info) -> float that returns target yaw
            gripper_state: Gripper state to maintain (-1=open, 1=closed)
            min_norm: Minimum norm for position differences
            gain_pos: Position gain
            gain_yaw: Yaw gain
            termination_threshold: Distance threshold for termination
        """
        super().__init__(name, env)
        self._target_pos_fn = target_pos_fn
        self._target_yaw_fn = target_yaw_fn
        self._gripper_state = gripper_state
        self._min_norm = min_norm
        self._gain_pos = gain_pos
        self._gain_yaw = gain_yaw
        self._termination_threshold = termination_threshold
        
    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm
            
    def can_initiate(self, ob, info):
        """Can always initiate this option."""
        return True
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        """Select action to move toward target position."""
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        
        target_pos = self._target_pos_fn(ob, info)
        
        # Compute position difference
        diff = target_pos - effector_pos
        diff = self.shape_diff(diff)
        
        # Compute yaw difference
        if self._target_yaw_fn is not None:
            target_yaw = self._target_yaw_fn(ob, info)
            yaw_diff = target_yaw - effector_yaw
        else:
            yaw_diff = 0.0
        
        # Construct action
        action = np.zeros(5)
        action[:3] = diff[:3] * self._gain_pos
        action[3] = yaw_diff * self._gain_yaw
        action[4] = self._gripper_state
        
        return np.clip(action, -1, 1)
        
    def is_terminated(self, ob, info):
        """Terminate when close enough to target."""
        effector_pos = info['proprio/effector_pos']
        target_pos = self._target_pos_fn(ob, info)
        distance = np.linalg.norm(target_pos - effector_pos)
        return distance <= self._termination_threshold


class LiftVerticallyOption(Option):
    """Option to lift the gripper vertically while keeping x,y fixed.
    
    This is used for phases 4 and 8 - lifting straight up for clearance.
    """
    
    def __init__(self, name, env, base_pos_fn, target_height, target_yaw_fn=None,
                 gripper_state=1, min_norm=0.4, gain_pos=5, gain_yaw=3,
                 termination_threshold=0.04):
        """Initialize the lift vertically option.
        
        Args:
            name: Option name
            env: Environment instance
            base_pos_fn: Function (ob, info) -> np.ndarray that returns base (x,y) position
            target_height: Target z height to lift to
            target_yaw_fn: Optional function (ob, info) -> float that returns target yaw
            gripper_state: Gripper state to maintain (-1=open, 1=closed)
            min_norm: Minimum norm for position differences
            gain_pos: Position gain
            gain_yaw: Yaw gain
            termination_threshold: Distance threshold for termination
        """
        super().__init__(name, env)
        self._base_pos_fn = base_pos_fn
        self._target_height = target_height
        self._target_yaw_fn = target_yaw_fn
        self._gripper_state = gripper_state
        self._min_norm = min_norm
        self._gain_pos = gain_pos
        self._gain_yaw = gain_yaw
        self._termination_threshold = termination_threshold
        
    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm
            
    def can_initiate(self, ob, info):
        """Can always initiate this option."""
        return True
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        """Select action to lift vertically."""
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        
        base_pos = self._base_pos_fn(ob, info)
        target_pos = np.array([base_pos[0], base_pos[1], self._target_height])
        
        # Compute position difference (only z should change significantly)
        diff = target_pos - effector_pos
        diff = self.shape_diff(diff)
        
        # Compute yaw difference
        if self._target_yaw_fn is not None:
            target_yaw = self._target_yaw_fn(ob, info)
            yaw_diff = target_yaw - effector_yaw
        else:
            yaw_diff = 0.0
        
        # Construct action
        action = np.zeros(5)
        action[:3] = diff[:3] * self._gain_pos
        action[3] = yaw_diff * self._gain_yaw
        action[4] = self._gripper_state
        
        return np.clip(action, -1, 1)
        
    def is_terminated(self, ob, info):
        """Terminate when at target height."""
        effector_pos = info['proprio/effector_pos']
        base_pos = self._base_pos_fn(ob, info)
        target_pos = np.array([base_pos[0], base_pos[1], self._target_height])
        distance = np.linalg.norm(target_pos - effector_pos)
        return distance <= self._termination_threshold


class GraspOption(Option):
    """Option to grasp an object at the current position.
    
    This option moves to the object, aligns, and closes the gripper.
    """
    
    def __init__(self, name, env, block_pos_fn, block_yaw_fn, min_norm=0.4,
                 gain_pos=5, gain_yaw=3, alignment_threshold=0.02):
        """Initialize the grasp option.
        
        Args:
            name: Option name
            env: Environment instance
            block_pos_fn: Function (ob, info) -> np.ndarray that returns block position
            block_yaw_fn: Function (ob, info) -> float that returns block yaw
            min_norm: Minimum norm for position differences
            gain_pos: Position gain
            gain_yaw: Yaw gain
            alignment_threshold: Position alignment threshold
        """
        super().__init__(name, env)
        self._block_pos_fn = block_pos_fn
        self._block_yaw_fn = block_yaw_fn
        self._min_norm = min_norm
        self._gain_pos = gain_pos
        self._gain_yaw = gain_yaw
        self._alignment_threshold = alignment_threshold
        self._phase = 'move'  # 'move' or 'grasp'
        
    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        else:
            return diff / (diff_norm + 1e-6) * self._min_norm
            
    def can_initiate(self, ob, info):
        """Can initiate if not already grasping."""
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        return not gripper_closed
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        self._phase = 'move'
        
    def select_action(self, ob, info):
        """Select action for grasping."""
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        
        block_pos = self._block_pos_fn(ob, info)
        block_yaw = self._block_yaw_fn(ob, info)
        
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= self._alignment_threshold
        
        action = np.zeros(5)
        
        if not pos_aligned or not gripper_closed:
            # Move to block and align
            diff = block_pos - effector_pos
            diff = self.shape_diff(diff)
            action[:3] = diff[:3] * self._gain_pos
            action[3] = (block_yaw - effector_yaw) * self._gain_yaw
            action[4] = 1 if pos_aligned else -1  # Close gripper if aligned
        else:
            # Already grasping
            action[4] = 1
            
        return np.clip(action, -1, 1)
        
    def is_terminated(self, ob, info):
        """Terminate when gripper is closed and aligned."""
        effector_pos = info['proprio/effector_pos']
        block_pos = self._block_pos_fn(ob, info)
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        
        pos_aligned = np.linalg.norm(block_pos - effector_pos) <= self._alignment_threshold
        return pos_aligned and gripper_closed


class ReleaseOption(Option):
    """Option to release the gripper at the current position."""
    
    def __init__(self, name, env):
        """Initialize the release option.
        
        Args:
            name: Option name
            env: Environment instance
        """
        super().__init__(name, env)
        
    def can_initiate(self, ob, info):
        """Can initiate if gripper is closed."""
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        return gripper_closed
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        """Select action to open gripper."""
        action = np.zeros(5)
        action[4] = -1  # Open gripper
        return action
        
    def is_terminated(self, ob, info):
        """Terminate when gripper is open."""
        gripper_open = info['proprio/gripper_contact'] < 0.1
        return gripper_open

