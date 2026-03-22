import numpy as np
from loguru import logger as logging
from ogbench.manipspace.oracles.hierarchical.option import Option
from ogbench.manipspace.oracles.markov.markov_agent import MarkovAgent


class MoveToPositionOption(Option):
    """Option to move the gripper to a target position.
    
    This option encapsulates the "move to position" behavior, which
    can be reused for moving to the block, target, or final position.
    """
    
    def __init__(
        self,
        name,
        env,
        target_pos_fn,
        target_yaw_fn=None, 
        gripper_state=-1,
        min_norm=0.4,
        gain_pos=5,
        gain_yaw=3,
        reward_type='sparse',

        # Position portion of reward
        use_position: bool = False,
        termination_threshold=0.04,
        position_reward_weight=0.33,

        # Yaw portion of reward
        use_yaw: bool = False,
        yaw_threshold: float = 0.5,
        yaw_reward_weight=0.33,

        # Gripper portion of reward
        use_gripper: bool = False,
        opening_threshold: float = 0.1,
        closing_threshold: float = 0.55,
        gripper_reward_weight=0.33,
    ):
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
            use_position: Whether to use position alignment in reward
            use_yaw: Whether to use yaw alignment in reward
            yaw_threshold: Yaw alignment threshold in radians
            use_gripper: Whether to use gripper state in reward
            opening_threshold: Threshold below which gripper is considered open
            closing_threshold: Threshold above which gripper is considered closed
        """
        super().__init__(name, env)
        assert reward_type in ['sparse', 'sparse_stepwise', 'dense'], f"Invalid reward type: {reward_type}"
        self._reward_type = reward_type
        self._target_pos_fn = target_pos_fn
        self._target_yaw_fn = target_yaw_fn
        self._gripper_state = gripper_state
        self._min_norm = min_norm
        self._gain_pos = gain_pos
        self._gain_yaw = gain_yaw

        self._use_position = use_position
        self._termination_threshold = termination_threshold
        self._position_reward_weight = position_reward_weight

        self._use_yaw = use_yaw
        self._yaw_threshold = yaw_threshold
        self._yaw_reward_weight = yaw_reward_weight

        self._use_gripper = use_gripper
        self._opening_threshold = opening_threshold
        self._closing_threshold = closing_threshold
        self._gripper_reward_weight = gripper_reward_weight

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

    def _calculate_sparse_position_reward(self, next_ob, next_info):
        effector_pos = next_info['proprio/effector_pos']
        target_pos = self._target_pos_fn(next_ob, next_info)
        distance = np.linalg.norm(target_pos - effector_pos)
        return 0.0 if distance <= self._termination_threshold else -1.0

    def _calculate_sparse_yaw_reward(self, next_ob, next_info):
        effector_yaw = next_info['proprio/effector_yaw'][0]
        target_yaw = self._target_yaw_fn(next_ob, next_info)
        # Shortest angular difference in [-pi, pi]
        yaw_diff = (target_yaw - effector_yaw + np.pi) % (2 * np.pi) - np.pi
        return 0.0 if np.abs(yaw_diff) <= self._yaw_threshold else -1.0

    def _calculate_sparse_gripper_reward(self, next_ob, next_info):
        gripper_opening = next_info['proprio/gripper_opening'][0]
        if self._gripper_state == 1:
            return 0.0 if gripper_opening >= self._closing_threshold else -1.0
        elif self._gripper_state == -1:
            return 0.0 if gripper_opening <= self._opening_threshold else -1.0
        else:
            raise NotImplementedError(f"Gripper state {self._gripper_state} not implemented")

    def _calculate_dense_position_reward(self, next_ob, next_info):
        # Same reward logic as in https://arxiv.org/pdf/2206.11403
        effector_pos = next_info['proprio/effector_pos']
        target_pos = self._target_pos_fn(next_ob, next_info)
        distance = np.linalg.norm(target_pos - effector_pos)
        return -1.0 * max(self._termination_threshold, distance)

    def _calculate_dense_yaw_reward(self, next_ob, next_info):
        # Same reward logic as in https://arxiv.org/pdf/2206.11403
        effector_yaw = next_info['proprio/effector_yaw'][0]
        target_yaw = self._target_yaw_fn(next_ob, next_info)
        # Shortest angular difference in [-pi, pi]
        yaw_diff = (target_yaw - effector_yaw + np.pi) % (2 * np.pi) - np.pi
        return -1.0 * max(self._yaw_threshold, np.abs(yaw_diff))

    def _calculate_dense_gripper_reward(self, next_ob, next_info):
        gripper_opening = next_info['proprio/gripper_opening'][0]
        if self._gripper_state == 1:
            return -1.0 * (1 - gripper_opening)  # Rewarded for being high
        elif self._gripper_state == -1:
            return -1.0 * gripper_opening  # Rewarded for being low
        else:
            raise NotImplementedError(f"Gripper state {self._gripper_state} not implemented")

    def calculate_reward(self, next_ob, next_info):
        if self._reward_type.startswith('sparse'):
            position_reward = self._calculate_sparse_position_reward(next_ob, next_info)
            yaw_reward = self._calculate_sparse_yaw_reward(next_ob, next_info)
            gripper_reward = self._calculate_sparse_gripper_reward(next_ob, next_info)

            if self._reward_type == 'sparse':
                # Needs to pass all criteria to get 0 reward
                reward = 0.0
                if self._use_position:
                    reward = min(reward, position_reward)
                if self._use_yaw:
                    reward = min(reward, yaw_reward)
                if self._use_gripper:
                    reward = min(reward, gripper_reward)
                return reward

            elif self._reward_type == 'sparse_stepwise':
                return (
                    self._position_reward_weight * position_reward +
                    self._yaw_reward_weight * yaw_reward +
                    self._gripper_reward_weight * gripper_reward
                )

            else:
                raise NotImplementedError(f"Invalid reward type: {self._reward_type}")

        elif self._reward_type == 'dense':
            # logging.info(f"Calculating dense reward")

            position_reward = self._calculate_dense_position_reward(next_ob, next_info)
            yaw_reward = self._calculate_dense_yaw_reward(next_ob, next_info)
            gripper_reward = self._calculate_dense_gripper_reward(next_ob, next_info)

            return (
                self._position_reward_weight * position_reward +
                self._yaw_reward_weight * yaw_reward +
                self._gripper_reward_weight * gripper_reward
            )
        
        else:
            raise NotImplementedError(f"Invalid reward type: {self._reward_type}")
        

class LiftVerticallyOption(MoveToPositionOption):
    """Option to lift the gripper vertically while keeping x,y fixed.

    This is used for phases 4 and 8 - lifting straight up for clearance.
    Subclass of MoveToPositionOption with target position (base_x, base_y, target_height).
    """

    # TODO: Add more args to match MoveToPositionOption
    def __init__(self, name, env, base_pos_fn, target_height, target_yaw_fn=None,
                 gripper_state=1, min_norm=0.4, gain_pos=5, gain_yaw=3,
                 termination_threshold=0.04, reward_type='sparse'):
        """Initialize the lift vertically option.

        Args:
            name: Option name
            env: Environment instance
            base_pos_fn: Function (ob, info) -> np.ndarray that returns base (x,y,z) position
            target_height: Target z height to lift to
            target_yaw_fn: Optional function (ob, info) -> float that returns target yaw
            gripper_state: Gripper state to maintain (-1=open, 1=closed)
            min_norm: Minimum norm for position differences
            gain_pos: Position gain
            gain_yaw: Yaw gain
            termination_threshold: Distance threshold for termination
            reward_type: Reward type
        """
        def target_pos_fn(ob, info):
            base = base_pos_fn(ob, info)
            return np.array([base[0], base[1], target_height])

        super().__init__(
            name, env, target_pos_fn, target_yaw_fn=target_yaw_fn,
            gripper_state=gripper_state, min_norm=min_norm,
            gain_pos=gain_pos, gain_yaw=gain_yaw,
            termination_threshold=termination_threshold,
            reward_type=reward_type,
        )


class GraspOption(Option):
    """Option to close the gripper (grasp) without moving the arm.
    
    This is effectively the opposite of ReleaseOption: it only changes the
    gripper state and terminates once the gripper is sufficiently closed.
    """
    
    def __init__(self, name, env, closing_threshold: float = 0.55):
        """Initialize the grasp option.
        
        Args:
            name: Option name
            env: Environment instance
            closing_threshold: Threshold on proprio/gripper_opening above which
                the option is considered to have successfully grasped.
        """
        super().__init__(name, env)
        self._closing_threshold = closing_threshold
        
    def can_initiate(self, ob, info):
        """Can initiate if not already grasping (based on contact)."""
        gripper_closed = info['proprio/gripper_contact'] > 0.5
        return not gripper_closed
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        """Select action to close the gripper only."""
        action = np.zeros(5)
        action[4] = 1  # Close gripper
        return action
        
    def is_terminated(self, ob, info):
        """Terminate when the gripper is sufficiently closed."""
        gripper_opening = info['proprio/gripper_opening'][0]
        gripper_closed = gripper_opening >= self._closing_threshold
        return gripper_closed

    # TODO: Implement dense reward
    def calculate_reward(self, next_ob, next_info):
        """Return 0 if grasp is complete (terminated), -1 otherwise."""
        return 0.0 if self.is_terminated(next_ob, next_info) else -1.0


class ReleaseOption(Option):
    """Option to release the gripper at the current position."""
    
    def __init__(self, name, env, opening_threshold: float = 0.1):
        """Initialize the release option.
        
        Args:
            name: Option name
            env: Environment instance
            opening_threshold: Threshold on proprio/gripper_opening below which
                the option is considered to have successfully released.
        """
        super().__init__(name, env)
        self._opening_threshold = opening_threshold

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
        gripper_opening = info['proprio/gripper_opening'][0]
        gripper_open = gripper_opening <= self._opening_threshold
        return gripper_open

    # TODO: Implement dense reward
    def calculate_reward(self, next_ob, next_info):
        """Return 0 if release is complete (terminated), -1 otherwise."""
        return 0.0 if self.is_terminated(next_ob, next_info) else -1.0


class NoOpOption(Option):
    """Option that performs no operation (zero actions) for a fixed number of timesteps."""
    
    def __init__(self, name, env, duration=10):
        """Initialize the no-op option.
        
        Args:
            name: Option name
            env: Environment instance
            duration: Number of timesteps the option should last
        """
        super().__init__(name, env)
        self._duration = duration
        
    def can_initiate(self, ob, info):
        """Can always initiate this option."""
        return True
        
    def initiate(self, ob, info):
        """Initialize the option."""
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        """Select zero action (no-op).
        
        Returns a zero action, which in the manipspace environment's relative
        control scheme means "no change" - the effector will maintain its
        current position, orientation, and gripper state.
        """
        action = np.zeros(5)
        return action
        
    def is_terminated(self, ob, info):
        """Terminate after the specified duration."""
        return self._step >= self._duration

