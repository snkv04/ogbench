import numpy as np


class Option:
    """Base class for hierarchical RL options (multi-timestep actions).
    
    An option consists of:
    - An initiation set: states where the option can be started
    - An intra-option policy: how to select actions while the option is executing
    - A termination condition: when the option should terminate
    """
    
    def __init__(self, name, env):
        """Initialize the option.
        
        Args:
            name: Name of the option (for debugging/logging)
            env: Environment instance
        """
        self.name = name
        self._env = env
        self._active = False
        self._step = 0
        
    def can_initiate(self, ob, info):
        """Check if the current state is in the initiation set for this option.
        
        Args:
            ob: Current observation
            info: Current info dictionary
            
        Returns:
            bool: True if option can be initiated
        """
        raise NotImplementedError
        
    def initiate(self, ob, info):
        """Initialize the option when it's first selected.
        
        Args:
            ob: Current observation
            info: Current info dictionary
        """
        self._active = True
        self._step = 0
        
    def select_action(self, ob, info):
        """Select the next action while the option is executing.
        
        This is the intra-option policy.
        
        Args:
            ob: Current observation
            info: Current info dictionary
            
        Returns:
            np.ndarray: Action to execute
        """
        raise NotImplementedError
        
    def is_terminated(self, ob, info):
        """Check if the current state is in the termination set for this option.
        
        Args:
            ob: Current observation
            info: Current info dictionary
            
        Returns:
            bool: True if option should terminate
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset the option state."""
        self._active = False
        self._step = 0
        
    @property
    def active(self):
        """Whether the option is currently active."""
        return self._active
    
    @property
    def step_count(self):
        """Get the current step count."""
        return self._step
        
    def step(self):
        """Increment internal step counter."""
        self._step += 1

