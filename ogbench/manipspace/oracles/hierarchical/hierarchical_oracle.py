import logging
import numpy as np
from typing import Union, List, Optional

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle
from ogbench.manipspace.oracles.hierarchical.option import Option

logger = logging.getLogger(__name__)


class HierarchicalOracle(MarkovOracle):
    """Hierarchical oracle that can select either primitive actions or options.
    
    The agent can choose between:
    - Primitive actions: Single-timestep actions (normal actions)
    - Options: Multi-timestep actions that run until termination
    
    When an option is selected, it executes its intra-option policy until
    it terminates, then control returns to the high-level policy.
    """
    
    def __init__(self, options: Optional[List] = None, *args, **kwargs):
        """Initialize the hierarchical oracle.
        
        Args:
            options: List of Option instances that can be selected
            *args, **kwargs: Arguments passed to MarkovOracle
        """
        super().__init__(*args, **kwargs)
        self._options = options or []
        self._active_option = None
        self._primitive_action_space_size = 5  # Default action space size
        self._option_step_counts = []  # Track step counts for terminated options
        
    def add_option(self, option):
        """Add an option to the available options.
        
        Args:
            option: Option instance to add
        """
        self._options.append(option)
        
    def select_high_level_action(self, ob, info):
        """Select either a primitive action or an option.
        
        This is the high-level policy. Override this method to implement
        your selection logic (e.g., using a learned policy, rules, etc.).
        
        Args:
            ob: Current observation
            info: Current info dictionary
            
        Returns:
            Union[np.ndarray, Option]: Either a primitive action array or an Option instance
        """
        raise NotImplementedError
        
    def select_action(self, ob, info):
        """Select action (handles both primitive actions and options).
        
        If an option is active, it continues executing. Otherwise, the high-level
        policy selects a new action (primitive or option).
        
        Args:
            ob: Current observation
            info: Current info dictionary
            
        Returns:
            np.ndarray: Action to execute
        """
        # If an option is active, continue executing it
        if self._active_option is not None and self._active_option.active:
            action = self._active_option.select_action(ob, info)
            self._active_option.step()
            
            # Check if option should terminate
            if self._active_option.is_terminated(ob, info):
                # self._record_terminated_option()
                self._active_option.reset()  # Deactivates the option but keeps the reference
                
            return action
        
        # Otherwise, select a new high-level action
        high_level_action = self.select_high_level_action(ob, info)
        
        # If it's an Option, initiate it
        if isinstance(high_level_action, Option):
            self._active_option = high_level_action
            # logging.info(f"Selected new option: {self._active_option.name}")
            self._active_option.initiate(ob, info)
            action = self._active_option.select_action(ob, info)
            self._active_option.step()
            
            # Check termination immediately (in case option terminates in one step)
            if self._active_option.is_terminated(ob, info):
                # self._record_terminated_option()
                self._active_option.reset()  # Deactivates the option but keeps the reference
                
            return action
        
        # Otherwise, it's a primitive action
        return high_level_action
        
    def reset(self, ob, info):
        """Reset the oracle and all options.
        
        Args:
            ob: Initial observation
            info: Initial info dictionary
        """
        super().reset(ob, info)
        if self._active_option is not None:
            self._active_option.reset()
            # Set _active_option to None only on full reset of agent to clear the reference
            self._active_option = None
        for option in self._options:
            option.reset()
    
    def _record_terminated_option(self):
        """Record information about a terminated option."""
        step_count = self._active_option.step_count
        self._option_step_counts.append(step_count)
        avg_steps = np.mean(self._option_step_counts)
        logger.info(f"Average option step count: {avg_steps:.2f} (based on {len(self._option_step_counts)} terminated options)")
            
    @property
    def active_option(self):
        """Get the currently active option (if any)."""
        return self._active_option

