import random

from ogbench.manipspace.oracles.hierarchical.cube_hierarchical import CubeHierarchicalOracle


class CubeHierarchicalRandom(CubeHierarchicalOracle):
    """Hierarchical agent that selects random high-level options.
    
    Inherits from CubeHierarchicalOracle but overrides the high-level policy
    to select options uniformly at random instead of using the rule-based oracle.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the random hierarchical agent."""
        # Remove oracle-specific kwargs if present
        kwargs.pop('no_op_option_prob', None)
        kwargs.pop('suboptimal_option_prob', None)
        super().__init__(no_op_option_prob=0.0, suboptimal_option_prob=0.0, *args, **kwargs)
        
    def select_high_level_action(self, ob, info):
        """Select a random high-level action (option).
        
        Uniformly selects from all available options.
        """
        return random.choice(self._options)
