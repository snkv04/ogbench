# Hierarchical RL with Options

This module implements a hierarchical RL framework where agents can select either **primitive actions** (single-timestep) or **options** (multi-timestep actions with intra-option policies).

## Overview

The hierarchical framework consists of:

1. **Option**: Base class for multi-timestep actions
2. **HierarchicalOracle**: Base class that can select options or primitive actions
3. **Concrete Options**: Task-specific options (e.g., `MoveToPositionOption`, `GraspOption`)
4. **Concrete Hierarchical Oracles**: Task-specific high-level policies (e.g., `CubeHierarchicalOracle`)

## Key Concepts

### Options
An option is a multi-timestep action that:
- Has an **initiation set**: States where it can be started
- Has an **intra-option policy**: How to select actions while executing
- Has a **termination condition**: When to stop executing

### Execution Flow

```
Main Loop:
  while not done:
    action = agent.select_action(ob, info)
    next_ob, reward, done, info = env.step(action)
    
HierarchicalOracle.select_action():
  if active_option exists:
    # Continue executing current option
    action = active_option.select_action(ob, info)
    if active_option.is_terminated(ob, info):
      active_option.reset()
      active_option = None
  else:
    # Select new high-level action
    high_level_action = select_high_level_action(ob, info)
    if high_level_action is Option:
      active_option = high_level_action
      active_option.initiate(ob, info)
      action = active_option.select_action(ob, info)
    else:
      # Primitive action
      action = high_level_action
```

## Usage Example

### 1. Define Options

```python
from ogbench.manipspace.oracles.hierarchical.option import Option

class MyOption(Option):
    def can_initiate(self, ob, info):
        # Return True if option can start in this state
        return True
        
    def initiate(self, ob, info):
        # Initialize option state
        super().initiate(ob, info)
        
    def select_action(self, ob, info):
        # Intra-option policy: select action while option is active
        return np.array([0.1, 0.2, 0.3, 0.0, -1])
        
    def is_terminated(self, ob, info):
        # Return True when option should terminate
        return some_condition
```

### 2. Create Hierarchical Oracle

```python
from ogbench.manipspace.oracles.hierarchical.hierarchical_oracle import HierarchicalOracle

class MyHierarchicalOracle(HierarchicalOracle):
    def select_high_level_action(self, ob, info):
        # High-level policy: select option or primitive action
        if should_use_option:
            return my_option
        else:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Primitive action
```

### 3. Use in Execution Loop

```python
agent = MyHierarchicalOracle(env=env)
agent.reset(ob, info)

while not done:
    # The oracle handles option execution automatically!
    action = agent.select_action(ob, info)
    next_ob, reward, done, info = env.step(action)
    ob = next_ob
```

## Benefits

1. **Abstraction**: Options encapsulate common behaviors (e.g., "grasp object", "move to position")
2. **Reusability**: Options can be reused across different tasks
3. **Learning Efficiency**: High-level policy learns to select options, not low-level actions
4. **Temporal Abstraction**: Options naturally handle multi-timestep behaviors

## Extending to Learned Policies

To use with learned hierarchical RL:

1. **High-level policy**: Replace `select_high_level_action()` with a learned policy network
2. **Intra-option policies**: Can be learned or use existing oracles (like `MarkovOracle`)
3. **Option discovery**: Learn which options are useful for the task

Example with learned high-level policy:

```python
class LearnedHierarchicalOracle(HierarchicalOracle):
    def __init__(self, high_level_policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._high_level_policy = high_level_policy
        
    def select_high_level_action(self, ob, info):
        # Learned policy outputs option index or primitive action
        action_or_option_idx = self._high_level_policy(ob, info)
        
        if isinstance(action_or_option_idx, int):
            # Option index
            return self._options[action_or_option_idx]
        else:
            # Primitive action
            return action_or_option_idx
```

## Files

- `option.py`: Base Option class
- `hierarchical_oracle.py`: Base HierarchicalOracle class
- `cube_options.py`: Example options for cube manipulation
- `cube_hierarchical.py`: Example hierarchical oracle for cube task
- `generate_manipspace_hierarchical.py`: Example execution script

