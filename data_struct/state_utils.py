from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
from flax import struct
from brax.envs.base import State as BraxState
from brax.envs.base import Env, PipelineEnv, Wrapper


from custom_types import (
    Action,
    Descriptor,
    Done,
    Observation,
    Reward,
    RNGKey,
    StateDescriptor,
)


@struct.dataclass
class CustomState:
    env_state: BraxState
    goal_state: struct.PyTreeNode
    extra_info: struct.PyTreeNode





