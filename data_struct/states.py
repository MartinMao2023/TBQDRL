from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
from flax import struct
from brax.envs.base import State as BraxState


class TaskState(struct.PyTreeNode):
    z: jax.Array


@struct.dataclass
class GeneralizedState:
    env_state: BraxState
    z_state: TaskState
    initial_z_state: TaskState # used in reset
    key: jax.Array
    


