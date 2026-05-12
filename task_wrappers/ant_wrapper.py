from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.linalg as lg
from flax.struct import PyTreeNode
from brax.envs.base import State
from brax.envs.base import Env, PipelineEnv
from task_wrappers.base import BaseTaskWrapper, BaseQDTaskWrapper
# from data_struct.states import GeneralizedState
# from data_struct.transitions import TransitionInfo
from data_struct.transitions import TransitionInfo
from custom_types import Params, RNGKey, Env, EnvState
from .tools import IntegrateMatern



class TaskState(PyTreeNode):
    z: jax.Array # last action



class State_info(PyTreeNode):
    dummy: jax.Array # (0,)



class GeneralizedState(PyTreeNode):
    env_state: State
    z_state: TaskState
    initial_state_info: State_info # used to resample initial_z_state
    initial_z_state: TaskState # used in reset
    key: jax.Array



class AntWrapper(BaseTaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.z_dim = env.action_size


    @property
    def has_z(self):
        return True
    

    @property
    def z_size(self):
        return self.z_dim
    

    def _extract_state_info_for_task(self, env_state):
        return State_info(dummy=jnp.zeros(0))
    

    def _init_task_state(self, state_info, key):
        return TaskState(z=jnp.zeros((self.z_dim,)))
    

    def get_obs(self, state):
        return state.env_state.obs, state.z_state.z
    

    def step(self, state, action):
        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation


        consistency_penalty = jnp.mean(jnp.square(action - state.z_state.z))

        transition_info = TransitionInfo(
            reward=jnp.array([next_env_state.reward - consistency_penalty]), 
            done=jnp.where(done > 0.5, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]))
        
        new_task_state = TaskState(z=action)
        
        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info
    

    






    

