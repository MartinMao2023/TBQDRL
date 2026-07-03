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
from data_struct.transitions import TransitionInfo, MOTransitionInfo
from custom_types import Params, RNGKey, Env, EnvState
from .tools import IntegrateMatern



class TaskState(PyTreeNode):
    last_action: jax.Array
    preference: jax.Array
    z: jax.Array # last action + preference



class State_info(PyTreeNode):
    dummy: jax.Array # (0,)



class GeneralizedState(PyTreeNode):
    env_state: State
    z_state: TaskState
    initial_state_info: State_info # used to resample initial_z_state
    initial_z_state: TaskState # used in reset
    key: jax.Array



class AntMOWrapper(BaseTaskWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.z_dim = env.action_size + 5
        self._action_dim = env.action_size
        self._preference_dim = 5


    @property
    def has_z(self):
        return True
    

    @property
    def z_size(self):
        return self.z_dim
    

    def _extract_state_info_for_task(self, env_state):
        return State_info(dummy=jnp.zeros(0))
    

    def _init_task_state(self, state_info, key):
        preference = jax.random.normal(key, shape=(self._preference_dim,))
        preference = jnp.where(jnp.array([False, False, True, True, True]), jnp.abs(preference), preference)
        preference = preference / jnp.sqrt(jnp.sum(preference**2) + 1e-6)
        initial_action = jnp.zeros((self._action_dim,))

        return TaskState(
            last_action=initial_action,
            preference=preference,
            z=jnp.concatenate([initial_action, preference])
            )
    
    
    def resample_task_state(self, state: GeneralizedState) -> GeneralizedState:
        """resample task state"""
        key, subkey = jax.random.split(state.key)
        state_info = self._extract_state_info_for_task(state.env_state)
        last_action = state.z_state.last_action
        new_preference = self._init_task_state(state_info, subkey).preference
        
        new_z_state = state.z_state.replace(
            preference=new_preference,
            z=jnp.concatenate([last_action, new_preference])
        )
        state = state.replace(z_state=new_z_state, key=key)
        return state
    

    def get_obs(self, state):
        return state.env_state.obs, state.z_state.z
    

    def step(self, state: GeneralizedState, action: jax.Array):
        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation
        last_action = state.z_state.last_action
        preference = state.z_state.preference
    
        vx = state.env_state.metrics["x_velocity"]
        vy = state.env_state.metrics["y_velocity"]
        height = state.env_state.pipeline_state.x.pos[0, 2]
        control_penalty = jnp.mean(jnp.square(action))
        consistency_penalty = jnp.mean(jnp.square(action - last_action))

        mo_reward = jnp.array([vx, vy, height, 1.0 - control_penalty, 1.0 - consistency_penalty])
        reward = jnp.sum(mo_reward * preference, keepdims=True)

        transition_info = MOTransitionInfo(
            reward=reward, 
            mo_reward=mo_reward,
            done=jnp.where(done > 0.5, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]),
            broken=jnp.array([0.0]))
        
        new_task_state = state.z_state.replace(last_action=action, z=jnp.concatenate([action, preference]))
        
        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info
    

    






    

