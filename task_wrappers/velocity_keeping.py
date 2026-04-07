from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from brax.envs.base import State
from brax.envs.base import Env, PipelineEnv
from task_wrappers.base import BaseTaskWrapper, BaseQDTaskWrapper
from data_struct.states import GeneralizedState
# from data_struct.transitions import TransitionInfo
from data_struct.qd_transitions import QDTransitionInfo


class TaskState(PyTreeNode):
    z: jnp.ndarray 



class AntVelocityWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        v_max=3.0):
        super().__init__(env)
        self.v_max = v_max


    @property
    def z_size(self):
        return 2

    @property
    def has_z(self):
        return True
    

    def _init_task_state(self, env_state: State, key: jax.Array) -> PyTreeNode:
        """initialize task state"""
  
        # find direction
        subkey1, subkey2 = jax.random.split(key) # (2,)
        velocity = jax.random.normal(subkey1, shape=(2,))
        velocity = jnp.where(velocity > 0, velocity + 1e-3, velocity - 1e-3)
        unit_v = velocity / jnp.sqrt(jnp.sum(velocity**2))

        # sample speed
        speed = jnp.sqrt(jax.random.uniform(subkey2) + 1e-8) * self.v_max
        velocity = unit_v * speed

        new_task_state = TaskState(
            z=velocity,
        )
        
        return new_task_state
    

    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, Tuple[jax.Array, ...]]:
        """extract observations and z (will be empty tuple if has_z == False)"""
        return state.env_state.obs, state.z_state.z
    

    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,    
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """return next state, reward, done, truncation"""
        
        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation

        l2_velocity_diff = jnp.sqrt(
            jnp.sum(jnp.square(state.z_state.z - next_env_state.obs[13:15]))
            )
        reward = 2 / (jnp.exp(-l2_velocity_diff) + jnp.exp(l2_velocity_diff))
        # fitness_reward = jnp.array([next_env_state.reward - next_env_state.metrics["x_velocity"] + 3.0])
        fitness_reward = 4 - jnp.sum(jnp.square(action))

        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]), 
            fitness_reward=jnp.array([fitness_reward]),
            done=jnp.array([done]),
            truncation=jnp.array([truncation]))

        new_task_state = jax.lax.cond(
            next_env_state.done > 0.9,
            lambda _: state.initial_z_state,
            lambda _: state.z_state,
            None,
            )

        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info




