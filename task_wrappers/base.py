import abc
from typing import Tuple
import jax
from brax.envs.base import Env, Wrapper
from data_struct.states import GeneralizedState
from data_struct.transitions import TransitionInfo
from data_struct.qd_transitions import QDTransitionInfo
from brax.envs.base import State
from flax.struct import PyTreeNode



class BaseTaskWrapper(Wrapper, abc.ABC):

    @property
    @abc.abstractmethod
    def z_size(self) -> int:
        pass
    
    @property
    @abc.abstractmethod
    def has_z(self) -> bool:
        pass

    
    @abc.abstractmethod
    def _extract_state_info_for_task(self, env_state: State) -> PyTreeNode:
        """extract state relavent information for task state"""
        pass


    @abc.abstractmethod
    def _init_task_state(self, state_info: PyTreeNode, key: jax.Array) -> PyTreeNode:
        """initialize task state"""
        pass


    @abc.abstractmethod
    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, Tuple[jax.Array, ...]]:
        """extract observations and z (will be empty if has_z == False)"""
        pass


    def reset(self, key: jax.Array) -> GeneralizedState:
        env_key, task_key, key = jax.random.split(key, num=3)
        env_state = self.env.reset(env_key)
        initial_state_info = self._extract_state_info_for_task(env_state)
        z_state = self._init_task_state(initial_state_info, task_key)
        state = GeneralizedState(env_state=env_state, z_state=z_state, initial_z_state=z_state, initial_state_info=initial_state_info, key=key)
        return state
    

    def resample_task_state(self, state: GeneralizedState) -> GeneralizedState:
        """resample task state"""
        key, subkey = jax.random.split(state.key)
        state_info = self._extract_state_info_for_task(state.env_state)
        z_state = self._init_task_state(state_info, subkey)
        state = state.replace(z_state=z_state, key=key)
        return state
    

    def resample_initial_task_state(self, state: GeneralizedState) -> GeneralizedState:
        """resample initial task state"""
        key, subkey = jax.random.split(state.key)
        initial_z_state = self._init_task_state(state.initial_state_info, subkey)
        state = state.replace(initial_z_state=initial_z_state, key=key)
        return state


    @abc.abstractmethod
    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,    
    ) -> Tuple[GeneralizedState, TransitionInfo]:
        """return next state, and transition information"""
        pass



class BaseQDTaskWrapper(BaseTaskWrapper, abc.ABC):

    @abc.abstractmethod
    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,    
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """return next state, and transition information"""
        pass
