import abc
from typing import Tuple
import jax
from brax.envs.base import Env, Wrapper
from data_struct.states import GeneralizedState, GeneralizedQDState
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
    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, jax.Array]:
        """extract observations and z (will be empty if has_z == False)"""
        pass


    def reset(self, key: jax.Array) -> GeneralizedState:
        env_key, task_key, key = jax.random.split(key, num=3)
        env_state = self.env.reset(env_key)
        initial_state_info = self._extract_state_info_for_task(env_state)
        z_state = self._init_task_state(initial_state_info, task_key)
        state = GeneralizedState(
            env_state=env_state, 
            z_state=z_state, 
            initial_z_state=z_state, 
            initial_state_info=initial_state_info, 
            key=key,
            )
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



class BaseQDTaskWrapper(Wrapper, abc.ABC):

    def __init__(self, env: Env, way_points: int, steps_per_way_point: int, dt: float):
        super().__init__(env)
        self.has_z = True
        self.way_points = way_points
        self.steps_per_way_point = steps_per_way_point
        self.dt = dt

        self.max_step_num = int(way_points * steps_per_way_point)
        self.horizon = way_points * steps_per_way_point * dt
        self.period_t = steps_per_way_point * dt


    @property
    @abc.abstractmethod
    def z_size(self) -> int:
        pass
    

    @abc.abstractmethod
    def get_obs(self, state: GeneralizedQDState) -> Tuple[jax.Array, jax.Array]:
        """extract observations and z"""
        pass


    @abc.abstractmethod
    def sample_task(self, env_state: State, key: jax.Array, **kwargs) -> PyTreeNode:
        """initialize task state"""
        pass


    def reset(self, key: jax.Array, **kwargs) -> GeneralizedQDState:
        env_key, task_key, key = jax.random.split(key, num=3)
        env_state = self.env.reset(env_key)
        z_state = self.sample_task(env_state, task_key, **kwargs)
        
        state = GeneralizedQDState(
            env_state=env_state, 
            z_state=z_state, 
            initial_z_state=z_state, 
            key=key,
            )
        return state
    

    def resample_task_state(self, state: GeneralizedQDState, **kwargs) -> GeneralizedQDState:
        """resample task state"""
        key, subkey = jax.random.split(state.key)
        z_state = self.sample_task(state.env_state, subkey, **kwargs)
        state = state.replace(z_state=z_state, key=key)
        return state


    @abc.abstractmethod
    def step(
        self, 
        state: GeneralizedQDState, 
        action: jax.Array,
    ) -> Tuple[GeneralizedQDState, QDTransitionInfo]:
        """return next state, and transition information"""
        pass


    @abc.abstractmethod
    def shift(self, state: GeneralizedQDState) -> GeneralizedQDState:
        """
        To shift:
            1) take cycle_t - self.period_t
            2) shift and pad reshaped_sequence
        """
        pass
