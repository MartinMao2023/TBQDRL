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
from data_struct.states import GeneralizedState
# from data_struct.transitions import TransitionInfo
from data_struct.qd_transitions import QDTransitionInfo
from custom_types import Params, RNGKey, Env, EnvState
from .tools import IntegrateMatern


class TaskState(PyTreeNode):
    deviation: jnp.ndarray # shape of (2,)
    task: jnp.ndarray # shape of (2, d)
    t: float
    z: jnp.ndarray # shape of (2 + 2 * d + 1,)
    r: float



class LineWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        v_min=1.0,
        v_max=2.5,
        dt=0.05,):
        super().__init__(env)
        self.v_min = v_min
        self.v_max = v_max
        self.dt = dt


    @property
    def z_size(self):
        return 4

    @property
    def has_z(self):
        return True
    

    def get_velocity_from_envstate(self, env_state: State) -> jax.Array:
        """This is for Ant, change this line if for other agent"""
        obs = env_state.obs
        current_velocity = obs[13:15] # (2,)

        return current_velocity 
    

    def get_position_from_envstate(self, env_state: State) -> jax.Array:
        """Change this line if doesn't work for other agent"""

        return env_state.pipeline_state.x.pos[0, :2] # (2,)
    

    def _init_task_state(self, env_state: State, key: jax.Array) -> PyTreeNode:
        """initialize task state"""
        current_velocity = self.get_velocity_from_envstate(env_state)
        current_speed = jnp.sqrt(jnp.sum(current_velocity**2))

        # find direction
        key, subkey = jax.random.split(key) # (2,)
        perturbed_velocity = current_velocity + jax.random.normal(subkey, shape=(2,)) * 0.25 # (2,)
        velocity = jnp.where(perturbed_velocity > 0, perturbed_velocity + 1e-2, perturbed_velocity - 1e-2)
        unit_v = velocity / jnp.sqrt(jnp.sum(velocity**2))

        # sample speed
        key, subkey = jax.random.split(key) # (2,)
        v_min = jnp.clip(current_speed - 2, min=self.v_min, max=self.v_max - 0.5)
        v_max = jnp.clip(current_speed + 2, min=self.v_min + 0.5, max=self.v_max)
        new_speed = jax.random.uniform(subkey, minval=v_min, maxval=v_max)

        # sample angle
        key, subkey = jax.random.split(key) # (2,)
        scale = jnp.clip(4 - jnp.square(new_speed - current_speed), min=0.0, max=1.0)
        velocity = unit_v * new_speed + jnp.array([unit_v[1], -unit_v[0]]) * jax.random.normal(subkey) * jnp.sqrt(scale)
        velocity = velocity / jnp.sqrt(jnp.sum(velocity**2)) * new_speed

        # sample deviation
        key, subkey = jax.random.split(key) # (2,)
        deviation = 0.5 * (velocity - current_velocity) + jax.random.normal(subkey, shape=(2,)) * 0.5

        new_task_state = TaskState(
            deviation=deviation,
            task=velocity,
            t=0.0,
            z=jnp.concatenate([
                deviation,
                velocity,
            ]),
            r=1.0
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
        displacement = self.get_position_from_envstate(next_env_state) - self.get_position_from_envstate(state.env_state)
        # displacement = next_env_state.pipeline_state.x.pos[0, :2] - state.env_state.pipeline_state.x.pos[0, :2]
        deviation = state.z_state.deviation + displacement - state.z_state.task * self.dt

        squared_distance = jnp.sum(deviation**2)
        reward = jnp.exp(-squared_distance*0.5)
        fail = squared_distance > 9
        new_deviation = jnp.where(fail, jnp.zeros_like(deviation), deviation)
        fitness_reward = jnp.array([next_env_state.reward - next_env_state.metrics["x_velocity"] + 3.0])

        next_task_state = state.z_state.replace(
            deviation=new_deviation,
            z=jnp.concatenate([
                new_deviation,
                state.z_state.task,
            ]),
        )

        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]), 
            fitness_reward=fitness_reward,
            done=jnp.where(done + fail > 0.9, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]))

        new_task_state = jax.lax.cond(
            next_env_state.done > 0.9,
            lambda _: state.initial_z_state,
            lambda _: next_task_state,
            None,
            )

        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info
    



class CircularWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        v_min=1.0,
        v_max=3.0,
        v_noise=0.5,
        a_mean=0.5236,
        dt=0.05,):
        super().__init__(env)
        self.a_mean = a_mean
        self.v_min = v_min
        self.v_max = v_max
        self.v_noise = v_noise
        self.dt = dt


    @property
    def z_size(self) -> int:
        return 5
    
    @property
    def has_z(self):
        return True
    

    def get_velocity_from_envstate(self, env_state: State) -> jax.Array:
        """This is for Ant, change this line if for other agent"""
        obs = env_state.obs
        current_velocity = obs[13:15] # (2,)

        return current_velocity 
    

    def get_position_from_envstate(self, env_state: State) -> jax.Array:
        """Change this line if doesn't work for other agent"""

        return env_state.pipeline_state.x.pos[0, :2] # (2,)
    

    def _init_task_state(self, env_state: State, key: jax.Array) -> PyTreeNode:
        """initialize task state"""
        current_velocity = self.get_velocity_from_envstate(env_state)
        current_speed = jnp.sqrt(jnp.sum(current_velocity**2))

        # find direction
        key, subkey = jax.random.split(key) # (2,)
        perturbed_velocity = current_velocity + jax.random.normal(subkey, shape=(2,)) * 0.25 # (2,)
        velocity = jnp.where(perturbed_velocity > 0, perturbed_velocity + 1e-2, perturbed_velocity - 1e-2)
        unit_v = velocity / jnp.sqrt(jnp.sum(velocity**2))
        
        # sample speed
        key, subkey = jax.random.split(key) # (2,)
        v_min = jnp.clip(current_speed - 1.25, min=self.v_min, max=self.v_max - 0.5)
        v_max = jnp.clip(current_speed + 1.25, min=self.v_min + 0.5, max=self.v_max)
        new_speed = jax.random.uniform(subkey, minval=v_min, maxval=v_max)
        
        # add angle change
        key, subkey = jax.random.split(key) # (2,)
        scale = jnp.clip(2 - jnp.square(new_speed - current_speed), min=0.0, max=1.0)
        velocity = unit_v * new_speed + jnp.array([unit_v[1], -unit_v[0]]) * jax.random.normal(subkey) * jnp.sqrt(scale)
        velocity = velocity / jnp.sqrt(jnp.sum(velocity**2)) * new_speed

        # add deviation
        key, subkey = jax.random.split(key) # (2,)
        deviation = 0.5 * (velocity - current_velocity) + jax.random.normal(subkey, shape=(2,)) * 0.5

        # sample angular velocity
        key, subkey = jax.random.split(key)
        omega = jnp.clip(jax.random.normal(subkey), -2.5, 2.5) * self.a_mean / new_speed

        # combine into task
        task = jnp.concatenate([
                velocity,
                jnp.array([omega,])
            ])
        new_task_state = TaskState(
            deviation=deviation,
            task=task,
            t=0.0,
            z=jnp.concatenate([
                deviation,
                task,
            ]),
            r=1.0,
        )
        
        return new_task_state
        


    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,    
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """return next state, reward, done, truncation"""

        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation
        # displacement = next_env_state.pipeline_state.x.pos[0, :2] - state.env_state.pipeline_state.x.pos[0, :2]
        displacement = self.get_position_from_envstate(next_env_state) - self.get_position_from_envstate(state.env_state)
        task_state = state.z_state

        v = task_state.task[:2, None]
        w = task_state.task[-1]
        theta = w * self.dt
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        acceleration_matrix = jnp.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ])
        displacement_matrix = jnp.array([
            [sin_theta, cos_theta - 1],
            [1 - cos_theta, sin_theta],
        ]) / w

        new_task = jnp.concatenate([
                jnp.reshape(acceleration_matrix @ v, (2,)),
                task_state.task[-1:]
            ])

        target_displacement = jnp.reshape(displacement_matrix @ v, (2,))
        deviation = task_state.deviation + displacement - target_displacement
        
        squared_distance = jnp.sum(deviation**2)
        reward = jnp.exp(-squared_distance*0.5)
        fail = squared_distance > 9
        deviation = jnp.where(fail, jnp.zeros_like(deviation), deviation)
        fitness_reward = 4 - jnp.sum(jnp.square(action))
        
        next_task_state = task_state.replace(
            deviation=deviation,
            task=new_task,
            z=jnp.concatenate([
                deviation,
                new_task,
            ]),
        )

        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]), 
            fitness_reward=jnp.array([fitness_reward]),
            done=jnp.where(done + fail > 0.9, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]))

        new_task_state = jax.lax.cond(
            next_env_state.done > 0.9,
            lambda _: state.initial_z_state,
            lambda _: next_task_state,
            None,
            )

        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info
    

    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, Tuple[jax.Array, ...]]:
        """extract observations and z (will be empty tuple if has_z == False)"""
        return state.env_state.obs, state.z_state.z




class GPTaskState(TaskState):
    position_offset: jnp.ndarray # (2,)
    coefs: jnp.ndarray # (2, way_points)


class AntMaternWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        horizon: float = 3,
        way_points: int = 12,
        var: float = 2.25,
        l: float = 1,
        v_noise_var: float = 0.0625,
        dt: float = 0.05,):
        
        super().__init__(env)
        self.dt = dt
        self.way_points = way_points
        self.z_dim = 2 * way_points + 3 # task, deviation, time
        matern_kernel = IntegrateMatern(l)
        self.horizon = horizon

        self.normalization_scale = float(way_points / (horizon * np.sqrt(var)))

        # prior sampling
        s_t = (1 + np.arange(way_points)) / way_points * horizon
        cov = matern_kernel.kernel(s_t, s_t) + np.eye(way_points) * 1e-8
        self.cov = jnp.array(cov, dtype=jnp.float32)


        x_v_cov = matern_kernel.xv_cross_covariance(s_t, np.array([0.0])) 
        prior_cov = cov - x_v_cov @ x_v_cov.T * var / (var + v_noise_var)
        self.prior_L = jnp.array(lg.cholesky(prior_cov * var), dtype=jnp.float32) # (way_points, way_points)
        self.prior_mu = jnp.array(x_v_cov.T * var / (var + v_noise_var), dtype=jnp.float32) # (1, way_points)


        t2 = jnp.array(s_t, dtype=jnp.float32) # (way_points, )
        self.s_ts = t2

        def covariance_fn(t):
            diff_t = jnp.abs(t - t2)
            min_t = jnp.where(t < t2, t, t2)
            cov = l**2 * (
                jnp.exp(-t/l) + jnp.exp(-t2/l) -jnp.exp(-diff_t/l) - 1
                ) + 2*l*min_t
            return cov # (way_points,)
        

        # inference
        # (d, way_points) @ (way_points,)
        self._base_fn = covariance_fn # (way_points,)



    @property
    def z_size(self):
        return self.z_dim

    @property
    def has_z(self):
        return True
    

    def _init_task_state(self, env_state: State, key: jax.Array) -> PyTreeNode:
        """initialize task state"""
        obs = env_state.obs
        current_velocity = obs[13:15] # (2,)

        key, subkey = jax.random.split(key)
        prior_s = self.prior_mu * current_velocity[:, None] # (2, way_points)
        task_s = prior_s + jax.random.normal(subkey, (2, self.way_points)) @ self.prior_L.T # (2, way_points)
        
        coefs = jnp.linalg.solve(self.cov, task_s.T).T
        P = jnp.eye(self.way_points) - jnp.eye(self.way_points, k=1)
        normalized_task = jnp.reshape(task_s @ P, (-1,)) * self.normalization_scale

        # sample deviation
        key, subkey = jax.random.split(key) # (2,)
        deviation = jax.random.normal(subkey, shape=(2,)) * 0.5

        new_task_state = GPTaskState(
            deviation=deviation,
            task=normalized_task,
            t=float(self.horizon),
            z=jnp.concatenate([
                deviation,
                normalized_task,
                jnp.array([self.horizon], dtype=jnp.float32),
            ]),
            r=1.0,
            coefs=coefs,
            position_offset=-env_state.pipeline_state.x.pos[0, :2] + deviation,
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

        current_t = self.horizon - state.z_state.t + self.dt
        target_position = jnp.sum(state.z_state.coefs * self._base_fn(current_t), axis=-1)
        current_position = next_env_state.pipeline_state.x.pos[0, :2] + state.z_state.position_offset
        deviation = current_position - target_position

        deviation, task_z, next_t = jax.lax.cond(
            state.z_state.t < self.dt,
            lambda x: (jnp.zeros_like(deviation), jnp.zeros_like(state.z_state.task), -1.0),
            lambda x: (deviation, state.z_state.task, state.z_state.t - self.dt),
            None,
        )

        squared_distance = jnp.sum(deviation**2)
        reward = jnp.exp(-squared_distance*0.5)
        fail = squared_distance > 9

        new_deviation = jnp.where(fail, jnp.zeros_like(deviation), deviation)
        new_position_offset = jnp.where(
            fail, 
            target_position - next_env_state.pipeline_state.x.pos[0, :2], 
            state.z_state.position_offset)

        next_task_state = state.z_state.replace(
            deviation=new_deviation,
            z=jnp.concatenate([
                new_deviation,
                task_z,
                jnp.array([next_t]),
            ]),
            t=next_t,
            position_offset=new_position_offset,
        )

        fitness_reward = jnp.array([next_env_state.reward - next_env_state.metrics["x_velocity"] + 3.0])
        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]), 
            fitness_reward=fitness_reward,
            done=jnp.where(done + fail > 0.9, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]))

        new_task_state = jax.lax.cond(
            next_env_state.done > 0.9,
            lambda _: state.initial_z_state,
            lambda _: next_task_state,
            None,
            )

        return state.replace(env_state=next_env_state, z_state=new_task_state), transition_info
    

    def resample_deviation(self, state: GeneralizedState) -> GeneralizedState:
        key, subkey = jax.random.split(state.key)
        z_state = state.z_state
        deviation = jax.lax.select(z_state.t > 0, jax.random.normal(subkey, shape=(2,)) * 0.5, jnp.zeros(2))
        env_state = state.env_state
        target_position = jnp.sum(z_state.coefs * self._base_fn(self.horizon - z_state.t), axis=-1)
        new_position_offset = target_position - env_state.pipeline_state.x.pos[0, :2] + deviation

        new_task_state = z_state.replace(
            deviation=deviation,
            z=jnp.concatenate([
                deviation,
                z_state.task,
                jnp.array([z_state.t]),
            ]),
            position_offset=new_position_offset,
        )
        
        state = state.replace(z_state=new_task_state, key=key)
        return state
    

    




