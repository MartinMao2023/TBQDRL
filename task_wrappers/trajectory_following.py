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
from data_struct.qd_transitions import QDTransitionInfo
from custom_types import Params, RNGKey, Env, EnvState
from .tools import IntegrateMatern



class MaternTaskState:
    position_offset: jax.Array # (2,)

    normalized_ds: jax.Array # (2, way_points)
    next_normalized_ds: jax.Array # (2, way_points)

    task_s: jax.Array # (2, way_points + 1)

    task_v: jax.Array # (2, way_points + 1)
    next_task_v: jax.Array # (2, way_points + 1)

    # coefs: jnp.ndarray # (2, 4)
    remaining_t: float # actual remaining time
    t: float # absolute time within a period
    z: jnp.ndarray # (-1,)


class GeneralizedState:
    env_state: State
    z_state: MaternTaskState
    initial_z_state: MaternTaskState # used in reset
    key: jax.Array



class FiniteMaternWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        horizon: float = 4,
        way_points: int = 8,
        var: float = 2.25,
        l: float = 1,
        v_noise_var: float = 0.0625,
        dt: float = 0.05,):
        
        super().__init__(env)
        self.dt = dt
        self.way_points = way_points
        self.z_dim = 4 * way_points + 6 # task, deviation, time * 2
        matern_kernel = IntegrateMatern(l)
        self.horizon = horizon

        period_t = float(horizon / way_points)
        self.period_t = period_t
        self.inv_period_t = 1 / period_t
        self.ds_normalization_scale = 1 / np.sqrt(2 * l * (l * np.exp(-period_t/l) - l + period_t))

        way_points_t = np.arange(way_points + 1) * period_t
        prior_cov = np.zeros((way_points * 2 + 1, way_points * 2 + 1))
        prior_cov[:way_points, :way_points] = matern_kernel.kernel(way_points_t[1:], way_points_t[1:])
        prior_cov[way_points:, way_points:] = matern_kernel.derivative_kernel(way_points_t, way_points_t)
        prior_cov[:way_points, way_points:] = matern_kernel.xv_cross_covariance(way_points_t[1:], way_points_t)
        prior_cov[way_points:, :way_points] = prior_cov[:way_points, way_points:].T
        Sigma22 = np.zeros((4, 4))
        Sigma22[2:, 2:] = prior_cov[way_points: way_points + 2, way_points: way_points + 2]
        Sigma22[1, 1] = prior_cov[0, 0]
        Sigma22[1, 2:] = prior_cov[0, way_points: way_points + 2]
        Sigma22[2:, 1] = prior_cov[way_points: way_points + 2, 0]
        Sigma22 = Sigma22 + np.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        prior_cov = prior_cov * var

        # Posterior sampling: Sigma12 @ Sigma22_inv @ Y = Sigma12 * v / Var_y
        Sigma12 = prior_cov[:, way_points: way_points + 1] # (all x 1)
        self.posterior_mu_T = jnp.array(Sigma12.T / (var + v_noise_var), dtype=jnp.float32) # (1, 2 * way_points + 1)
        posterior_cov = prior_cov - Sigma12 @ Sigma12.T / (var + v_noise_var)
        self.posterior_L = jnp.array(lg.cholesky(posterior_cov), dtype=jnp.float32) # target = L @ X 

        # Inference -> (2,)
        self._base_fn = lambda t: jnp.array([1.0, jnp.exp(-t/l), jnp.exp(t/l), t]) # (4,)
        A_T = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [l**2*(np.exp(-period_t/l) - 1) + 1, l**2, -l**2*np.exp(-period_t/l), 2*l],
            [l, -l, 0.0, 0.0],
            [-l*np.exp(-period_t/l), 0.0, l*np.exp(-period_t/l), 0.0],
        ])
        self.Sigma22_inv_AT = lg.solve(Sigma22, A_T) # (4, 4)



    @property
    def z_size(self):
        return self.z_dim

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

        # sample way_points data
        current_velocity = self.get_velocity_from_envstate(env_state) # (2,)
        mean_ys = self.posterior_mu_T * current_velocity[:, None] # (2, 2 * waypoints + 1)
        key, subkey = jax.random.split(key)
        ys = mean_ys + jax.random.normal(subkey, (2, 2 * self.way_points + 1)) @ self.posterior_L # (2, 2 * way_points + 1)

        # sample deviation
        deviation = jnp.zeros((2, 1)) # (2, 1)
        
        # collect data
        task_v = ys[:, self.way_points:] # (2, way_points + 1)
        next_task_v = jnp.concatenate([task_v[:, 1:], jnp.zeros((2, 1))], axis=-1)
        task_s = jnp.concatenate([-deviation, ys[:, :self.way_points] - deviation], axis=-1) # (2, way_points + 1)
        P = jnp.eye(self.way_points) - jnp.eye(self.way_points, k=1)
        normalized_ds = ys[:, :self.way_points] @ P * self.ds_normalization_scale # (2, way_points)
        next_normalized_ds = jnp.concatenate([normalized_ds[:, 1:], jnp.zeros((2, 1))], axis=-1)
        z = jnp.concatenate([-deviation, normalized_ds, task_v, jnp.ones((2, 1))], axis=-1)

        new_task_state = MaternTaskState(
            position_offset=-self.get_position_from_envstate(env_state),
            normalized_ds=normalized_ds,
            next_normalized_ds=next_normalized_ds,
            task_s=task_s,
            task_v=task_v,
            next_task_v=next_task_v,
            remaining_t=float(self.horizon),
            t=0.0,
            z=jnp.reshape(z, (-1,)),
        )

        return new_task_state
    

    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, Tuple[jax.Array, ...]]:
        """extract observations and z (will be empty tuple if has_z == False)"""
        return state.env_state.obs, state.z_state.z
    

    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,
        inv_r: jax.Array = 1.0, # inverse of radius (array)
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """return next state, reward, done, truncation"""
        
        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation

        # check shift
        current_t = state.z_state.t + self.dt
        need_shift = current_t > self.period_t

        shifted_data = (
            current_t - self.period_t, 
            state.z_state.next_normalized_ds, 
            state.z_state.next_task_v,
            jnp.concatenate([state.z_state.task_s[:, 1:], jnp.zeros((2, 1))], axis=-1), 
            jnp.concatenate([state.z_state.next_normalized_ds[:, 1:], jnp.zeros((2, 1))], axis=-1), 
            jnp.concatenate([state.z_state.next_task_v[:, 1:], jnp.zeros((2, 1))], axis=-1),
        )
        regular_data = (
            current_t, 
            state.z_state.normalized_ds, 
            state.z_state.task_v,
            state.z_state.task_s, 
            state.z_state.next_normalized_ds, 
            state.z_state.next_task_v,
        )
        (current_t, normalized_ds, task_v, task_s, next_normalized_ds, next_task_v) = jax.lax.cond(
            need_shift,
            lambda x: shifted_data,
            lambda x: regular_data,
            None,
        ) # apply changes

        coefs = jnp.concatenate([task_s[:, :2], task_v[:, :2]], axis=-1) @ self.Sigma22_inv_AT # (2, 4)
        target_position = jnp.sum(coefs * self._base_fn(current_t), axis=-1) # (2,)
        current_position = self.get_position_from_envstate(next_env_state) + state.z_state.position_offset
        deviation = current_position - target_position # (2,)

        # check task finish
        new_remaining_t = state.z_state.remaining_t - self.dt
        deviation, new_remaining_t, normalized_t = jax.lax.cond(
            new_remaining_t < 0,
            lambda x: (jnp.zeros_like(deviation), -1.0, -1.0),
            lambda x: (deviation, new_remaining_t, 2*new_remaining_t/self.horizon - 1),
            None,
        )
        squared_distance = jnp.sum(jnp.square(deviation * inv_r))
        reward = jnp.exp(-squared_distance*0.5)

        # reset deviation if fail
        fail = squared_distance > 9
        new_deviation = jnp.where(fail, jnp.zeros((2, 1)), jnp.reshape(deviation, (2, 1))) # (2, 1)
        new_position_offset = jnp.where(
            fail, 
            state.z_state.position_offset - deviation,
            state.z_state.position_offset)

        # compose z
        t_portion = current_t * self.inv_period_t
        ds_repr = normalized_ds + t_portion * (next_normalized_ds - normalized_ds)
        v_repr = task_v + t_portion * (next_task_v - task_v)
        z = jnp.concatenate([-new_deviation, ds_repr, v_repr, jnp.ones((2, 1)) * normalized_t], axis=-1)

        next_task_state = MaternTaskState(
            position_offset=new_position_offset,
            normalized_ds=normalized_ds,
            next_normalized_ds=next_normalized_ds,
            task_s=task_s,
            task_v=task_v,
            next_task_v=next_task_v,
            remaining_t=new_remaining_t,
            t=current_t,
            z=jnp.reshape(z, (-1,)),
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
    


    def resample_deviation(self, state: GeneralizedState, deviation_std: jax.Array = 0.5) -> GeneralizedState:
        z_state = state.z_state
        env_state = state.env_state
        key, subkey = jax.random.split(state.key)
        deviation = jax.lax.select(
            z_state.remaining_t > 0, 
            jax.random.normal(subkey, shape=(2, 1)) * deviation_std, 
            jnp.zeros((2, 1)),
            )
        
        coefs = jnp.concatenate([z_state.task_s[:, :2], z_state.task_v[:, :2]], axis=-1) @ self.Sigma22_inv_AT # (2, 4)
        target_position = jnp.sum(coefs * self._base_fn(z_state.t), axis=-1) # (2,)
        new_position_offset = target_position - self.get_position_from_envstate(env_state) + jnp.reshape(deviation, (-1,))

        z = jnp.concatenate([-deviation, jnp.reshape(z_state.z, (2, -1))[:, 1:]], axis=-1)

        new_task_state = z_state.replace(
            position_offset=new_position_offset,
            z=jnp.reshape(z, (-1,)), # change 
        )

        state = state.replace(z_state=new_task_state, key=key)
        return state
    

    




