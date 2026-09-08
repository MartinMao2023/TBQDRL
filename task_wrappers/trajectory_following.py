from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.linalg as lg
from flax.struct import PyTreeNode
from brax.envs.base import State
from brax.envs.base import Env, PipelineEnv
from task_wrappers.base import BaseQDTaskWrapper
# from data_struct.states import GeneralizedState
# from data_struct.transitions import TransitionInfo
from data_struct.qd_transitions import QDTransitionInfo
from custom_types import Params, RNGKey, Env, EnvState
from .tools import IntegrateMatern



class MaternTaskState(PyTreeNode):
    position_offset: jax.Array # (2,)
    last_action: jax.Array # (action_dim,)

    reshaped_sequence: jax.Array # (2, way_points + 1, 4)
    padding_element: jax.Array # (2, 1, 4)

    steps_taken: jax.Array
    cycle_t: float # absolute time within a period
    z: jnp.ndarray # (-1,)
    # z in the form [ys_x, ys_y, vs_x, vs_y, sin(cycle_t), cos(cycle_t), 2 * task_t - 1]


class GeneralizedState(PyTreeNode):
    env_state: State
    z_state: MaternTaskState
    initial_z_state: MaternTaskState # used in reset
    key: jax.Array



class AntFiniteMaternWrapper(BaseQDTaskWrapper):
    def __init__(
        self, 
        env: Env, 
        way_points: int = 12,
        steps_per_way_point: int = 8,
        var: float = 2.25,
        l: float = 1,
        tolerance_radius: float = 0.0,
        inner_radius: float = 0.5,
        outer_radius: float = 1.5,
        max_radius: float = 3.0,
        v_noise_var: float = 0.25,
        dt: float = 0.05,
        ):
        
        super().__init__(env, way_points, steps_per_way_point, dt)
        self.z_dim = 4 * way_points + 7 # 2 * (2 * way_points + 2) + 3 
        matern_kernel = IntegrateMatern(l)
        self.t_normalization_scale = dt * 2 / self.horizon

        period_t = self.period_t
        self.inv_period_t = 1 / period_t
        self.omega = jnp.pi * 2 / period_t

        self.tolerance_radius = tolerance_radius
        self.max_square_dist = max_radius**2
        self.inner_scale = 1 / inner_radius**2
        self.outer_scale = 1 / outer_radius**2

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
        P = np.array([
            [1.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 1.0, 0.0,],
            [0.0, 1.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0, 1.0,],
        ])
        Sigma22 = P.T @ Sigma22 @ P + np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]) 

        P = np.zeros((way_points * 2 + 1, way_points * 2 + 1))
        for i in range(way_points):
            P[i, 2 * i + 1] = 1
        for i in range(way_points + 1):
            P[way_points + i, 2 * i] = 1
        prior_cov = var * (P.T @ prior_cov @ P)

        # Posterior sampling: Sigma12 @ Sigma22_inv @ Y = Sigma12 * v / Var_y
        Sigma12 = prior_cov[:, :1] # (all x 1)
        self.posterior_mu_T = jnp.array(Sigma12.T / (var + v_noise_var), dtype=jnp.float32) # (1, 2 * way_points + 1)
        posterior_cov = prior_cov - Sigma12 @ Sigma12.T / (var + v_noise_var)
        self.posterior_L = jnp.array(lg.cholesky(posterior_cov), dtype=jnp.float32) # target = L @ X 

        # Inference -> (2,)
        self._base_fn = lambda t: jnp.array([1.0, jnp.exp(-t/l), jnp.exp(t/l), t]) # (4,)
        S_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [l, -l, 0.0, 0.0],
            [l**2*(np.exp(-period_t/l) - 1) + 1, l**2, -l**2*np.exp(-period_t/l), 2*l],
            [-l*np.exp(-period_t/l), 0.0, l*np.exp(-period_t/l), 0.0],
        ])
        V_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2*l, -l, -l*np.exp(-period_t/l), 0.0],
            [0.0, 0.0, np.exp(-period_t/l), 0.0],
        ])
        self.Sigma22_inv_S = jnp.array(lg.solve(Sigma22, S_matrix), dtype=jnp.float32)
        self.Sigma22_inv_V = jnp.array(lg.solve(Sigma22, V_matrix), dtype=jnp.float32)
        # jnp.sum(((2, 4) @ self.Sigma22_inv_AT) * base_fn, axis=-1) # shape of (2,)


    @property
    def z_size(self):
        return self.z_dim
    

    def _get_position_from_envstate(self, env_state: State) -> jax.Array:
        """Change this line if doesn't work for other agent"""

        return env_state.pipeline_state.x.pos[0, :2] # (2,)


    def _organize_z_state(self, sampled_sequence: jax.Array, position_offset: jax.Array) -> MaternTaskState:

        task_sequence = jnp.concatenate(
            [jnp.zeros((2, 1)), sampled_sequence, sampled_sequence[:, -2:-1], jnp.zeros((2, 1))],
            axis=-1,
        ) # (2, 2 * way_points + 4)
        task_sequence = jnp.reshape(task_sequence, (2, -1, 2)) # (2, way_points + 2, 2)
        reshaped_sequence = jnp.concatenate(
            [task_sequence[:, :-1], task_sequence[:, 1:]],
            axis=-1,
            ) # (2, way_points + 1, 4)
        padding_element = jnp.concatenate([task_sequence[:, -1:], task_sequence[:, -1:]], axis=-1) # (2, 1, 4)
        ys = task_sequence[:, :-1, 0] # (2, way_points + 1)
        ys = jnp.diff(ys, axis=1, prepend=0) * self.inv_period_t # special step for x-y position
        vs = task_sequence[:, :-1, 1] # (2, way_points + 1)
        z = jnp.concatenate(
            [
                jnp.reshape(ys, (-1,)),
                jnp.reshape(vs, (-1,)),
                jnp.array([0.0, 1.0, -1.0]),
            ],
            axis=-1,
        )

        task_state = MaternTaskState(
            position_offset=position_offset,
            last_action=jnp.zeros((self.action_size,)),
            reshaped_sequence=reshaped_sequence,
            padding_element=padding_element,
            steps_taken=jnp.int32(0),
            cycle_t=0.0,
            z=z,
        )

        return task_state
    

    def sample_task(self, env_state: State, key: jax.Array) -> MaternTaskState:
        """initialize task state"""

        # sample way_points data
        means = self.posterior_mu_T * env_state.obs[13:15, None] # (2, 2 * waypoints + 1)
        sampled_sequence = means + jax.random.normal(key, (2, 2 * self.way_points + 1)) @ self.posterior_L.T # (2, 2 * way_points + 1)
        position_offset=-self._get_position_from_envstate(env_state)
        task_state = self._organize_z_state(sampled_sequence, position_offset)

        return task_state


    def get_obs(self, state: GeneralizedState) -> Tuple[jax.Array, Tuple[jax.Array, ...]]:
        """extract observations and z (will be empty tuple if has_z == False)"""
        # return jnp.concatenate(
        #     [state.env_state.obs, state.z_state.last_action], 
        #     axis=0,
        #     ), state.z_state.z

        return state.env_state.obs, state.z_state.z
    

    def step(
        self, 
        state: GeneralizedState, 
        action: jax.Array,
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """return next state, reward, done, truncation"""

        current_cycle_t = state.z_state.cycle_t + self.dt # always assume this is smaller than period_t
        current_step_num = state.z_state.steps_taken + 1

        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info['truncation']
        done = next_env_state.done - truncation

        has_reset = next_env_state.done > 0.5
        z_state, current_step_num = jax.lax.cond(
            has_reset,
            lambda _: (state.initial_z_state, current_step_num % self.steps_per_way_point),
            lambda _: (state.z_state, current_step_num),
            None,
            )

        complete = current_step_num >= self.max_step_num
        completed = current_step_num > self.max_step_num
        normalized_task_t, z_cycle_t = jax.lax.cond(
            completed,
            lambda _: (1.0, self.period_t),
            lambda _: (current_step_num * self.t_normalization_scale - 1.0, current_cycle_t),
            None,
            ) # make sure z makes sense
        
        base_fn_values = self._base_fn(z_cycle_t) # (4,)
        shifted_ys = jnp.sum(
            (z_state.reshaped_sequence @ self.Sigma22_inv_S) * base_fn_values,  # (2, way_points + 1, 4)
            axis=-1,
            ) # (2, way_points + 1)
        shifted_vs = jnp.sum(
            (z_state.reshaped_sequence @ self.Sigma22_inv_V) * base_fn_values,  # (2, way_points + 1, 4)
            axis=-1,
            ) # (2, way_points + 1)
        
        target_position = shifted_ys[:, 0] # (2,)
        current_position = self._get_position_from_envstate(next_env_state) + z_state.position_offset
        deviation = target_position - current_position # (2,)

        # calculate reward
        l2_dist = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.square(deviation))) - self.tolerance_radius, 
            0.0,
            )
        squared_distance = jnp.square(l2_dist)
        fail = squared_distance > self.max_square_dist
        reward = 0.5 * (
            jnp.exp(-squared_distance * self.inner_scale) + 
            jnp.exp(-squared_distance * self.outer_scale)
            )
        reward = jnp.where(completed | has_reset, 0.0, reward)
        last_action = jnp.where(has_reset, jnp.zeros_like(action), action)
        compensation = jnp.where(
            fail | has_reset, 
            deviation,
            0.0,
            )
        corrected_deviation = target_position - current_position - compensation # (2,)

        z = jnp.concatenate([
            jnp.reshape(
                jnp.diff(shifted_ys - corrected_deviation[:, None], axis=-1, prepend=0), 
                (-1,),
            ) * self.inv_period_t, # special step for x-y position
            jnp.reshape(shifted_vs, (-1,)),
            jnp.array([
                jnp.sin(self.omega * z_cycle_t), 
                jnp.cos(self.omega * z_cycle_t), 
                normalized_task_t,
                ]),
            ], 
            axis=-1,
        )

        next_task_state = z_state.replace(
            position_offset=z_state.position_offset + compensation,
            last_action=last_action,
            steps_taken=current_step_num,
            cycle_t=current_cycle_t,
            z=z,
        )

        fitness_reward = jnp.array([next_env_state.reward - next_env_state.metrics["x_velocity"] + 3.0])
        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]), 
            fitness_reward=fitness_reward,
            done=jnp.where(fail | (done > 0.5), jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            completion=jnp.where(complete, jnp.ones(shape=(1,)), jnp.zeros(shape=(1,))),
            truncation=jnp.array([truncation]),
            broken=jnp.array([0.0]))

        return state.replace(env_state=next_env_state, z_state=next_task_state), transition_info


    def shift(self, state: GeneralizedState) -> GeneralizedState:
        """
        To shift:
            1) take cycle_t - self.period_t
            2) shift and pad reshaped_sequence
        """
        
        z_state = state.z_state
        new_reshaped_sequence = jnp.concatenate(
            [z_state.reshaped_sequence[:, 1:, :], z_state.padding_element],
            axis=1,
        ) # (2, way_points + 1, 4)

        next_task_state = z_state.replace(
            cycle_t=z_state.cycle_t - self.period_t,
            reshaped_sequence=new_reshaped_sequence,
        )

        return state.replace(z_state=next_task_state)
    



class AntLineMaternWrapper(AntFiniteMaternWrapper):


    def sample_task(self, env_state, key):
        # sample way_points data

        key1, key2 = jax.random.split(key)
        angle = jax.random.uniform(key1, minval=0.0, maxval=2*jnp.pi)
        speed = jax.random.uniform(key2, 1.5, 3.0)
        target_velocity = jnp.array([
            jnp.cos(angle) * speed,
            jnp.sin(angle) * speed,
        ]) # (2,)

        current_velocity = env_state.obs[13:15]


        def scan_track(carry, _):
            pos, vel = carry
            dv = target_velocity - vel
            dv_norm = jnp.sqrt(jnp.sum(dv**2))
            acc = dv / (1e-6 + dv_norm) * 0.8

            acc_t = jnp.minimum(self.period_t, dv_norm * 1.25)
            rest_t = self.period_t - acc_t
            d_pos = vel * acc_t + 0.5 * acc * acc_t**2 + target_velocity * rest_t

            new_pos = pos + d_pos
            new_vel = vel + acc * acc_t

            return (new_pos, new_vel), jnp.concatenate([new_pos, new_vel]) # (pos_x, pos_y, vel_x, vel_y)


        _, sampled_sequence = jax.lax.scan(
            scan_track,
            (jnp.zeros((2,)), current_velocity),
            length=self.way_points,
        )
        sampled_sequence = jnp.reshape(
            jnp.concatenate([
                current_velocity, 
                jnp.reshape(sampled_sequence, (-1)),
                ]),
            (-1, 2),
        ).T # (2, 2 * way_points + 1)

        position_offset=-self._get_position_from_envstate(env_state)
        task_state = self._organize_z_state(sampled_sequence, position_offset)

        return task_state



