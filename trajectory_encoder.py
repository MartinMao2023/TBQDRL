from functools import partial
from typing import Any, Tuple, Callable

import jax
from jax import numpy as jnp
from custom_types import Params, RNGKey, Env, EnvState
from flax.struct import PyTreeNode
import numpy as np
import numpy.linalg as lg
import abc


class IntegrateMatern:
    def __init__(
        self,
        l: float, # lengthscale
        v: float = 0.5, # smoothness
        ):

        self.l = l
        if v == 0.5:
            def derivative_kernel(t1, t2):
                
                return np.exp(-np.abs(t1.reshape(-1, 1) - t2.reshape(1, -1))/l)

            self.derivative_kernel = derivative_kernel

            def kernel(t1, t2):
                t1 = t1.reshape(-1, 1)
                t2 = t2.reshape(1, -1)
                diff_t = np.abs(t1 - t2)
                min_t = np.where(t1 < t2, t1, t2)
                cov = l**2 * (
                    np.exp(-t1/l) + np.exp(-t2/l) -np.exp(-diff_t/l) - 1
                    ) + 2*l*min_t
                return cov
            
            self.kernel = kernel

            def cross_kernel(xt, vt):
                xt = xt.reshape(-1, 1)
                vt = vt.reshape(1, -1)
                cov = np.where(
                    xt > vt,
                    2*l - l * np.exp((vt - xt)/l),
                    l * np.exp((xt - vt)/l),
                    ) - l * np.exp(-vt/l)

                return cov
            
            self.xv_kernel = cross_kernel


        # elif v == 1.5:
        #     self.kernel = lambda d: (1 + math.sqrt(3)*d/l) * np.exp(-math.sqrt(3)*d/l)
        # elif v == 2.5:
        #     self.kernel = lambda d: (1 + math.sqrt(5)*d/l + 5/3*(d/l)**2) * np.exp(-math.sqrt(5)*d/l)
        else:
            raise Exception(f"not implemented smoothness, v={v}")


    def covariance(
        self,
        t1: np.ndarray, # shape N1 x 1
        t2: np.ndarray, # shape N2 x 1
        ) -> np.ndarray: # shape N1 x N2

        cov = self.kernel(t1, t2)

        return cov
    

    def xv_cross_covariance(
        self, 
        xt: np.ndarray, 
        vt: np.ndarray,
        ) -> np.ndarray:

        cov = self.xv_kernel(xt, vt)
        return cov



class TaskState(PyTreeNode):

    deviation: jnp.ndarray # shape of (2,)
    task: jnp.ndarray # shape of (2, d)
    t: float
    z: jnp.ndarray # shape of (2 + 2 * d + 1,)
    key: RNGKey
    r: float



class GPTaskState(TaskState):

    position: jnp.ndarray # (2,)



class TaskEncoder(abc.ABC):
    """Interface for driving training and inference."""

    @abc.abstractmethod
    def reset(self, obs: jnp.ndarray, key: RNGKey) -> TaskState:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, task_state: TaskState, displacement: jnp.ndarray) -> TaskState:
        """Run one timestep of the environment's dynamics."""

    @property
    @abc.abstractmethod
    def z_dim(self) -> int:
        """The size of the observation vector returned in step and reset."""



class MaternEncoder(TaskEncoder):
    def __init__(
        self,
        l: float = 2.0,
        horizon: int = 3,
        period_length: float = 1.0,
        var: float = 2.25,
        v_noise_var: float = 0.0625,
        dt: float = 0.05,
        ):

        self.l = l
        self.horizon = horizon
        t1 = 0.5 * period_length
        self.t1 = t1
        self.dt = dt
        self.var = var
        t2 = period_length
        matern_object = IntegrateMatern(l, v=0.5)

        # inference
        cov = np.zeros((4, 4))
        cov[:2, :2] = matern_object.derivative_kernel(np.array([0.0, t2]), np.array([0.0, t2]))
        cov[2:, 2:] = matern_object.kernel(np.array([t1, t2]), np.array([t1, t2]))
        cross_cov = matern_object.xv_cross_covariance(np.array([t1, t2]), np.array([0.0, t2]))
        cov[2:, :2] = cross_cov
        cov[:2, 2:] = cross_cov.T

        P = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        inv_cov = lg.inv(P @ cov @ P.T)

        A_1 = inv_cov @ np.array([
            [0, -l, 0, l],
            [-l**2*np.exp(-t1/l), l**2, 2*l, l**2*(np.exp(-t1/l) - 1)],
            [-l**2*np.exp(-t2/l), l**2, 2*l, l**2*(np.exp(-t2/l) - 1)],
            [l*np.exp(-t2/l), 0, 0, -l*np.exp(-t2/l)],
        ])
        self.A_1 = jnp.array(A_1, dtype=jnp.float32)

        A_2 = inv_cov @ np.array([
            [0, -l, 0, l],
            [0, l**2*(1 - np.exp(t1/l)), 0, l**2*(np.exp(-t1/l) - 1) + 2*l*t1],
            [-l**2*np.exp(-t2/l), l**2, 2*l, l**2*np.exp(-t2/l)-l**2],
            [l*np.exp(-t2/l), 0, 0, -l*np.exp(-t2/l)],
        ])
        self.A_2 = jnp.array(A_2, dtype=jnp.float32)
        
        # posterior (used in step)
        cov2 = np.ones((3, 3))
        cov2[:2, :2] = matern_object.kernel(np.array([t1, t2]), np.array([t1, t2]))
        cross_cov2 = matern_object.xv_cross_covariance(np.array([t1, t2]), np.array([t2,]))
        cov2[:2, 2:] = cross_cov2
        cov2[2:, :2] = cross_cov2.T
        sigma12 = np.ones((3, 1))
        sigma12[:2, :] = matern_object.xv_cross_covariance(np.array([t1, t2]), np.array([0.0,]))
        sigma12[2:, :] = matern_object.derivative_kernel(np.array([0.0]), np.array([t2]))
        posterior_cov = cov2 + np.eye(3) * 1e-6 - sigma12 @ sigma12.T
        self.posterior_cov = jnp.array(var * posterior_cov, dtype=jnp.float32)[None, :, :]  # (1, 3, 3)
        self.posterior_mu = jnp.array(sigma12.T, dtype=jnp.float32) # (1, 3)

        # prior (used in reset)
        cov3 = np.ones((1 + horizon * 3, 1 + horizon * 3))
        cov3[:1 + horizon, :1 + horizon] = matern_object.derivative_kernel(
            np.arange(horizon + 1) * t2, 
            np.arange(horizon + 1) * t2,
            )
        s_t = np.reshape(
            np.array([t1, t2]) + np.arange(horizon).reshape(-1, 1) * t2,
            (-1,))
        cov3[1 + horizon:, 1 + horizon:] =  matern_object.kernel(s_t, s_t)
        cov3[1 + horizon:, :1 + horizon] = matern_object.xv_cross_covariance(
            s_t, 
            np.arange(horizon + 1) * t2
            )
        cov3[:1 + horizon, 1 + horizon:] = cov3[1 + horizon:, :1 + horizon].T

        P_combination = np.eye(1 + horizon * 3)
        for i in range(2*horizon - 2):
            column_index = i + horizon + 3
            row_index = horizon + 2 + i // 2 * 2
            P_combination[row_index, column_index] = -1

        P_position = np.zeros((1 + horizon * 3, 1 + horizon * 3))
        for i in range(1 + horizon): # v positions
            P_position[i, 3 * i] = 1

        for i in range(2 * horizon): # s positions
            P_position[1 + horizon + i, 1 + i % 2 + (i // 2) * 3] = 1

        P = P_combination @ P_position
        cov3 = P.T @ cov3 @ P

        prior_cov = cov3 + np.eye(1 + horizon * 3) * 1e-6 - cov3[:, :1] @ cov3[:1, :] * var / (var + v_noise_var)
        self.prior_cov = jnp.array(var * prior_cov, dtype=jnp.float32)[None, :, :]  # (1, 10, 10)
        self.prior_mu = jnp.array(cov3[:1, :] * var / (var + v_noise_var), dtype=jnp.float32) # (1, 10)


    @property
    def z_dim(self) -> int:
        return 5 + 6 * self.horizon 


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def predict(self, task, t):
        t_value, coefs, position_offset = jax.lax.cond(
            t < 1,
            self._default_params,
            self._shifted_params,
            task, t
        )
        A = jnp.where(t_value < self.t1, self.A_1, self.A_2)
        base_array = jnp.array([jnp.exp(t_value/self.l), jnp.exp(-t_value/self.l), t_value, 1.0]).reshape(-1, 1)

        # (2, 4) @ (4, 4) @ (4, 1) -> (2, 1)
        position = coefs @ A @ base_array
        return jnp.reshape(position, (-1,)) + position_offset, position_offset
    

    def _default_params(self, task, t):
        coefs = task[:, :4]
        return t, coefs, jnp.zeros((2,))


    def _shifted_params(self, task, t):
        coefs = task[:, 3: 7]
        return t - 1, coefs, task[:, 2]
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def step(
        self, 
        task_state: GPTaskState, 
        displacement: jnp.ndarray,
        ) -> GPTaskState:

        new_position = displacement + task_state.position
        target_position, offset = self.predict(task_state.task, task_state.t + self.dt)
        deviation = new_position - target_position
        key, subkey = jax.random.split(task_state.key)
        sampled_matrix = jax.random.multivariate_normal(
            subkey,
            self.posterior_mu * task_state.task[:, -1:], # (2, 3)
            self.posterior_cov, # (1, 3, 3)
        ) # (2, 3)
        candidate_matrix = jnp.concatenate([task_state.task[:, 3:], sampled_matrix], axis=-1)
        t, task, position = jax.lax.cond(
            task_state.t > 1 - self.dt, 
            lambda x: (x - 1 + self.dt, candidate_matrix, new_position - offset),
            lambda x: (x + self.dt, task_state.task, new_position),
            task_state.t
        )
   
        new_task_state = task_state.replace(
            deviation=deviation,
            task=task,
            t=t,
            z=jnp.concatenate([
                deviation,
                jnp.reshape(task, (-1,)),
                jnp.array([t]),
            ]),
            key=key,
            position=position,
        )
        
        return new_task_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def reset(
        self,
        obs: jnp.ndarray, 
        key: RNGKey,
        ) -> GPTaskState:

        velocity = obs[13: 15]
        velocity = jnp.reshape(
            velocity,
            shape=(-1, 1),
            ) # (2, 1)
        
        key, subkey = jax.random.split(key)
        deviation = jax.random.normal(subkey, shape=(2,)) * 0.5
        key, subkey = jax.random.split(key)
        sampled_task = jax.random.multivariate_normal(
            subkey,
            self.prior_mu * velocity, # (2, 10)
            self.prior_cov, # (1, 10, 10)
        ) # (2, 10)

        new_task_state = GPTaskState(
            deviation=deviation,
            task=sampled_task,
            t=0.0,
            z=jnp.concatenate([
                deviation,
                jnp.reshape(sampled_task, (-1,)),
                jnp.zeros((1,)),
            ]),
            key=key,
            r=1.0,
            position=deviation,
        )
        
        return new_task_state




class CircularEncoder(TaskEncoder):
    def __init__(
        self, 
        v_min=1.0,
        v_max=2.5,
        v_noise=0.5,
        a_mean=0.5236,
        dt=0.05
        ):

        self.a_mean = a_mean
        self.v_min = v_min
        self.v_max = v_max
        self.v_noise = v_noise
        self.dt = dt


    @property
    def z_dim(self) -> int:
        return 5



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def step(
        self, 
        task_state: TaskState, 
        displacement: jnp.ndarray,
        ) -> TaskState:

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
        
        new_task_state = task_state.replace(
            deviation=deviation,
            task=new_task,
            z=jnp.concatenate([
                deviation,
                new_task,
            ]),
        )
        
        return new_task_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def reset(
        self,
        obs: jnp.ndarray, 
        key: RNGKey,
        ) -> TaskState:

    
        current_velocity = obs[13:15]
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
            key=key,
            r=1.0,
        )
        
        return new_task_state




class LineEncoder(TaskEncoder):
    def __init__(
        self, 
        v_min=1.0,
        v_max=2.5,
        dt=0.05,
        ):

        self.v_min = v_min
        self.v_max = v_max
        self.dt = dt


    @property
    def z_dim(self) -> int:
        return 4



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def step(
        self, 
        task_state: TaskState, 
        displacement: jnp.ndarray,
        ) -> TaskState:

        deviation = task_state.deviation + displacement - task_state.task * self.dt
        
        new_task_state = task_state.replace(
            deviation=deviation,
            z=jnp.concatenate([
                deviation,
                task_state.task,
            ]),
        )
        
        return new_task_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def reset(
        self,
        obs: jnp.ndarray, 
        key: RNGKey,
        ) -> TaskState:

        current_velocity = obs[13:15]
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
            key=key,
            # r=jnp.clip(speed, min=1.0),
            r=1.0
        )
        
        return new_task_state




class MaternTaskState(PyTreeNode):
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



class State_info(PyTreeNode):
    current_velocity: jax.Array # (2,)
    current_position: jax.Array # (2,)
    


class FiniteMaternWrapper:
    def __init__(
        self, 
        horizon: float = 4,
        way_points: int = 8,
        var: float = 2.25,
        l: float = 1,
        v_noise_var: float = 0.0625,
        dt: float = 0.05,):
        
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
        prior_cov = prior_cov * var + np.eye(way_points * 2 + 1) * 1e-8

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

    

    def _init_task_state(self, state_info: State_info, key: jax.Array) -> PyTreeNode:
        """initialize task state"""

        # sample way_points data
        mean_ys = self.posterior_mu_T * state_info.current_velocity[:, None] # (2, 2 * waypoints + 1)
        key, subkey = jax.random.split(key)
        ys = mean_ys + jax.random.normal(subkey, (2, 2 * self.way_points + 1)) @ self.posterior_L.T # (2, 2 * way_points + 1)

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
            position_offset=-state_info.current_position,
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
    


    def step(
        self, 
        z_state: MaternTaskState, 
    ) -> Tuple:
        """return next state, reward, done, truncation"""

        # check shift
        current_t = z_state.t + self.dt
        need_shift = current_t > self.period_t

        shifted_data = (
            current_t - self.period_t, 
            z_state.next_normalized_ds, 
            z_state.next_task_v,
            jnp.concatenate([z_state.task_s[:, 1:], jnp.zeros((2, 1))], axis=-1), 
            jnp.concatenate([z_state.next_normalized_ds[:, 1:], jnp.zeros((2, 1))], axis=-1), 
            jnp.concatenate([z_state.next_task_v[:, 1:], jnp.zeros((2, 1))], axis=-1),
        )
        regular_data = (
            current_t, 
            z_state.normalized_ds, 
            z_state.task_v,
            z_state.task_s, 
            z_state.next_normalized_ds, 
            z_state.next_task_v,
        )
        (current_t, normalized_ds, task_v, task_s, next_normalized_ds, next_task_v) = jax.lax.cond(
            need_shift,
            lambda x: shifted_data,
            lambda x: regular_data,
            None,
        ) # apply changes

        coefs = jnp.concatenate([task_s[:, :2], task_v[:, :2]], axis=-1) @ self.Sigma22_inv_AT # (2, 4)
        target_position = jnp.sum(coefs * self._base_fn(current_t), axis=-1) # (2,)
        deviation = target_position


        # check task finish
        new_remaining_t = z_state.remaining_t - self.dt
        deviation, new_remaining_t, normalized_t = jax.lax.cond(
            new_remaining_t < 0,
            lambda x: (jnp.zeros_like(deviation), -1.0, -1.0),
            lambda x: (deviation, new_remaining_t, 2*new_remaining_t/self.horizon - 1),
            None,
        )

        next_task_state = z_state.replace(
            normalized_ds=normalized_ds,
            next_normalized_ds=next_normalized_ds,
            task_s=task_s,
            task_v=task_v,
            next_task_v=next_task_v,
            remaining_t=new_remaining_t,
            t=current_t,
        )

        return next_task_state, target_position, normalized_t
    



