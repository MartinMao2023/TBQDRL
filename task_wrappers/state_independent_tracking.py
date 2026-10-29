"""Tracking with physical trajectory coordinates and precomputed task inputs."""

from typing import Tuple

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from flax.struct import PyTreeNode

from data_struct.qd_transitions import QDTransitionInfo
from data_struct.states import GeneralizedQDState as GeneralizedState
from jax_rq_nsf.safe_flow import NormalizingFlow
from task_wrappers.base import BaseFlowQDWrapper

from .trajectory_encoding import MaternTrajectoryEncoding


# def _tracking_reward(distance: jax.Array) -> jax.Array:
#     """Returns the smooth, bounded tracking penalty for an L2 distance."""

#     return jnp.clip(jnp.where(
#         distance <= 2.0,
#         1.0 - 0.125 * jnp.square(distance),
#         0.5 * jnp.exp(2.0 - distance),
#         ),
#         min=0.1)


def _tracking_reward(distance: jax.Array) -> jax.Array:
    """Returns the smooth, bounded tracking penalty for an L2 distance."""

    return jnp.clip(jnp.where(
        distance <= 1.0,
        1.0 - 0.5 * jnp.square(distance),
        0.5 * jnp.exp(2.0 - 2.0 * distance),
        ),
        min=0.05)


def _bounded_deviation(deviation: jax.Array) -> jax.Array:
    """Bounds each deviation component to (-4, 4) for the policy input."""

    return 4.0 * jnp.tanh(0.25 * deviation)


class MaternTaskState(PyTreeNode):
    """Cached task data, including time zero and the final horizon step.

    The NF samples two blocks of metric-transformed [positions, velocities].
    Policy z contains phase/progress, bounded tracking error, and the cached
    relative future encoding for the current step.
    """

    position_offset: jax.Array  # (2,)
    last_action: jax.Array  # (action_dim,)
    future_encodings: jax.Array  # (max_step_num + 1, 4 * way_points + 2)
    desired_positions: jax.Array  # (max_step_num + 1, 2), relative to task origin
    steps_taken: jax.Array
    z: jax.Array


class StateIndependentTrackingWrapper(BaseFlowQDWrapper):
    """Track NF samples expressed directly in physical metric coordinates."""

    def __init__(
        self,
        env: Env,
        flow: NormalizingFlow,
        way_points: int = 12,
        steps_per_way_point: int = 8,
        l: float = 1,
        dt: float = 0.05,
        metric_samples: int = 128,
    ):
        super().__init__(env, flow, way_points, steps_per_way_point, dt)
        self.z_dim = 4 * way_points + 7
        self.trajectory_encoding = MaternTrajectoryEncoding(
            way_points, self.period_t, l, metric_samples,
        )
        self.L = jnp.asarray(self.trajectory_encoding.L, dtype=jnp.float32)
        self.projection_matrix = jnp.asarray(
            self.trajectory_encoding.A, dtype=jnp.float32,
        )
        future_matrices, target_matrix = self.trajectory_encoding.rollout_tables(
            dt, self.max_step_num,
        )
        self._future_encoding_matrices = jnp.asarray(
            future_matrices, dtype=jnp.float32,
        )
        self._target_position_matrix = jnp.asarray(target_matrix, dtype=jnp.float32)
        steps = jnp.arange(self.max_step_num + 1, dtype=jnp.float32)
        phase = steps * (2.0 * jnp.pi / steps_per_way_point)
        self._time_features = jnp.stack([
            jnp.sin(phase),
            jnp.cos(phase),
            2.0 * steps / self.max_step_num - 1.0,
        ], axis=-1)

    @property
    def z_size(self):
        return self.z_dim

    def _get_position_from_envstate(self, env_state: State) -> jax.Array:
        """Return the agent's planar position."""

        return env_state.pipeline_state.x.pos[0, :2]

    def _organize_z_state(
        self, encoded_sequence: jax.Array, position_offset: jax.Array
    ) -> MaternTaskState:
        """Decode an NF sample and cache every future task input and target."""

        encoded_sequence = jnp.reshape(
            encoded_sequence, (2, 2 * self.way_points + 1),
        )
        raw_sequence = self.trajectory_encoding.decode(encoded_sequence)
        future_encodings = jnp.einsum(
            "ac,tcd->tad", raw_sequence, self._future_encoding_matrices,
        ).reshape((self.max_step_num + 1, -1))
        # Preserve the sampled coordinates exactly at time zero.
        future_encodings = future_encodings.at[0].set(encoded_sequence.reshape(-1))
        desired_positions = jnp.einsum(
            "ac,tc->ta", raw_sequence, self._target_position_matrix,
        )
        z = jnp.concatenate([
            self._time_features[0], jnp.zeros((2,)), future_encodings[0],
        ])
        return MaternTaskState(
            position_offset=position_offset,
            last_action=jnp.zeros((self.action_size,)),
            future_encodings=future_encodings,
            desired_positions=desired_positions,
            steps_taken=jnp.int32(0),
            z=z,
        )

    def sample_task(
        self, env_state: State, key: jax.Array, params: PyTreeNode
    ) -> MaternTaskState:
        """Sample metric coordinates from the NF, independently of agent state."""

        encoded_sequence = self.flow.apply(params, key, 1, method=self.flow.sample)
        position_offset = -self._get_position_from_envstate(env_state)
        return self._organize_z_state(encoded_sequence, position_offset)

    def resample_task_state(
        self, state: GeneralizedState, params: PyTreeNode
    ) -> GeneralizedState:
        """Replace both the active task and the task used on environment reset."""

        key, task_key = jax.random.split(state.key)
        task = self.sample_task(state.env_state, task_key, params)
        return state.replace(z_state=task, initial_z_state=task, key=key)

    def get_obs(
        self, state: GeneralizedState
    ) -> Tuple[jax.Array, jax.Array]:
        return state.env_state.obs, state.z_state.z

    def step(
        self,
        state: GeneralizedState,
        action: jax.Array,
    ) -> Tuple[GeneralizedState, QDTransitionInfo]:
        """Advance physics, then index cached task data at the resulting time."""

        next_env_state = self.env.step(state.env_state, action)
        truncation = next_env_state.info["truncation"]
        done = next_env_state.done - truncation
        has_reset = next_env_state.done > 0.5

        z_state = state.z_state
        current_step_num = jnp.where(has_reset, 0, z_state.steps_taken + 1)
        table_index = jnp.minimum(current_step_num, self.max_step_num)
        complete = current_step_num >= self.max_step_num
        completed = current_step_num > self.max_step_num

        current_position = self._get_position_from_envstate(next_env_state)
        position_offset = jnp.where(
            has_reset, -current_position, z_state.position_offset,
        )
        target_position = z_state.desired_positions[table_index]
        deviation = target_position - (current_position + position_offset)
        distance = jnp.linalg.norm(deviation)
        reward = jnp.where(
            completed | has_reset, 0.0, _tracking_reward(distance),
        )
        z = jnp.concatenate([
            self._time_features[table_index],
            _bounded_deviation(deviation),
            z_state.future_encodings[table_index],
        ])
        next_task_state = z_state.replace(
            position_offset=position_offset,
            last_action=jnp.where(has_reset, jnp.zeros_like(action), action),
            steps_taken=current_step_num,
            z=z,
        )
        fitness_reward = jnp.array([
            next_env_state.reward - next_env_state.metrics["x_velocity"] + 3.0,
        ])
        transition_info = QDTransitionInfo(
            reward=jnp.array([reward]),
            fitness_reward=fitness_reward,
            done=jnp.array([done > 0.5], dtype=jnp.float32),
            completion=jnp.array([complete], dtype=jnp.float32),
            truncation=jnp.array([truncation]),
            broken=jnp.array([0.0]),
        )
        return state.replace(
            env_state=next_env_state, z_state=next_task_state,
        ), transition_info

    def shift(self, state: GeneralizedState) -> GeneralizedState:
        """Satisfy the base wrapper API; cached tasks require no shift calls."""

        return state
