"""Physical coordinates for finite integrated-Matern trajectories."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .tools import IntegrateMatern


class MaternTrajectoryEncoding:
    """Encode ``[x_1, ..., x_n, v_0, ..., v_n]`` using physical path MSE.

    The initial position is zero. Each spatial axis uses the same transform.
    For column vectors, ``encoded = L.T @ raw``, where ``L @ L.T = A.T @ A``
    and A decodes positions at 128 equally spaced times, divided by sqrt(128).
    Summing squared encoding differences over both axes therefore gives the
    mean squared Euclidean path distance, averaged over time.

    Beyond the horizon, retain the existing wrapper's padding convention: one
    bridge from the final position/velocity to the same position at rest, then
    a constant position. Padding is used for future lookahead only.
    """

    def __init__(
        self, way_points: int, period_t: float, l: float = 1.0,
        metric_samples: int = 128,
    ):
        if way_points < 1 or period_t <= 0 or l <= 0:
            raise ValueError("way_points, period_t, and l must be positive")
        if metric_samples <= 2 * way_points + 1:
            raise ValueError(
                "metric_samples must exceed the encoding dimension; "
                "the fixed initial position contributes a zero row"
            )
        self.way_points = way_points
        self.period_t = period_t
        self.horizon = way_points * period_t
        self.dimension = 2 * way_points + 1
        self.metric_samples = metric_samples
        self.kernel = IntegrateMatern(l)
        waypoint_times = np.arange(way_points + 1) * period_t
        position_times = waypoint_times[1:]
        position_velocity = self.kernel.xv_cross_covariance(
            position_times, waypoint_times,
        )
        self.raw_covariance = np.block([
            [self.kernel.kernel(position_times, position_times), position_velocity],
            [position_velocity.T,
             self.kernel.derivative_kernel(waypoint_times, waypoint_times)],
        ])

        # Covariance of [x(0), v(0), x(T), v(T)]. A unit-variance initial
        # position makes this invertible; conditioning fixes its actual value.
        t = np.asarray([period_t], dtype=np.float64)
        zero = np.asarray([0.0], dtype=np.float64)
        position_variance = self.kernel.kernel(t, t)[0, 0]
        cross_start = self.kernel.xv_cross_covariance(t, zero)[0, 0]
        cross_end = self.kernel.xv_cross_covariance(t, t)[0, 0]
        velocity_cross = self.kernel.derivative_kernel(zero, t)[0, 0]
        self.boundary_covariance = np.asarray([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, cross_start, velocity_cross],
            [1.0, cross_start, 1.0 + position_variance, cross_end],
            [0.0, velocity_cross, cross_end, 1.0],
        ])

        self.metric_times = np.linspace(0.0, self.horizon, metric_samples)
        self.position_projection = self.projection(self.metric_times)
        self.A = self.position_projection / np.sqrt(metric_samples)
        self.metric_matrix = self.A.T @ self.A
        self.L = np.linalg.cholesky(self.metric_matrix)

    def projection(self, times, *, derivative: bool = False):
        """Return a float64 matrix mapping raw coordinates to x(t) or v(t)."""
        times = np.asarray(times, dtype=np.float64)
        if np.any(times < 0):
            raise ValueError("trajectory times must be nonnegative")
        shape = times.shape
        times = times.reshape(-1)
        periods = times / self.period_t
        # Snap arithmetic roundoff at waypoint boundaries before flooring.
        nearest = np.rint(periods)
        periods = np.where(np.abs(periods - nearest) < 1e-10, nearest, periods)
        segments = np.floor(periods).astype(np.int64)
        local_t = (periods - segments) * self.period_t
        n = self.way_points

        # Select [x_i, v_i, x_{i+1}, v_{i+1}] for each segment. The appended
        # segment ends at x_n with zero velocity; later segments are stationary.
        basis = np.eye(self.dimension, dtype=np.float64)
        positions = np.concatenate([np.zeros((1, self.dimension)), basis[:n]])
        velocities = basis[n:]
        start = np.minimum(segments, n)
        end = np.minimum(segments + 1, n)
        boundary = np.stack([
            positions[start],
            np.where((segments <= n)[:, None], velocities[start], 0.0),
            positions[end],
            np.where((segments < n)[:, None], velocities[end], 0.0),
        ], axis=1)

        zero = np.asarray([0.0])
        terminal = np.asarray([self.period_t])
        if derivative:
            cross = np.column_stack([
                np.zeros_like(local_t),
                self.kernel.derivative_kernel(local_t, zero)[:, 0],
                self.kernel.xv_cross_covariance(terminal, local_t)[0],
                self.kernel.derivative_kernel(local_t, terminal)[:, 0],
            ])
        else:
            cross = np.column_stack([
                np.ones_like(local_t),
                self.kernel.xv_cross_covariance(local_t, zero)[:, 0],
                1.0 + self.kernel.kernel(local_t, terminal)[:, 0],
                self.kernel.xv_cross_covariance(local_t, terminal)[:, 0],
            ])
        weights = np.linalg.solve(self.boundary_covariance, cross.T).T
        projection = np.einsum("ti,tij->tj", weights, boundary)
        return projection.reshape(shape + (self.dimension,))

    def encode(self, raw):
        """Transform row vectors ending in [positions, velocities]."""
        return jnp.asarray(raw) @ jnp.asarray(self.L)

    def decode(self, encoded):
        """Recover raw coordinates without explicitly inverting L."""
        encoded = jnp.asarray(encoded)
        return solve_triangular(
            jnp.asarray(self.L).T,
            encoded.reshape((-1, self.dimension)).T,
            lower=False,
        ).T.reshape(encoded.shape)

    def rollout_tables(self, dt: float, steps: int):
        """Build linear maps for all future encodings and desired positions.

        At step k, evaluate x/v at k*dt + j*T, subtract x(k*dt) from future
        positions, drop the first (zero) position, and transform with L.
        Matrix construction happens once, outside the JAX rollout.
        """
        times = np.arange(steps + 1, dtype=np.float64) * dt
        lookahead = times[:, None] + np.arange(self.way_points + 1) * self.period_t
        positions = self.projection(lookahead)
        velocities = self.projection(lookahead, derivative=True)
        relative_positions = positions[:, 1:] - positions[:, :1]
        raw_future = np.concatenate([relative_positions, velocities], axis=1)
        encoded_future = np.einsum("tij,ik->tjk", raw_future, self.L)
        return encoded_future, positions[:, 0]
