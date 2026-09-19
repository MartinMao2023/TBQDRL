"""A coupling rational-quadratic neural spline flow implemented in JAX/Flax."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class FlowConfig:
    """Configuration for :class:`NormalizingFlow`."""

    dimension: int
    num_layers: int = 8
    hidden_features: tuple[int, ...] = (128, 128)
    num_bins: int = 8
    tail_bound: float = 3.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.tail_bound <= 0:
            raise ValueError("tail_bound must be positive")
        if self.num_bins * self.min_bin_width >= 2 * self.tail_bound:
            raise ValueError("min_bin_width is too large for the spline interval")
        if self.num_bins * self.min_bin_height >= 2 * self.tail_bound:
            raise ValueError("min_bin_height is too large for the spline interval")


def _gather(values: Array, indices: Array) -> Array:
    return jnp.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def _rational_quadratic_spline(
    inputs: Array,
    parameters: Array,
    *,
    inverse: bool,
    num_bins: int,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[Array, Array]:
    """Apply elementwise monotonic RQ splines with identity linear tails."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    spline_inputs = jnp.clip(inputs, -tail_bound, tail_bound)

    raw_widths = parameters[..., :num_bins]
    raw_heights = parameters[..., num_bins : 2 * num_bins]
    raw_derivatives = parameters[..., 2 * num_bins :]

    interval = 2.0 * tail_bound
    widths = min_bin_width + (
        interval - min_bin_width * num_bins
    ) * jax.nn.softmax(raw_widths, axis=-1)
    heights = min_bin_height + (
        interval - min_bin_height * num_bins
    ) * jax.nn.softmax(raw_heights, axis=-1)

    derivative_offset = math.log(math.expm1(1.0 - min_derivative))
    internal_derivatives = min_derivative + jax.nn.softplus(
        raw_derivatives + derivative_offset
    )
    boundary = jnp.ones_like(internal_derivatives[..., :1])
    derivatives = jnp.concatenate(
        [boundary, internal_derivatives, boundary], axis=-1
    )

    left = jnp.full_like(widths[..., :1], -tail_bound)
    x_knots = jnp.concatenate([left, left + jnp.cumsum(widths, axis=-1)], axis=-1)
    y_knots = jnp.concatenate(
        [left, left + jnp.cumsum(heights, axis=-1)], axis=-1
    )

    knots = y_knots if inverse else x_knots
    bin_index = jnp.sum(spline_inputs[..., None] >= knots[..., 1:], axis=-1)
    bin_index = jnp.clip(bin_index, 0, num_bins - 1)

    x_left = _gather(x_knots[..., :-1], bin_index)
    y_left = _gather(y_knots[..., :-1], bin_index)
    bin_width = _gather(widths, bin_index)
    bin_height = _gather(heights, bin_index)
    derivative_left = _gather(derivatives[..., :-1], bin_index)
    derivative_right = _gather(derivatives[..., 1:], bin_index)
    slope = bin_height / bin_width

    if inverse:
        y_minus_left = spline_inputs - y_left
        common = derivative_left + derivative_right - 2.0 * slope
        a = y_minus_left * common + bin_height * (slope - derivative_left)
        b = bin_height * derivative_left - y_minus_left * common
        c = -slope * y_minus_left
        discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
        theta = (2.0 * c) / (-b - jnp.sqrt(discriminant))
        theta = jnp.clip(theta, 0.0, 1.0)
    else:
        theta = (spline_inputs - x_left) / bin_width

    one_minus_theta = 1.0 - theta
    theta_product = theta * one_minus_theta
    denominator = slope + (
        derivative_left + derivative_right - 2.0 * slope
    ) * theta_product

    if inverse:
        outputs = x_left + theta * bin_width
    else:
        numerator = bin_height * (
            slope * theta * theta + derivative_left * theta_product
        )
        outputs = y_left + numerator / denominator

    derivative_numerator = slope * slope * (
        derivative_right * theta * theta
        + 2.0 * slope * theta_product
        + derivative_left * one_minus_theta * one_minus_theta
    )
    log_det = jnp.log(derivative_numerator) - 2.0 * jnp.log(denominator)
    if inverse:
        log_det = -log_det

    outputs = jnp.where(inside, outputs, inputs)
    log_det = jnp.where(inside, log_det, 0.0)
    return outputs, log_det


class _Conditioner(nn.Module):
    hidden_features: tuple[int, ...]
    output_size: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for width in self.hidden_features:
            x = nn.Dense(width)(x)
            x = nn.selu(x)
        return nn.Dense(
            self.output_size,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros_init(),
        )(x)


class _SplineCoupling(nn.Module):
    mask: tuple[bool, ...]
    permutation: tuple[int, ...]
    hidden_features: tuple[int, ...]
    num_bins: int
    tail_bound: float
    min_bin_width: float
    min_bin_height: float
    min_derivative: float

    def setup(self) -> None:
        dimension = len(self.mask)
        self.identity_indices = jnp.asarray(
            [i for i, fixed in enumerate(self.mask) if fixed]
        )
        self.transform_indices = jnp.asarray(
            [i for i, fixed in enumerate(self.mask) if not fixed]
        )
        self.permutation_array = jnp.asarray(self.permutation)
        inverse_permutation = [0] * dimension
        for destination, source in enumerate(self.permutation):
            inverse_permutation[source] = destination
        self.inverse_permutation_array = jnp.asarray(inverse_permutation)

        parameters_per_dimension = 3 * self.num_bins - 1
        self.conditioner = _Conditioner(
            hidden_features=self.hidden_features,
            output_size=len(self.transform_indices) * parameters_per_dimension,
        )

    def _couple(self, inputs: Array, *, inverse: bool) -> tuple[Array, Array]:
        identity = inputs[..., self.identity_indices]
        transformed = inputs[..., self.transform_indices]
        parameters = self.conditioner(identity)
        parameters = parameters.reshape(
            transformed.shape + (3 * self.num_bins - 1,)
        )
        transformed, elementwise_log_det = _rational_quadratic_spline(
            transformed,
            parameters,
            inverse=inverse,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        outputs = inputs.at[..., self.transform_indices].set(transformed)
        return outputs, jnp.sum(elementwise_log_det, axis=-1)

    def __call__(self, inputs: Array) -> tuple[Array, Array]:
        permuted = inputs[..., self.permutation_array]
        return self._couple(permuted, inverse=False)

    def inverse(self, inputs: Array) -> tuple[Array, Array]:
        outputs, log_det = self._couple(inputs, inverse=True)
        return outputs[..., self.inverse_permutation_array], log_det


class NormalizingFlow(nn.Module):
    """RQ-spline coupling flow with a standard multivariate Gaussian base."""

    config: FlowConfig

    def setup(self) -> None:
        dimension = self.config.dimension
        layers = []
        for layer_index in range(self.config.num_layers):
            mask = tuple(
                (coordinate + layer_index) % 2 == 0
                for coordinate in range(dimension)
            )
            shift = layer_index % dimension
            permutation = tuple(
                list(range(dimension))[shift:] + list(range(dimension))[:shift]
            )
            layers.append(
                _SplineCoupling(
                    mask=mask,
                    permutation=permutation,
                    hidden_features=self.config.hidden_features,
                    num_bins=self.config.num_bins,
                    tail_bound=self.config.tail_bound,
                    min_bin_width=self.config.min_bin_width,
                    min_bin_height=self.config.min_bin_height,
                    min_derivative=self.config.min_derivative,
                    name=f"coupling_{layer_index}",
                )
            )
        self.layers = tuple(layers)

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(
                f"expected final dimension {self.config.dimension}, "
                f"got shape {inputs.shape}"
            )

    def forward(self, base_samples: Array) -> tuple[Array, Array]:
        """Map base samples to data space and return the forward log-Jacobian."""
        self._check_shape(base_samples)
        outputs = base_samples
        log_det = jnp.zeros(base_samples.shape[:-1], dtype=base_samples.dtype)
        for layer in self.layers:
            outputs, layer_log_det = layer(outputs)
            log_det = log_det + layer_log_det
        return outputs, log_det

    def inverse(self, samples: Array) -> tuple[Array, Array]:
        """Map data samples to base space and return the inverse log-Jacobian."""
        self._check_shape(samples)
        outputs = samples
        log_det = jnp.zeros(samples.shape[:-1], dtype=samples.dtype)
        for layer in reversed(self.layers):
            outputs, layer_log_det = layer.inverse(outputs)
            log_det = log_det + layer_log_det
        return outputs, log_det

    def base_log_prob(self, base_samples: Array) -> Array:
        """Log-density under the standard multivariate Gaussian base."""
        self._check_shape(base_samples)
        normalizer = self.config.dimension * math.log(2.0 * math.pi)
        return -0.5 * (jnp.sum(base_samples * base_samples, axis=-1) + normalizer)

    def log_prob(self, samples: Array) -> Array:
        """Evaluate normalized log-density in data space."""
        base_samples, inverse_log_det = self.inverse(samples)
        return self.base_log_prob(base_samples) + inverse_log_det

    def __call__(self, samples: Array) -> Array:
        return self.log_prob(samples)

    def sample_and_log_prob(
        self, key: Array, num_samples: int
    ) -> tuple[Array, Array]:
        """Draw samples and return their normalized log-density."""
        base_samples = jax.random.normal(
            key, (num_samples, self.config.dimension)
        )
        samples, forward_log_det = self.forward(base_samples)
        return samples, self.base_log_prob(base_samples) - forward_log_det

    def sample(self, key: Array, num_samples: int) -> Array:
        """Draw samples from the flow."""
        samples, _ = self.sample_and_log_prob(key, num_samples)
        return samples


def init_flow(
    key: Array, config: FlowConfig, *, dtype: Any = jnp.float32
) -> tuple[NormalizingFlow, dict[str, Any]]:
    """Construct a flow and initialize its Flax variables."""
    flow = NormalizingFlow(config)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = flow.init(key, example)
    return flow, variables
