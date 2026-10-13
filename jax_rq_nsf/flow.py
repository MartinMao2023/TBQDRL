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


def _final_affine(
    inputs: Array,
    raw_scale: Array,
    bias: Array,
    *,
    inverse: bool,
    return_log_det: bool = False,
) -> Array | tuple[Array, Array]:
    """Apply an elementwise affine transform with stable positive scales."""
    stable_scale = jnp.maximum(0.5, 1.0 + raw_scale)
    stable_log_scale = jnp.clip(raw_scale, -5.0, 5.0)

    if inverse:
        inv_scale = jnp.where(
            raw_scale < 0.0,
            jnp.exp(-stable_log_scale),
            1 / stable_scale,
        )
        log_scale = jnp.where(
            raw_scale < 0.0,
            stable_log_scale,
            jnp.log(stable_scale),
        )
        outputs = (inputs - bias) * inv_scale
        return outputs, -jnp.sum(log_scale)
    else:
        scale = jnp.where(
            raw_scale < 0.0,
            jnp.exp(stable_log_scale),
            stable_scale,
        )
        outputs = inputs * scale + bias
        if return_log_det:
            log_scale = jnp.where(
                raw_scale < 0.0,
                stable_log_scale,
                jnp.log(stable_scale),
            )
            return outputs, jnp.sum(log_scale)
        return outputs


def _rational_quadratic_spline(
    inputs: Array,
    parameters: Array,
    *,
    inverse: bool,
    return_log_det: bool = False,
    num_bins: int,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> Array | tuple[Array, Array]:
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

    outputs = jnp.where(inside, outputs, inputs)
    if not inverse and not return_log_det:
        return outputs

    derivative_numerator = slope * slope * (
        derivative_right * theta * theta
        + 2.0 * slope * theta_product
        + derivative_left * one_minus_theta * one_minus_theta
    )
    log_det = jnp.log(derivative_numerator) - 2.0 * jnp.log(denominator)
    if inverse:
        log_det = -log_det

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

    def _couple(
        self,
        inputs: Array,
        *,
        inverse: bool,
        return_log_det: bool = False,
    ) -> Array | tuple[Array, Array]:
        identity = inputs[..., self.identity_indices]
        transformed = inputs[..., self.transform_indices]
        parameters = self.conditioner(identity)
        parameters = parameters.reshape(
            transformed.shape + (3 * self.num_bins - 1,)
        )
        spline_result = _rational_quadratic_spline(
            transformed,
            parameters,
            inverse=inverse,
            return_log_det=return_log_det,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        if inverse or return_log_det:
            transformed, elementwise_log_det = spline_result
        else:
            transformed = spline_result
        outputs = inputs.at[..., self.transform_indices].set(transformed)
        if inverse or return_log_det:
            return outputs, jnp.sum(elementwise_log_det, axis=-1)
        return outputs

    def __call__(self, inputs: Array) -> Array:
        permuted = inputs[..., self.permutation_array]
        return self._couple(permuted, inverse=False)

    def forward_with_log_det(self, inputs: Array) -> tuple[Array, Array]:
        permuted = inputs[..., self.permutation_array]
        return self._couple(
            permuted,
            inverse=False,
            return_log_det=True,
        )

    def inverse(self, inputs: Array) -> tuple[Array, Array]:
        outputs, log_det = self._couple(inputs, inverse=True)
        return outputs[..., self.inverse_permutation_array], log_det


class InvertibleNSF(nn.Module):
    """Distribution-free RQ-spline mapping between equal-sized encodings."""

    config: FlowConfig
    permute: bool = False

    def setup(self) -> None:
        dimension = self.config.dimension
        layers = []
        for layer_index in range(self.config.num_layers):
            mask = tuple(
                (coordinate + layer_index) % 2 == 0
                for coordinate in range(dimension)
            )
            if self.permute:
                shift = layer_index % dimension
                permutation = tuple(
                    list(range(dimension))[shift:]
                    + list(range(dimension))[:shift]
                )
            else:
                permutation = tuple(range(dimension))
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
        self.initial_raw_scale = self.param(
            "initial_raw_scale",
            nn.initializers.zeros_init(),
            (dimension,),
        )
        self.initial_bias = self.param(
            "initial_bias",
            nn.initializers.zeros_init(),
            (dimension,),
        )
        self.final_raw_scale = self.param(
            "final_raw_scale",
            nn.initializers.zeros_init(),
            (dimension,),
        )
        self.final_bias = self.param(
            "final_bias",
            nn.initializers.zeros_init(),
            (dimension,),
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(
                f"expected final dimension {self.config.dimension}, "
                f"got shape {inputs.shape}"
            )

    def forward(self, inputs: Array) -> tuple[Array, Array]:
        """Map intended target encodings ``z'`` to policy encodings ``z``."""
        self._check_shape(inputs)
        outputs, affine_log_det = _final_affine(
            inputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=True,
        )
        log_det = (
            jnp.zeros(inputs.shape[:-1], dtype=inputs.dtype)
            + affine_log_det
        )
        for layer in reversed(self.layers):
            outputs, layer_log_det = layer.inverse(outputs)
            log_det = log_det + layer_log_det
        outputs, initial_log_det = _final_affine(
            outputs,
            self.initial_raw_scale,
            self.initial_bias,
            inverse=True,
        )
        log_det = log_det + initial_log_det
        return outputs, log_det

    def reverse(self, inputs: Array) -> tuple[Array, Array]:
        """Map policy encodings ``z`` to target encodings ``z'``."""
        self._check_shape(inputs)
        outputs, initial_log_det = _final_affine(
            inputs,
            self.initial_raw_scale,
            self.initial_bias,
            inverse=False,
            return_log_det=True,
        )
        log_det = (
            jnp.zeros(inputs.shape[:-1], dtype=inputs.dtype)
            + initial_log_det
        )
        for layer in self.layers:
            outputs, layer_log_det = layer.forward_with_log_det(outputs)
            log_det = log_det + layer_log_det
        outputs, affine_log_det = _final_affine(
            outputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=False,
            return_log_det=True,
        )
        return outputs, log_det + affine_log_det

    def inverse(self, inputs: Array) -> tuple[Array, Array]:
        """Inverse of :meth:`forward`, mapping policy encodings to targets."""
        return self.reverse(inputs)

    def __call__(self, inputs: Array) -> tuple[Array, Array]:
        return self.forward(inputs)


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
        self.final_raw_scale = self.param(
            "final_raw_scale",
            nn.initializers.zeros_init(),
            (dimension,),
        )
        self.final_bias = self.param(
            "final_bias",
            nn.initializers.zeros_init(),
            (dimension,),
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(
                f"expected final dimension {self.config.dimension}, "
                f"got shape {inputs.shape}"
            )

    def forward(self, base_samples: Array) -> Array:
        """Map base samples to data space without computing Jacobians."""
        self._check_shape(base_samples)
        outputs = base_samples
        for layer in self.layers:
            outputs = layer(outputs)
        return _final_affine(
            outputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=False,
        )

    def inverse(self, samples: Array) -> tuple[Array, Array]:
        """Map data samples to base space and return the inverse log-Jacobian."""
        self._check_shape(samples)
        outputs, affine_log_det = _final_affine(
            samples,
            self.final_raw_scale,
            self.final_bias,
            inverse=True,
        )
        log_det = (
            jnp.zeros(samples.shape[:-1], dtype=samples.dtype)
            + affine_log_det
        )
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

    def sample(self, key: Array, num_samples: int) -> Array:
        """Draw samples without computing their density."""
        base_samples = jax.random.normal(
            key, (num_samples, self.config.dimension)
        )
        return self.forward(base_samples)


def init_invertible_nsf(
    key: Array,
    config: FlowConfig,
    *,
    permute: bool = False,
    dtype: Any = jnp.float32,
) -> tuple[InvertibleNSF, dict[str, Any]]:
    """Construct and initialize a distribution-free NSF mapping.

    ``permute=False`` keeps every coupling in the input coordinate order, so
    initialization stays close to the identity. ``permute=True`` applies the
    same cyclic coordinate shifts used by :class:`NormalizingFlow`.
    """
    mapping = InvertibleNSF(config, permute=permute)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = mapping.init(key, example)
    return mapping, variables


def init_flow(
    key: Array, config: FlowConfig, *, dtype: Any = jnp.float32
) -> tuple[NormalizingFlow, dict[str, Any]]:
    """Construct a flow and initialize its Flax variables."""
    flow = NormalizingFlow(config)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = flow.init(key, example)
    return flow, variables
