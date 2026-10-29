"""A coupling rational-quadratic neural spline flow implemented in JAX/Flax."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .coupling_layers import SplineCoupling
from .spline_transforms import (
    _gather, rational_quadratic_spline as _rational_quadratic_spline,
)

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
    # Geometry bounds used by safe_flow's ConstrainedSplineCoupling.
    width_bound: float = 1.5
    slope_bound: float = 1.5
    derivative_bound: float = 1.0

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
        for name in ("width_bound", "slope_bound", "derivative_bound"):
            bound = getattr(self, name)
            if not math.isfinite(bound) or bound < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


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


# def _final_affine(
#     inputs: Array,
#     raw_scale: Array,
#     bias: Array,
#     *,
#     inverse: bool,
#     return_log_det: bool = False,
# ) -> Array | tuple[Array, Array]:
#     """Apply an elementwise affine transform with stable positive scales."""

#     if inverse:
#         outputs = inputs - bias
#         return outputs, 0.0
#     else:
#         outputs = inputs + bias
#         if return_log_det:
#             return outputs, 0.0
#         return outputs



def _even_coupling_shift(
    layer_index: int, dimension: int, num_layers: int
) -> int:
    """Return the left-rotation applied by one coupling layer.

    Earlier rotations remain in the activation, so a constant step of
    ``dimension // num_layers`` makes the cumulative alignments
    ``0, step, 2 * step, ...``.
    """
    if layer_index == 0:
        return 0
    return dimension // num_layers


def _rotated_indices(dimension: int, shift: int) -> tuple[int, ...]:
    coordinates = list(range(dimension))
    return tuple(coordinates[shift:] + coordinates[:shift])


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
                shift = _even_coupling_shift(
                    layer_index, dimension, self.config.num_layers
                )
                permutation = _rotated_indices(dimension, shift)
            else:
                permutation = tuple(range(dimension))
            layers.append(
                SplineCoupling(
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

    def forward(self, inputs: Array) -> tuple[Array, Array, Array]:
        """Map intended targets ``z'`` to policy inputs ``z``.

        Returns the mapped encoding, its log absolute determinant, and
        ``forward_boundary_loss``.
        """
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
        forward_boundary_loss = 0.0
        for layer in reversed(self.layers):
            outputs, layer_log_det, layer_boundary_loss = layer.inverse(
                outputs,
                return_boundary_loss=True,
            )
            log_det = log_det + layer_log_det
            forward_boundary_loss = forward_boundary_loss + layer_boundary_loss
        outputs, initial_log_det = _final_affine(
            outputs,
            self.initial_raw_scale,
            self.initial_bias,
            inverse=False,
            return_log_det=True,
        )
        log_det = log_det + initial_log_det
        return outputs, log_det, forward_boundary_loss / self.config.num_layers

    def reverse(self, inputs: Array) -> tuple[Array, Array, Array]:
        """Map policy inputs ``z`` to target encodings ``z'``.

        Returns the mapped encoding, its log absolute determinant, and
        ``reverse_boundary_loss``.
        """
        self._check_shape(inputs)
        outputs, initial_log_det = _final_affine(
            inputs,
            self.initial_raw_scale,
            self.initial_bias,
            inverse=True,
        )
        log_det = (
            jnp.zeros(inputs.shape[:-1], dtype=inputs.dtype)
            + initial_log_det
        )
        reverse_boundary_loss = 0.0
        for layer in self.layers:
            (
                outputs,
                layer_log_det,
                layer_boundary_loss,
            ) = layer.forward_with_log_det(outputs)
            log_det = log_det + layer_log_det
            reverse_boundary_loss = reverse_boundary_loss + layer_boundary_loss
        outputs, affine_log_det = _final_affine(
            outputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=False,
            return_log_det=True,
        )
        return outputs, log_det + affine_log_det, reverse_boundary_loss / self.config.num_layers

    def inverse(self, inputs: Array) -> tuple[Array, Array, Array]:
        """Inverse of :meth:`forward`, mapping policy encodings to targets."""
        return self.reverse(inputs)

    def __call__(self, inputs: Array) -> tuple[Array, Array, Array]:
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
                SplineCoupling(
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

    def base_log_prob(self, base_samples: Array, *, include_constant: bool = False) -> Array:
        """Gaussian log-density, omitting d/2 * log(2*pi) by default."""
        self._check_shape(base_samples)
        normalizer = self.config.dimension * math.log(2.0 * math.pi) if include_constant else 0.0
        return -0.5 * (jnp.sum(base_samples * base_samples, axis=-1) + normalizer)

    def log_prob(self, samples: Array, *, include_constant: bool = False) -> Array:
        """Evaluate log-density; include_constant=True restores normalization."""
        base_samples, inverse_log_det = self.inverse(samples)
        return self.base_log_prob(base_samples, include_constant=include_constant) + inverse_log_det

    def __call__(self, samples: Array, *, include_constant: bool = False) -> Array:
        return self.log_prob(samples, include_constant=include_constant)

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
    initialization stays close to the identity. ``permute=True`` rotates each
    layer after the first by ``dimension // num_layers``, giving evenly spaced
    cumulative alignments.
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
