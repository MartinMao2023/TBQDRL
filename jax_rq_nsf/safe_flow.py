"""Numerically safer rational-quadratic spline flows.

This module shares the flow API of ``jax_rq_nsf.flow``. It bounds and couples
the spline-bin geometry produced by
the conditioner, and its inverse spline solver normalizes the quadratic
coefficients by a stopped-gradient integer power of two before evaluating the
discriminant.  Power-of-two scaling preserves the represented floating-point
significand and avoids the additional rounding introduced by an arbitrary
division.

NormalizingFlow uses spline couplings with bounded conditioner-dependent input
offsets, followed by a learned final affine.

Even cumulative rotations are enabled by default.  Output coordinates are
restored before the final affine transform, with the matching permutation in
the inverse.  This changes checkpoint semantics relative to earlier safe-flow
versions; use this architecture for newly trained flows.  Legacy task-sampling
checkpoints continue to use ``jax_rq_nsf.flow``.
"""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp

from .flow import (
    FlowConfig,
    _even_coupling_shift,
    _final_affine,
    _rotated_indices,
)

from .coupling_layers import ConstrainedSplineCoupling
from .spline_transforms import (
    _normalize_quadratic_coefficients, _solve_quadratic_on_unit_interval,
    constrain_spline_parameters as _constrain_spline_parameters,
    constrained_rational_quadratic_spline as _rational_quadratic_spline,
)

Array = jax.Array
NF_ARCHITECTURE = "safe_flow_even_permutation_restore_conditional_offset_v6"


def _output_coordinate_orders(layers, dimension):
    """Cumulative permutation and its inverse at the end of the stack."""
    order = list(range(dimension))
    for layer in layers:
        order = [order[source] for source in layer.permutation]
    inverse = [0] * dimension
    for destination, source in enumerate(order):
        inverse[source] = destination
    return jnp.asarray(order), jnp.asarray(inverse)


class InvertibleNSF(nn.Module):
    """Distribution-free RQ-spline mapping with a normalized inverse solve."""

    config: FlowConfig
    permute: bool = True
    spline_dtype: Any | None = None
    conditioner_dtype: Any | None = None

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
                ConstrainedSplineCoupling(
                    mask=mask,
                    permutation=permutation,
                    hidden_features=self.config.hidden_features,
                    num_bins=self.config.num_bins,
                    tail_bound=self.config.tail_bound,
                    min_bin_width=self.config.min_bin_width,
                    min_bin_height=self.config.min_bin_height,
                    min_derivative=self.config.min_derivative,
                    spline_dtype=self.spline_dtype,
                    conditioner_dtype=self.conditioner_dtype,
                    width_bound=self.config.width_bound,
                    slope_bound=self.config.slope_bound,
                    derivative_bound=self.config.derivative_bound,
                    name=f"coupling_{layer_index}",
                )
            )
        self.layers = tuple(layers)
        self.output_order, self.restore_order = _output_coordinate_orders(
            layers, dimension
        )
        self.initial_raw_scale = self.param(
            "initial_raw_scale", nn.initializers.zeros_init(), (dimension,)
        )
        self.initial_bias = self.param(
            "initial_bias", nn.initializers.zeros_init(), (dimension,)
        )
        self.final_raw_scale = self.param(
            "final_raw_scale", nn.initializers.zeros_init(), (dimension,)
        )
        self.final_bias = self.param(
            "final_bias", nn.initializers.zeros_init(), (dimension,)
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(
                f"expected final dimension {self.config.dimension}, "
                f"got shape {inputs.shape}"
            )

    def forward(self, inputs: Array) -> tuple[Array, Array, Array]:
        """Map intended targets to policy inputs."""
        self._check_shape(inputs)
        outputs, affine_log_det = _final_affine(
            inputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=True,
        )
        log_det = jnp.zeros(inputs.shape[:-1], dtype=inputs.dtype) + affine_log_det
        outputs = outputs[..., self.output_order]
        forward_boundary_loss = 0.0
        for layer in reversed(self.layers):
            outputs, layer_log_det, layer_boundary_loss = layer.inverse(
                outputs, return_boundary_loss=True
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
        return (
            outputs,
            log_det + initial_log_det,
            forward_boundary_loss / self.config.num_layers,
        )

    def reverse(self, inputs: Array) -> tuple[Array, Array, Array]:
        """Map policy inputs to target encodings."""
        self._check_shape(inputs)
        outputs, initial_log_det = _final_affine(
            inputs,
            self.initial_raw_scale,
            self.initial_bias,
            inverse=True,
        )
        log_det = jnp.zeros(inputs.shape[:-1], dtype=inputs.dtype) + initial_log_det
        reverse_boundary_loss = 0.0
        for layer in self.layers:
            outputs, layer_log_det, layer_boundary_loss = layer.forward_with_log_det(
                outputs
            )
            log_det = log_det + layer_log_det
            reverse_boundary_loss = reverse_boundary_loss + layer_boundary_loss
        outputs = outputs[..., self.restore_order]
        outputs, affine_log_det = _final_affine(
            outputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=False,
            return_log_det=True,
        )
        return (
            outputs,
            log_det + affine_log_det,
            reverse_boundary_loss / self.config.num_layers,
        )

    def inverse(self, inputs: Array) -> tuple[Array, Array, Array]:
        return self.reverse(inputs)

    def __call__(self, inputs: Array) -> tuple[Array, Array, Array]:
        return self.forward(inputs)


class NormalizingFlow(nn.Module):
    """RQ-spline flow followed by a learned diagonal affine."""

    config: FlowConfig
    permute: bool = True
    spline_dtype: Any | None = None
    conditioner_dtype: Any | None = None

    def setup(self) -> None:
        dimension = self.config.dimension
        layers = []
        for layer_index in range(self.config.num_layers):
            mask = tuple(
                (coordinate + layer_index) % 2 == 0
                for coordinate in range(dimension)
            )
            shift = (
                _even_coupling_shift(layer_index, dimension, self.config.num_layers)
                if self.permute else 0
            )
            permutation = _rotated_indices(dimension, shift)
            layers.append(
                ConstrainedSplineCoupling(
                    mask=mask,
                    permutation=permutation,
                    hidden_features=self.config.hidden_features,
                    num_bins=self.config.num_bins,
                    tail_bound=self.config.tail_bound,
                    min_bin_width=self.config.min_bin_width,
                    min_bin_height=self.config.min_bin_height,
                    min_derivative=self.config.min_derivative,
                    spline_dtype=self.spline_dtype,
                    conditioner_dtype=self.conditioner_dtype,
                    width_bound=self.config.width_bound,
                    slope_bound=self.config.slope_bound,
                    derivative_bound=self.config.derivative_bound,
                    name=f"coupling_{layer_index}",
                )
            )
        self.layers = tuple(layers)
        self.output_order, self.restore_order = _output_coordinate_orders(
            layers, dimension
        )
        self.final_raw_scale = self.param(
            "final_raw_scale", nn.initializers.zeros_init(), (dimension,)
        )
        self.final_bias = self.param(
            "final_bias", nn.initializers.zeros_init(), (dimension,)
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(
                f"expected final dimension {self.config.dimension}, "
                f"got shape {inputs.shape}"
            )

    def forward(
        self, base_samples: Array, *, return_boundary_loss: bool = False,
    ) -> Array | tuple[Array, Array]:
        """Transform base samples, optionally reporting internal tail excursions.

        The penalty sums squared excess beyond tail_bound over the transformed
        coordinates of each coupling, then averages over coupling layers. It
        has one value per sample and does not include the final affine output.
        """
        self._check_shape(base_samples)
        outputs = base_samples
        boundary_loss = jnp.zeros(base_samples.shape[:-1], dtype=base_samples.dtype)
        for layer in self.layers:
            if return_boundary_loss:
                outputs, _, layer_boundary = layer.forward_with_log_det(outputs)
                boundary_loss = boundary_loss + layer_boundary
            else:
                outputs = layer(outputs)
        outputs = outputs[..., self.restore_order]
        outputs = _final_affine(
            outputs,
            self.final_raw_scale,
            self.final_bias,
            inverse=False,
        )
        if return_boundary_loss:
            return outputs, boundary_loss / self.config.num_layers
        return outputs

    def inverse(
        self, samples: Array, *, return_boundary_loss: bool = False,
    ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        self._check_shape(samples)
        outputs, affine_log_det = _final_affine(
            samples,
            self.final_raw_scale,
            self.final_bias,
            inverse=True,
        )
        log_det = jnp.zeros(samples.shape[:-1], dtype=samples.dtype) + affine_log_det
        outputs = outputs[..., self.output_order]
        boundary_loss = jnp.zeros(samples.shape[:-1], dtype=samples.dtype)
        for layer in reversed(self.layers):
            if return_boundary_loss:
                outputs, layer_log_det, layer_boundary = layer.inverse(
                    outputs, return_boundary_loss=True,
                )
                boundary_loss = boundary_loss + layer_boundary
            else:
                outputs, layer_log_det = layer.inverse(outputs)
            log_det = log_det + layer_log_det
        if return_boundary_loss:
            return outputs, log_det, boundary_loss / self.config.num_layers
        return outputs, log_det

    def base_log_prob(self, base_samples: Array, *, include_constant: bool = False) -> Array:
        """Gaussian log-density, omitting d/2 * log(2*pi) by default."""
        self._check_shape(base_samples)
        normalizer = self.config.dimension * math.log(2.0 * math.pi) if include_constant else 0.0
        return -0.5 * (
            jnp.sum(base_samples * base_samples, axis=-1) + normalizer
        )

    def log_prob(
        self, samples: Array, *, return_boundary_loss: bool = False,
        include_constant: bool = False,
    ) -> Array | tuple[Array, Array]:
        """Log-density without the 2*pi constant unless explicitly requested."""
        if return_boundary_loss:
            base_samples, inverse_log_det, boundary_loss = self.inverse(
                samples, return_boundary_loss=True,
            )
            return (self.base_log_prob(base_samples, include_constant=include_constant)
                    + inverse_log_det, boundary_loss)
        base_samples, inverse_log_det = self.inverse(samples)
        return self.base_log_prob(base_samples, include_constant=include_constant) + inverse_log_det

    def __call__(
        self, samples: Array, *, return_boundary_loss: bool = False,
        include_constant: bool = False,
    ) -> Array | tuple[Array, Array]:
        return self.log_prob(samples, return_boundary_loss=return_boundary_loss,
                             include_constant=include_constant)

    def sample(
        self, key: Array, num_samples: int, *, return_boundary_loss: bool = False,
    ) -> Array | tuple[Array, Array]:
        base_samples = jax.random.normal(
            key, (num_samples, self.config.dimension)
        )
        return self.forward(base_samples, return_boundary_loss=return_boundary_loss)


def restore_flow_variables(template: Any, encoded: bytes) -> Any:
    """Restore bias-free NF variables with the same permutation convention."""
    state = serialization.msgpack_restore(encoded)
    parameters = state.get("params")
    if isinstance(parameters, dict) and any(
        name in parameters for name in ("pre_spline_bias", "inter_layer_bias")
    ):
        raise ValueError("Checkpoint contains removed spline biases; retrain the flow")
    return serialization.from_state_dict(template, state)


def init_invertible_nsf(
    key: Array,
    config: FlowConfig,
    *,
    permute: bool = True,
    dtype: Any = jnp.float32,
) -> tuple[InvertibleNSF, dict[str, Any]]:
    """Construct and initialize a safe distribution-free NSF mapping."""
    mapping = InvertibleNSF(config, permute=permute)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = mapping.init(key, example)
    return mapping, variables


def init_flow(
    key: Array, config: FlowConfig, *, dtype: Any = jnp.float32,
    permute: bool = True,
) -> tuple[NormalizingFlow, dict[str, Any]]:
    """Construct and initialize a safe normalizing flow."""
    flow = NormalizingFlow(config, permute=permute)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = flow.init(key, example)
    return flow, variables


__all__ = [
    "FlowConfig",
    "InvertibleNSF",
    "NormalizingFlow",
    "NF_ARCHITECTURE",
    "init_flow",
    "init_invertible_nsf",
    "restore_flow_variables",
]
