"""Reusable affine and spline coupling layers for normalizing flows."""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .spline_transforms import (
    constrain_spline_parameters, constrained_rational_quadratic_spline,
    rational_quadratic_spline,
)

Array = jax.Array


class _AffineConditioner(nn.Module):
    hidden_features: tuple[int, ...]
    output_size: int
    dtype: Any | None = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        outputs = inputs
        for width in self.hidden_features:
            outputs = nn.selu(nn.Dense(width, dtype=self.dtype)(outputs))
        # Zero only the output layer: the coupling starts at exact identity
        # while its output weights can receive gradients immediately.
        return nn.Dense(
            self.output_size, dtype=self.dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(), name="output",
        )(outputs)


class AffineCoupling(nn.Module):
    """Keep masked coordinates fixed and transform the remaining coordinates.

    With ``a = inputs[..., mask]``, the other coordinates obey
    ``y_b = x_b * exp(log_scale_bound * tanh(s(a))) + t(a)``.
    Translation is unbounded. Log scales are bounded; the layer has no spline
    interval or identity tails. Both directions support arbitrary batch axes.
    """

    mask: tuple[bool, ...]
    hidden_features: tuple[int, ...] = (128, 128)
    log_scale_bound: float = 1.0
    dtype: Any | None = None

    def setup(self) -> None:
        if len(self.mask) < 2 or not any(self.mask) or all(self.mask):
            raise ValueError("mask must include both fixed and transformed coordinates")
        if not math.isfinite(self.log_scale_bound) or self.log_scale_bound <= 0:
            raise ValueError("log_scale_bound must be finite and positive")
        if any(width < 1 for width in self.hidden_features):
            raise ValueError("hidden feature sizes must be positive")
        self.fixed_indices = jnp.asarray([i for i, fixed in enumerate(self.mask) if fixed])
        transformed = [i for i, fixed in enumerate(self.mask) if not fixed]
        self.transform_indices = jnp.asarray(transformed)
        self.conditioner = _AffineConditioner(
            self.hidden_features, 2 * len(transformed), dtype=self.dtype,
        )

    def _transform(self, inputs: Array, *, inverse: bool) -> tuple[Array, Array]:
        if inputs.ndim < 1 or inputs.shape[-1] != len(self.mask):
            raise ValueError(f"Expected final dimension {len(self.mask)}, got {inputs.shape}")
        parameters = self.conditioner(inputs[..., self.fixed_indices])
        raw_scale, shift = jnp.split(parameters, 2, axis=-1)
        log_scale = self.log_scale_bound * jnp.tanh(raw_scale)
        transformed = inputs[..., self.transform_indices]
        if inverse:
            transformed = (transformed - shift) * jnp.exp(-log_scale)
            log_det = -jnp.sum(log_scale, axis=-1)
        else:
            transformed = transformed * jnp.exp(log_scale) + shift
            log_det = jnp.sum(log_scale, axis=-1)
        return inputs.at[..., self.transform_indices].set(transformed), log_det

    def forward(self, inputs: Array) -> Array:
        return self._transform(inputs, inverse=False)[0]

    def forward_with_log_det(self, inputs: Array) -> tuple[Array, Array]:
        return self._transform(inputs, inverse=False)

    def inverse(self, inputs: Array) -> tuple[Array, Array]:
        return self._transform(inputs, inverse=True)

    def __call__(self, inputs: Array) -> Array:
        return self.forward(inputs)


class _Conditioner(nn.Module):
    hidden_features: tuple[int, ...]
    output_size: int
    num_bins: int
    dtype: Any | None = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for width in self.hidden_features:
            x = nn.Dense(width, dtype=self.dtype)(x)
            x = nn.selu(x)

        def initialize_output_kernel(key, shape, dtype=jnp.float32):
            kernel = nn.initializers.normal(1e-3)(key, shape, dtype)
            # Each group's final output is its offset; start it at zero.
            stride = 3 * self.num_bins
            return kernel.at[..., stride - 1::stride].set(0.0)

        return nn.Dense(
            self.output_size,
            dtype=self.dtype,
            kernel_init=initialize_output_kernel,
            bias_init=nn.initializers.zeros_init(),
        )(x)


class SplineCoupling(nn.Module):
    """RQ-spline coupling with raw bin parameters and a conditional input shift."""

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

        parameters_per_dimension = 3 * self.num_bins
        self.conditioner = _Conditioner(
            hidden_features=self.hidden_features,
            output_size=len(self.transform_indices) * parameters_per_dimension,
            num_bins=self.num_bins,
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
            transformed.shape + (3 * self.num_bins,)
        )
        spline_result = rational_quadratic_spline(
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
        offset = self.tail_bound * parameters[..., -1]
        boundary_inputs = (
            transformed if inverse else inputs[..., self.transform_indices]
        ) + offset
        boundary_loss = jnp.mean(
            jnp.square(jnp.maximum(jnp.abs(boundary_inputs) - self.tail_bound, 0.0)),
            axis=-1,
        )
        outputs = inputs.at[..., self.transform_indices].set(transformed)
        if inverse or return_log_det:
            return (
                outputs,
                jnp.sum(elementwise_log_det, axis=-1),
                boundary_loss,
            )
        return outputs

    def __call__(self, inputs: Array) -> Array:
        permuted = inputs[..., self.permutation_array]
        return self._couple(permuted, inverse=False)

    def forward_with_log_det(
        self, inputs: Array
    ) -> tuple[Array, Array, Array]:
        permuted = inputs[..., self.permutation_array]
        return self._couple(
            permuted,
            inverse=False,
            return_log_det=True,
        )

    def inverse(
        self, inputs: Array, *, return_boundary_loss: bool = False
    ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        outputs, log_det, boundary_loss = self._couple(inputs, inverse=True)
        outputs = outputs[..., self.inverse_permutation_array]
        if return_boundary_loss:
            return outputs, log_det, boundary_loss
        return outputs, log_det


class ConstrainedSplineCoupling(nn.Module):
    """RQ-spline coupling with bounded geometry, input shift, and stable inverse."""

    mask: tuple[bool, ...]
    permutation: tuple[int, ...]
    hidden_features: tuple[int, ...]
    num_bins: int
    tail_bound: float
    min_bin_width: float
    min_bin_height: float
    min_derivative: float
    spline_dtype: Any | None = None
    conditioner_dtype: Any | None = None
    width_bound: float = 1.5
    slope_bound: float = 1.5
    derivative_bound: float = 1.0

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

        parameters_per_dimension = 3 * self.num_bins
        self.conditioner = _Conditioner(
            hidden_features=self.hidden_features,
            output_size=len(self.transform_indices) * parameters_per_dimension,
            num_bins=self.num_bins,
            dtype=self.conditioner_dtype,
        )

    def _couple(
        self,
        inputs: Array,
        *,
        inverse: bool,
        return_log_det: bool = False,
    ) -> Array | tuple[Array, Array, Array]:
        working_inputs = (
            inputs
            if self.spline_dtype is None
            else inputs.astype(self.spline_dtype)
        )
        conditioner_inputs = working_inputs[..., self.identity_indices]
        if self.conditioner_dtype is not None:
            conditioner_inputs = conditioner_inputs.astype(self.conditioner_dtype)
        transformed = working_inputs[..., self.transform_indices]
        parameters = jnp.tanh(self.conditioner(conditioner_inputs))
        parameters = parameters.reshape(
            transformed.shape + (3 * self.num_bins,)
        )
        spline_result = constrained_rational_quadratic_spline(
            transformed,
            parameters,
            inverse=inverse,
            return_log_det=return_log_det,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            computation_dtype=self.spline_dtype,
            width_bound=self.width_bound,
            slope_bound=self.slope_bound,
            derivative_bound=self.derivative_bound,
        )
        if inverse or return_log_det:
            transformed, elementwise_log_det = spline_result
        else:
            transformed = spline_result
        offset = self.tail_bound * parameters[..., -1]
        boundary_inputs = (
            transformed if inverse else working_inputs[..., self.transform_indices]
        ) + offset
        boundary_loss = jnp.sum(
            jnp.square(jnp.maximum(jnp.abs(boundary_inputs) - self.tail_bound, 0.0)),
            axis=-1,
        )
        outputs = working_inputs.at[..., self.transform_indices].set(transformed)
        if inverse or return_log_det:
            return outputs, jnp.sum(elementwise_log_det, axis=-1), boundary_loss
        return outputs

    def __call__(self, inputs: Array) -> Array:
        permuted = inputs[..., self.permutation_array]
        return self._couple(permuted, inverse=False)

    def forward_with_log_det(
        self, inputs: Array
    ) -> tuple[Array, Array, Array]:
        permuted = inputs[..., self.permutation_array]
        return self._couple(permuted, inverse=False, return_log_det=True)

    def inverse(
        self, inputs: Array, *, return_boundary_loss: bool = False
    ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        outputs, log_det, boundary_loss = self._couple(inputs, inverse=True)
        outputs = outputs[..., self.inverse_permutation_array]
        if return_boundary_loss:
            return outputs, log_det, boundary_loss
        return outputs, log_det


__all__ = ["AffineCoupling", "SplineCoupling", "ConstrainedSplineCoupling",
           "constrain_spline_parameters"]
