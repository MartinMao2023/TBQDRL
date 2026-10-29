"""Constrained NSF followed by RealNVP affine couplings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .coupling_layers import AffineCoupling, ConstrainedSplineCoupling
from .flow import FlowConfig as SplineConfig, _even_coupling_shift, _rotated_indices
from .real_nvp import RealNVPConfig
from .safe_flow import _output_coordinate_orders

Array = jax.Array
NF_ARCHITECTURE = "combined_nsf_real_nvp_v2"


@dataclass(frozen=True)
class CombinedFlowConfig:
    dimension: int
    num_spline_layers: int = 4
    num_affine_layers: int = 4
    hidden_features: tuple[int, ...] = (128, 128)
    num_bins: int = 8
    tail_bound: float = 3.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    width_bound: float = 1.5
    slope_bound: float = 1.5
    derivative_bound: float = 1.0
    log_scale_bound: float = 1.0

    def __post_init__(self) -> None:
        # Use the same validation as the standalone implementations.
        SplineConfig(
            dimension=self.dimension, num_layers=self.num_spline_layers,
            hidden_features=self.hidden_features, num_bins=self.num_bins,
            tail_bound=self.tail_bound, min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height, min_derivative=self.min_derivative,
            width_bound=self.width_bound, slope_bound=self.slope_bound,
            derivative_bound=self.derivative_bound,
        )
        RealNVPConfig(
            dimension=self.dimension, num_layers=self.num_affine_layers,
            hidden_features=self.hidden_features, log_scale_bound=self.log_scale_bound,
        )


class CombinedFlow(nn.Module):
    """Standard normal -> spline stack -> affine coupling stack.

    The spline stack retains its learned pre-spline biases and optional
    permutations, restoring coordinate order before the affine couplings.
    Boundary losses cover only spline inputs, averaged over spline layers.
    Neither stack contains an additional output affine. Log probabilities
    omit 2*pi by default.
    """

    config: CombinedFlowConfig
    permute: bool = True
    spline_dtype: Any | None = None
    conditioner_dtype: Any | None = None

    def setup(self) -> None:
        cfg = self.config
        self.spline_layers = tuple(
            ConstrainedSplineCoupling(
                mask=tuple((coordinate + index) % 2 == 0
                           for coordinate in range(cfg.dimension)),
                permutation=_rotated_indices(
                    cfg.dimension,
                    _even_coupling_shift(index, cfg.dimension, cfg.num_spline_layers)
                    if self.permute else 0,
                ),
                hidden_features=cfg.hidden_features, num_bins=cfg.num_bins,
                tail_bound=cfg.tail_bound, min_bin_width=cfg.min_bin_width,
                min_bin_height=cfg.min_bin_height, min_derivative=cfg.min_derivative,
                width_bound=cfg.width_bound, slope_bound=cfg.slope_bound,
                derivative_bound=cfg.derivative_bound,
                spline_dtype=self.spline_dtype, conditioner_dtype=self.conditioner_dtype,
                name=f"spline_coupling_{index}",
            )
            for index in range(cfg.num_spline_layers)
        )
        self.spline_output_order, self.spline_restore_order = _output_coordinate_orders(
            self.spline_layers, cfg.dimension,
        )
        self.affine_layers = tuple(
            AffineCoupling(
                mask=tuple((coordinate + index) % 2 == 0
                           for coordinate in range(cfg.dimension)),
                hidden_features=cfg.hidden_features, log_scale_bound=cfg.log_scale_bound,
                dtype=self.conditioner_dtype, name=f"affine_coupling_{index}",
            )
            for index in range(cfg.num_affine_layers)
        )
        self.pre_spline_bias = self.param(
            "pre_spline_bias", nn.initializers.zeros_init(),
            (cfg.num_spline_layers, cfg.dimension),
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(f"Expected final dimension {self.config.dimension}, got {inputs.shape}")

    def _forward(self, base_samples: Array, *, return_log_det: bool,
                 return_boundary_loss: bool) -> tuple[Array, Array, Array]:
        self._check_shape(base_samples)
        outputs = base_samples
        log_det = jnp.zeros(base_samples.shape[:-1], dtype=base_samples.dtype)
        boundary_loss = jnp.zeros_like(log_det)
        for index, layer in enumerate(self.spline_layers):
            outputs = outputs + self.pre_spline_bias[index]
            if return_log_det or return_boundary_loss:
                outputs, layer_log_det, layer_boundary = layer.forward_with_log_det(outputs)
                if return_log_det:
                    log_det = log_det + layer_log_det
                if return_boundary_loss:
                    boundary_loss = boundary_loss + layer_boundary
            else:
                outputs = layer(outputs)
        outputs = outputs[..., self.spline_restore_order]
        for layer in self.affine_layers:
            if return_log_det:
                outputs, layer_log_det = layer.forward_with_log_det(outputs)
                log_det = log_det + layer_log_det
            else:
                outputs = layer(outputs)
        return outputs, log_det, boundary_loss / self.config.num_spline_layers

    def forward(self, base_samples: Array, *, return_boundary_loss: bool = False):
        outputs, _, boundary_loss = self._forward(
            base_samples, return_log_det=False, return_boundary_loss=return_boundary_loss,
        )
        return (outputs, boundary_loss) if return_boundary_loss else outputs

    def forward_with_log_det(self, base_samples: Array, *, return_boundary_loss: bool = False):
        outputs, log_det, boundary_loss = self._forward(
            base_samples, return_log_det=True, return_boundary_loss=return_boundary_loss,
        )
        return (outputs, log_det, boundary_loss) if return_boundary_loss else (outputs, log_det)

    def inverse(self, samples: Array, *, return_boundary_loss: bool = False):
        self._check_shape(samples)
        outputs = samples
        log_det = jnp.zeros(samples.shape[:-1], dtype=samples.dtype)
        for layer in reversed(self.affine_layers):
            outputs, layer_log_det = layer.inverse(outputs)
            log_det = log_det + layer_log_det
        outputs = outputs[..., self.spline_output_order]
        boundary_loss = jnp.zeros_like(log_det)
        for index in reversed(range(self.config.num_spline_layers)):
            layer = self.spline_layers[index]
            if return_boundary_loss:
                outputs, layer_log_det, layer_boundary = layer.inverse(
                    outputs, return_boundary_loss=True,
                )
                boundary_loss = boundary_loss + layer_boundary
            else:
                outputs, layer_log_det = layer.inverse(outputs)
            outputs = outputs - self.pre_spline_bias[index]
            log_det = log_det + layer_log_det
        if return_boundary_loss:
            return outputs, log_det, boundary_loss / self.config.num_spline_layers
        return outputs, log_det

    def base_log_prob(self, base_samples: Array, *, include_constant: bool = False) -> Array:
        self._check_shape(base_samples)
        normalizer = self.config.dimension * math.log(2.0 * math.pi) if include_constant else 0.0
        return -0.5 * (jnp.sum(jnp.square(base_samples), axis=-1) + normalizer)

    def log_prob(self, samples: Array, *, return_boundary_loss: bool = False,
                 include_constant: bool = False):
        if return_boundary_loss:
            base_samples, log_det, boundary_loss = self.inverse(samples, return_boundary_loss=True)
            result = self.base_log_prob(base_samples, include_constant=include_constant) + log_det
            return result, boundary_loss
        base_samples, log_det = self.inverse(samples)
        return self.base_log_prob(base_samples, include_constant=include_constant) + log_det

    def __call__(self, samples: Array, *, return_boundary_loss: bool = False,
                 include_constant: bool = False):
        return self.log_prob(samples, return_boundary_loss=return_boundary_loss,
                             include_constant=include_constant)

    def sample(self, key: Array, num_samples: int, *, return_boundary_loss: bool = False):
        base_samples = jax.random.normal(key, (num_samples, self.config.dimension))
        return self.forward(base_samples, return_boundary_loss=return_boundary_loss)


def init_flow(key: Array, config: CombinedFlowConfig, *, dtype: Any = jnp.float32,
              permute: bool = True):
    """Initialize the spline and affine coupling stacks."""
    flow = CombinedFlow(config, permute=permute)
    return flow, flow.init(key, jnp.zeros((1, config.dimension), dtype=dtype))


FlowConfig = CombinedFlowConfig
NormalizingFlow = CombinedFlow

__all__ = ["CombinedFlowConfig", "CombinedFlow", "FlowConfig", "NormalizingFlow",
           "NF_ARCHITECTURE", "init_flow"]
