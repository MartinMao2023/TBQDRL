"""RealNVP with bounded affine couplings and the existing final affine.

The coupling stack starts at identity. The final affine retains the safe
flow's scale parameterization: expansion is unbounded, and contraction has
an exp(-5) floor. This module implements a standalone affine flow; its reusable
couplings can also be placed after an NSF stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from .coupling_layers import AffineCoupling
from .flow import _final_affine

Array = jax.Array
NF_ARCHITECTURE = "real_nvp_affine_coupling_v1"


@dataclass(frozen=True)
class RealNVPConfig:
    dimension: int
    num_layers: int = 4
    hidden_features: tuple[int, ...] = (128, 128)
    log_scale_bound: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if any(width < 1 for width in self.hidden_features):
            raise ValueError("hidden feature sizes must be positive")
        if not math.isfinite(self.log_scale_bound) or self.log_scale_bound <= 0:
            raise ValueError("log_scale_bound must be finite and positive")


class RealNVP(nn.Module):
    """Standard-normal base, alternating affine couplings, and final affine.

    ``forward``, ``inverse``, ``sample`` and ``log_prob`` match safe_flow's
    return conventions. Optional boundary losses are zero: affine layers
    have no spline boundaries. Coordinates retain their original ordering.
    """

    config: RealNVPConfig

    def setup(self) -> None:
        self.layers = tuple(
            AffineCoupling(
                mask=tuple((coordinate + index) % 2 == 0
                           for coordinate in range(self.config.dimension)),
                hidden_features=self.config.hidden_features,
                log_scale_bound=self.config.log_scale_bound,
                name=f"coupling_{index}",
            )
            for index in range(self.config.num_layers)
        )
        self.final_raw_scale = self.param(
            "final_raw_scale", nn.initializers.zeros_init(), (self.config.dimension,),
        )
        self.final_bias = self.param(
            "final_bias", nn.initializers.zeros_init(), (self.config.dimension,),
        )

    def _check_shape(self, inputs: Array) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.config.dimension:
            raise ValueError(f"Expected final dimension {self.config.dimension}, got {inputs.shape}")

    def forward_with_log_det(self, base_samples: Array) -> tuple[Array, Array]:
        self._check_shape(base_samples)
        outputs = base_samples
        log_det = jnp.zeros(base_samples.shape[:-1], dtype=base_samples.dtype)
        for layer in self.layers:
            outputs, layer_log_det = layer.forward_with_log_det(outputs)
            log_det = log_det + layer_log_det
        outputs, affine_log_det = _final_affine(
            outputs, self.final_raw_scale, self.final_bias,
            inverse=False, return_log_det=True,
        )
        return outputs, log_det + affine_log_det

    def forward(self, base_samples: Array, *, return_boundary_loss: bool = False):
        self._check_shape(base_samples)
        outputs = base_samples
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = _final_affine(outputs, self.final_raw_scale, self.final_bias, inverse=False)
        if return_boundary_loss:
            return outputs, jnp.zeros(outputs.shape[:-1], dtype=outputs.dtype)
        return outputs

    def inverse(self, samples: Array, *, return_boundary_loss: bool = False):
        self._check_shape(samples)
        outputs, affine_log_det = _final_affine(
            samples, self.final_raw_scale, self.final_bias, inverse=True,
        )
        log_det = jnp.zeros(samples.shape[:-1], dtype=samples.dtype) + affine_log_det
        for layer in reversed(self.layers):
            outputs, layer_log_det = layer.inverse(outputs)
            log_det = log_det + layer_log_det
        if return_boundary_loss:
            return outputs, log_det, jnp.zeros_like(log_det)
        return outputs, log_det

    def base_log_prob(self, base_samples: Array, *, include_constant: bool = False) -> Array:
        """Gaussian log-density, omitting d/2 * log(2*pi) by default."""
        self._check_shape(base_samples)
        return -0.5 * (
            jnp.sum(jnp.square(base_samples), axis=-1)
            + (self.config.dimension * math.log(2.0 * math.pi) if include_constant else 0.0)
        )

    def log_prob(self, samples: Array, *, return_boundary_loss: bool = False,
                 include_constant: bool = False):
        base_samples, log_det = self.inverse(samples)
        result = self.base_log_prob(base_samples, include_constant=include_constant) + log_det
        if return_boundary_loss:
            return result, jnp.zeros_like(result)
        return result

    def __call__(self, samples: Array, *, return_boundary_loss: bool = False,
                 include_constant: bool = False):
        return self.log_prob(samples, return_boundary_loss=return_boundary_loss,
                             include_constant=include_constant)

    def sample(self, key: Array, num_samples: int, *, return_boundary_loss: bool = False):
        base_samples = jax.random.normal(key, (num_samples, self.config.dimension))
        return self.forward(base_samples, return_boundary_loss=return_boundary_loss)


def init_flow(key: Array, config: RealNVPConfig, *, dtype: Any = jnp.float32):
    """Construct an identity-initialized RealNVP using the safe flow API."""
    flow = RealNVP(config)
    return flow, flow.init(key, jnp.zeros((1, config.dimension), dtype=dtype))


# Allow the same module-level imports used for the spline implementations.
FlowConfig = RealNVPConfig
NormalizingFlow = RealNVP

__all__ = ["AffineCoupling", "RealNVPConfig", "RealNVP", "FlowConfig",
           "NormalizingFlow", "NF_ARCHITECTURE", "init_flow"]
