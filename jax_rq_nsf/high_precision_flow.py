"""Selectively high-precision bounded rational-quadratic spline flows.

By default, conditioner calculations and their checkpoint parameters remain in
float32, while conditioner outputs are promoted before spline states, knots,
rational-quadratic evaluations, inverse quadratic solves, and log determinants
are evaluated in float64.  Conditioner calculations can optionally be promoted
to float64 as well; their stored parameters remain float32.  The module has the
same parameter-tree layout as :mod:`safe_flow` and can load its checkpoints
without conversion.

JAX 64-bit support must be enabled by the calling program before these models
are initialized or JIT-compiled, for example with
``jax.config.update("jax_enable_x64", True)``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .flow import FlowConfig
from .safe_flow import (
    InvertibleNSF as _SafeInvertibleNSF,
    NormalizingFlow as _SafeNormalizingFlow,
)

Array = jax.Array


class InvertibleNSF(_SafeInvertibleNSF):
    """Safe INN with float64 spline arithmetic and float32 conditioners."""

    spline_dtype: Any | None = jnp.float64
    conditioner_dtype: Any | None = jnp.float32


class NormalizingFlow(_SafeNormalizingFlow):
    """Safe normalizing flow with selectively float64 spline arithmetic."""

    spline_dtype: Any | None = jnp.float64
    conditioner_dtype: Any | None = jnp.float32


def _require_x64() -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "high_precision_flow requires JAX 64-bit mode; call "
            "jax.config.update('jax_enable_x64', True) before initialization"
        )


def init_invertible_nsf(
    key: Array,
    config: FlowConfig,
    *,
    permute: bool = True,
    dtype: Any = jnp.float32,
    high_precision_conditioner: bool = False,
) -> tuple[InvertibleNSF, dict[str, Any]]:
    """Construct a checkpoint-compatible selectively high-precision INN."""
    _require_x64()
    conditioner_dtype = (
        jnp.float64 if high_precision_conditioner else jnp.float32
    )
    mapping = InvertibleNSF(
        config,
        permute=permute,
        conditioner_dtype=conditioner_dtype,
    )
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = mapping.init(key, example)
    return mapping, variables


def init_flow(
    key: Array,
    config: FlowConfig,
    *,
    dtype: Any = jnp.float32,
    high_precision_conditioner: bool = False,
) -> tuple[NormalizingFlow, dict[str, Any]]:
    """Construct a checkpoint-compatible selectively precise flow."""
    _require_x64()
    conditioner_dtype = (
        jnp.float64 if high_precision_conditioner else jnp.float32
    )
    flow = NormalizingFlow(config, conditioner_dtype=conditioner_dtype)
    example = jnp.zeros((1, config.dimension), dtype=dtype)
    variables = flow.init(key, example)
    return flow, variables


__all__ = [
    "FlowConfig",
    "InvertibleNSF",
    "NormalizingFlow",
    "init_flow",
    "init_invertible_nsf",
]
