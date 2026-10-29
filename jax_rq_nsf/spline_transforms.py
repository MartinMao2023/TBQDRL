"""Elementwise rational-quadratic spline geometry and inverse solvers."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array


def _gather(values: Array, indices: Array) -> Array:
    return jnp.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def rational_quadratic_spline(
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
    """Apply S(x + offset), or S^-1(y) - offset, with raw parameters.

    ``parameters`` contains 3*num_bins raw conditioner outputs per
    coordinate: widths, heights, internal derivatives, and one input offset.
    The offset is tail_bound times its raw output, without tanh.
    Only the spline contributes to the elementwise log determinant.
    """
    if parameters.shape[-1] != 3 * num_bins:
        raise ValueError("Expected 3*num_bins spline parameters per coordinate")

    raw_widths = parameters[..., :num_bins]
    raw_heights = parameters[..., num_bins : 2 * num_bins]
    raw_derivatives = parameters[..., 2 * num_bins : -1]
    offset = tail_bound * parameters[..., -1]
    shifted_inputs = inputs if inverse else inputs + offset
    inside = (shifted_inputs >= -tail_bound) & (shifted_inputs <= tail_bound)
    spline_inputs = jnp.clip(shifted_inputs, -tail_bound, tail_bound)

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

    outputs = jnp.where(inside, outputs, shifted_inputs)
    if inverse:
        outputs = outputs - offset
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


def _normalize_quadratic_coefficients(
    a: Array,
    b: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """Normalize each quadratic by an exact, stopped-gradient ``2**k``.

    ``frexp`` expresses the largest coefficient magnitude as ``m * 2**e``
    with ``m`` in ``[0.5, 1)``.  Scaling all three coefficients by
    ``2**(1-e)`` therefore places their maximum magnitude in ``[1, 2)``.
    A common scaling does not change either quadratic root.  The integer shift
    is stopped so gradients only follow the normalized coefficients, not the
    discrete choice of normalization scale.
    """
    maximum_magnitude = jnp.maximum(
        jnp.maximum(jnp.abs(a), jnp.abs(b)),
        jnp.abs(c),
    )
    _, exponent = jnp.frexp(maximum_magnitude)
    binary_shift = jax.lax.stop_gradient(1 - exponent)
    return (
        jnp.ldexp(a, binary_shift),
        jnp.ldexp(b, binary_shift),
        jnp.ldexp(c, binary_shift),
    )


def _solve_quadratic_on_unit_interval(
    a: Array,
    b: Array,
    c: Array,
    discriminant: Array,
) -> Array:
    """Select the stable Kahan root lying in the spline interval.

    For ``a * theta**2 + b * theta + c = 0``, Kahan's construction computes
    the root with the larger-magnitude numerator first and obtains the other
    root from their product.  This avoids choosing one fixed rationalization,
    which is unstable for one sign of ``b``.  A monotonic spline inverse has a
    unique valid root in ``[0, 1]``; if roundoff places both candidates just
    inside that interval, their normalized-polynomial residual breaks the tie.
    """
    sqrt_discriminant = jnp.sqrt(discriminant)
    signed_sqrt = jnp.copysign(sqrt_discriminant, b)
    q = -0.5 * (b + signed_sqrt)

    is_linear = a == 0.0
    safe_a = jnp.where(is_linear, jnp.ones_like(a), a)
    safe_q = jnp.where(q == 0.0, jnp.ones_like(q), q)
    safe_b = jnp.where(b == 0.0, jnp.ones_like(b), b)

    roots = jnp.stack((q / safe_a, c / safe_q), axis=-1)
    clipped_roots = jnp.clip(roots, 0.0, 1.0)
    tolerance = 32.0 * jnp.finfo(a.dtype).eps
    valid = (
        jnp.isfinite(roots)
        & (roots >= -tolerance)
        & (roots <= 1.0 + tolerance)
    )
    residuals = jnp.abs(
        (a[..., None] * clipped_roots + b[..., None]) * clipped_roots
        + c[..., None]
    )
    valid_scores = jnp.where(valid, residuals, jnp.inf)
    valid_index = jnp.argmin(valid_scores, axis=-1)
    valid_root = jnp.take_along_axis(
        clipped_roots, valid_index[..., None], axis=-1
    )[..., 0]

    # This fallback is only expected when roundoff moves both roots outside
    # the interval.  It still chooses the endpoint-clipped candidate with the
    # smaller residual rather than relying on a particular coefficient sign.
    fallback_scores = jnp.where(jnp.isfinite(roots), residuals, jnp.inf)
    fallback_index = jnp.argmin(fallback_scores, axis=-1)
    fallback_root = jnp.take_along_axis(
        clipped_roots, fallback_index[..., None], axis=-1
    )[..., 0]
    quadratic_root = jnp.where(jnp.any(valid, axis=-1), valid_root, fallback_root)

    linear_root = jnp.clip(-c / safe_b, 0.0, 1.0)
    return jnp.where(is_linear, linear_root, quadratic_root)


def constrain_spline_parameters(
    parameters: Array,
    *,
    num_bins: int,
    tail_bound: float,
    width_bound: float = 1.5,
    slope_bound: float = 1.5,
    derivative_bound: float = 1.0,
) -> tuple[Array, Array, Array, Array]:
    """Convert tanh-bounded conditioner outputs to spline geometry and offset.

    The same bounded ``u`` logits define the widths and contribute to the
    heights.  Bounded ``v`` logits therefore describe relative bin slopes,
    rather than allowing width and height conditioning to become independent.
    Internal knot derivatives are the geometric mean of their neighboring bin
    slopes, multiplied by a factor in
    ``[exp(-derivative_bound), exp(derivative_bound)]``. Width logits and
    relative slope logits use ``width_bound * u`` and ``slope_bound * v``;
    the caller applies tanh once to all 3*num_bins conditioner outputs.
    Normalization still determines the actual bin
    widths and slopes. Bounds must be finite and nonnegative. Zero disables
    the corresponding modulation.

    Returns ``(widths, heights, derivatives, offset)`` with
    ``offset = tail_bound * parameters[..., -1]``. The first and last
    derivatives are exactly one so the spline joins its identity tails with a
    continuous derivative.
    """
    for name, bound in (("width_bound", width_bound), ("slope_bound", slope_bound),
                        ("derivative_bound", derivative_bound)):
        if not math.isfinite(bound) or bound < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if parameters.shape[-1] != 3 * num_bins:
        raise ValueError("Expected 3*num_bins bounded spline parameters per coordinate")
    raw_widths = parameters[..., :num_bins]
    raw_relative_slopes = parameters[..., num_bins : 2 * num_bins]
    raw_derivatives = parameters[..., 2 * num_bins : -1]
    offset = tail_bound * parameters[..., -1]

    width_logits = width_bound * raw_widths
    relative_log_slopes = slope_bound * raw_relative_slopes
    log_width_proportions = jax.nn.log_softmax(width_logits, axis=-1)
    log_height_proportions = jax.nn.log_softmax(
        width_logits + relative_log_slopes,
        axis=-1,
    )

    interval = 2.0 * tail_bound
    widths = interval * jnp.exp(log_width_proportions)
    heights = interval * jnp.exp(log_height_proportions)
    log_slopes = log_height_proportions - log_width_proportions

    internal_derivatives = jnp.exp(
        derivative_bound * raw_derivatives
        + 0.5 * (log_slopes[..., :-1] + log_slopes[..., 1:])
    )
    boundary = jnp.ones_like(internal_derivatives[..., :1])
    derivatives = jnp.concatenate(
        [boundary, internal_derivatives, boundary], axis=-1
    )
    return widths, heights, derivatives, offset


def constrained_rational_quadratic_spline(
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
    computation_dtype: Any | None = None,
    width_bound: float = 1.5,
    slope_bound: float = 1.5,
    derivative_bound: float = 1.0,
) -> Array | tuple[Array, Array]:
    """Apply a shifted bounded RQ spline with a normalized inverse solve.

    ``parameters`` must already be tanh-bounded. Forward maps S(x + offset);
    inverse maps S^-1(y) - offset, including outside the spline interval.
    ``min_bin_width``, ``min_bin_height``, and ``min_derivative`` are accepted
    for API compatibility with :mod:`jax_rq_nsf.flow`, but this bounded
    parameterization does not use additive floors.
    """
    del min_bin_width, min_bin_height, min_derivative
    if computation_dtype is not None:
        inputs = inputs.astype(computation_dtype)
        parameters = parameters.astype(computation_dtype)
    widths, heights, derivatives, offset = constrain_spline_parameters(
        parameters,
        num_bins=num_bins,
        tail_bound=tail_bound,
        width_bound=width_bound,
        slope_bound=slope_bound,
        derivative_bound=derivative_bound,
    )
    shifted_inputs = inputs if inverse else inputs + offset
    inside = (shifted_inputs >= -tail_bound) & (shifted_inputs <= tail_bound)
    spline_inputs = jnp.clip(shifted_inputs, -tail_bound, tail_bound)

    left = jnp.full_like(widths[..., :1], -tail_bound)
    x_knots = jnp.concatenate(
        [left, left + jnp.cumsum(widths, axis=-1)], axis=-1
    )
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
        a, b, c = _normalize_quadratic_coefficients(a, b, c)
        discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
        theta = _solve_quadratic_on_unit_interval(a, b, c, discriminant)
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

    outputs = jnp.where(inside, outputs, shifted_inputs)
    if inverse:
        outputs = outputs - offset
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
