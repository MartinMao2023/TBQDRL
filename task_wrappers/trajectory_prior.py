"""Trajectory prior covariances and the conditioned heuristic Gaussian target."""

import math

WAY_POINTS = 12
STEPS_PER_WAY_POINT = 8
VAR = 4
LENGTH_SCALE = 1.0
VELOCITY_NOISE_VAR = 0.25
DT = 0.05
METRIC_SAMPLES = 128
ENCODING_TYPE = "physical_positions_velocities_v1"
HEURISTIC_KERNELS = ("matern12", "matern32", "rbf")
HEURISTIC_LENGTH_SCALES = tuple(range(1, 9))


def _integrated_velocity_primitives(r, kernel, length_scale):
    """Return F(r), G(r), and unit-variance velocity covariance k(r).

    F is the odd first primitive of the even stationary kernel; G is its
    even second primitive, with F(0) = G(0) = 0. All arithmetic is float64.
    """
    import numpy as np

    distance = np.abs(r)
    if kernel == "matern12":
        u = distance / length_scale
        first = -length_scale * np.expm1(-u)
        # G(r) = l^2 * (u + exp(-u) - 1). Use its series near zero
        # to preserve the leading displacement variance for short times.
        small_u = np.minimum(u, 0.05)
        second_series = small_u**2 * np.polynomial.polynomial.polyval(
            small_u,
            [0.5, -1.0 / 6.0, 1.0 / 24.0, -1.0 / 120.0,
             1.0 / 720.0, -1.0 / 5040.0, 1.0 / 40320.0, -1.0 / 362880.0],
        )
        second = length_scale**2 * np.where(
            u <= 0.05, second_series, u + np.expm1(-u),
        )
        return np.sign(r) * first, second, np.exp(-u)

    if kernel == "matern32":
        # Standard Matern-3/2 convention: a = sqrt(3) / length_scale.
        a = math.sqrt(3.0) / length_scale
        u = a * distance
        decay = np.exp(-u)
        first = (-2.0 * np.expm1(-u) - u * decay) / a
        # The closed form loses precision near zero. Its Taylor expansion
        # starts at u^2/2 and has no cubic term; retain terms through u^9.
        small_u = np.minimum(u, 0.05)
        second_series = small_u**2 * np.polynomial.polynomial.polyval(
            small_u,
            [0.5, 0.0, -1.0 / 24.0, 1.0 / 60.0, -1.0 / 240.0,
             1.0 / 1260.0, -1.0 / 8064.0, 1.0 / 60480.0],
        )
        second = np.where(u <= 0.05, second_series,
                          3.0 * u + (u + 3.0) * np.expm1(-u)) / a**2
        return np.sign(r) * first, second, (1.0 + u) * decay

    if kernel == "rbf":
        from scipy.special import erf

        u = distance / length_scale
        first = length_scale * math.sqrt(math.pi / 2.0) * erf(u / math.sqrt(2.0))
        # expm1 avoids subtracting two almost equal exponentials for small u.
        second = distance * first + length_scale**2 * np.expm1(-0.5 * u**2)
        return np.sign(r) * first, second, np.exp(-0.5 * u**2)

    raise ValueError("kernel must be 'matern12', 'matern32', or 'rbf'")


def integrated_velocity_covariances(t1, t2, *, kernel="matern32",
                                     length_scale=LENGTH_SCALE, var=VAR):
    """Return pairwise (K_xx, K_xv, K_vv) arrays of shape (len(t1), len(t2)).

    x(t) = integral_0^t v(u) du, so x(0) is fixed at zero, while v(0) is
    unconditioned and has variance ``var``. Times are nonnegative scalars or
    one-dimensional arrays. ``K_xv[i, j] = Cov(x(t1[i]), v(t2[j]))``;
    its transpose is the corresponding velocity-position covariance block.

    ``matern12`` uses exp(-r/l), ``matern32`` uses
    (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l), and ``rbf`` uses exp(-r^2/(2*l^2)).
    Names are case-insensitive; ``matern05`` and ``matern0.5`` also select
    Matérn 0.5. These priors do not change the physical encoding.
    """
    import numpy as np

    kernel = kernel.lower()
    if kernel in ("matern05", "matern0.5"):
        kernel = "matern12"
    if kernel not in ("matern12", "matern32", "rbf"):
        raise ValueError("kernel must be 'matern12', 'matern32', or 'rbf'")
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    if not math.isfinite(var) or var < 0:
        raise ValueError("var must be finite and nonnegative")
    times = []
    for name, values in (("t1", t1), ("t2", t2)):
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} must contain finite nonnegative times in one dimension")
        times.append(values)
    t = times[0][:, None]
    s = times[1][None, :]
    _, g_t, _ = _integrated_velocity_primitives(t, kernel, length_scale)
    f_s, g_s, _ = _integrated_velocity_primitives(s, kernel, length_scale)
    f_delta, g_delta, k_delta = _integrated_velocity_primitives(t - s, kernel, length_scale)
    return var * (g_t + g_s - g_delta), var * (f_delta + f_s), var * k_delta


def trajectory_covariance(way_points=WAY_POINTS, period_t=STEPS_PER_WAY_POINT * DT,
                          *, kernel="matern32", length_scale=LENGTH_SCALE, var=VAR,
                          L=None):
    """Return one axis's unconditional trajectory covariance in float64.

    Coordinates are [x_1, ..., x_n, v_0, ..., v_n], with t_i = i * period_t.
    The matrix has shape (2*n+1, 2*n+1). Initial displacement is zero and
    excluded; initial velocity remains random, without observation conditioning.
    Kernel names are case-insensitive: ``matern12``, ``matern32``, or ``rbf``.

    If the existing encoding's ``L`` is supplied, return L.T @ K_raw @ L.
    Reuse the SAME L across kernel families and length scales. Stack calls
    with np.stack to obtain a covariance bank; independent x/y axes use two
    identical diagonal blocks. No jitter or covariance conditioning is added.
    """
    import numpy as np

    if (not isinstance(way_points, (int, np.integer))
            or isinstance(way_points, (bool, np.bool_)) or way_points < 1):
        raise ValueError("way_points must be a positive integer")
    if not math.isfinite(period_t) or period_t <= 0:
        raise ValueError("period_t must be finite and positive")
    times = np.arange(way_points + 1, dtype=np.float64) * period_t
    k_xx, k_xv, k_vv = integrated_velocity_covariances(
        times, times, kernel=kernel, length_scale=length_scale, var=var,
    )
    covariance = np.block([[k_xx[1:, 1:], k_xv[1:]], [k_xv[1:].T, k_vv]])
    if L is not None:
        L = np.asarray(L, dtype=np.float64)
        if L.shape != covariance.shape or not np.all(np.isfinite(L)):
            raise ValueError("L must be a finite square matrix matching the per-axis encoding")
        covariance = L.T @ covariance @ L
    # Remove only floating-point asymmetry, without changing the prior spectrum.
    return 0.5 * (covariance + covariance.T)






def make_heuristic_affine(encoding, *, kernels=HEURISTIC_KERNELS,
                          length_scales=HEURISTIC_LENGTH_SCALES, weights=None,
                          var=1.0, spatial_dimensions=2):
    """Build a fixed physical output transform from a bank of raw priors.

    All calculations and returned arrays use NumPy float64. Component order
    is kernel-major, then length-scale-major. ``weights`` is a nonnegative
    vector in that order; None gives equal probabilities (1/24 by default).
    Variance is 1.0 by default and initial velocity is unconditioned.

    For each axis, K_bar = sum(p_k * K_k), C = chol(K_bar), T = L.T @ C.
    Column samples obey z = T @ u; row samples obey z = u @ T.T. Independent
    axes use identical diagonal blocks. The inverse is calculated before any
    float32 conversion. No Cholesky of the projected covariance is needed.
    """
    import numpy as np

    kernels, length_scales = tuple(kernels), tuple(length_scales)
    if not kernels or not length_scales:
        raise ValueError("kernels and length_scales must be nonempty")
    if (not isinstance(spatial_dimensions, (int, np.integer))
            or isinstance(spatial_dimensions, (bool, np.bool_)) or spatial_dimensions < 1):
        raise ValueError("spatial_dimensions must be a positive integer")
    if not math.isfinite(var) or var <= 0:
        raise ValueError("var must be finite and positive for an invertible affine")
    covariances = np.stack([
        trajectory_covariance(encoding.way_points, encoding.period_t,
                              kernel=kernel, length_scale=scale, var=var)
        for kernel in kernels for scale in length_scales
    ])
    probabilities = (np.ones(len(covariances), dtype=np.float64) if weights is None
                     else np.asarray(weights, dtype=np.float64))
    if (probabilities.shape != (len(covariances),)
            or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.isfinite(probabilities.sum()) or probabilities.sum() <= 0):
        raise ValueError("weights must be a finite nonnegative vector with positive total mass")
    probabilities = probabilities / probabilities.sum()
    average = np.einsum("k,kij->ij", probabilities, covariances)
    average = 0.5 * (average + average.T)
    factor = np.linalg.cholesky(average)
    L = np.asarray(encoding.L, dtype=np.float64)
    if L.shape != average.shape or not np.all(np.isfinite(L)):
        raise ValueError("encoding.L must be finite and match the per-axis covariance")
    sign, log_det_L = np.linalg.slogdet(L)
    if sign == 0:
        raise ValueError("encoding.L must be invertible")
    axis_matrix = L.T @ factor
    axis_inverse = np.linalg.solve(factor, np.linalg.solve(L.T, np.eye(len(L))))
    log_abs_det = float(spatial_dimensions * (log_det_L + np.log(np.diag(factor)).sum()))
    return {
        "matrix": np.kron(np.eye(spatial_dimensions), axis_matrix),
        "inverse": np.kron(np.eye(spatial_dimensions), axis_inverse),
        "log_abs_det": log_abs_det,
        "entropy_offset": log_abs_det,
        "log_prob_offset": -log_abs_det,
        "covariances": covariances,
        "average_covariance": average,
        "raw_cholesky": factor,
        "probabilities": probabilities,
        "metadata": {
            "kernels": [kernel.lower() for kernel in kernels],
            "length_scales": [float(scale) for scale in length_scales],
            "variance": float(var), "probabilities": probabilities.tolist(),
            "component_order": "kernel-major, length-scale-minor",
            "spatial_dimensions": int(spatial_dimensions),
            "initial_velocity_conditioned": False,
            "matrix_convention": "column: z = T @ u; row: z = u @ T.T",
            "construction_dtype": "float64", "storage_dtype": "float32",
        },
    }


def make_conditioned_noise_bank(encoding, *, prior_velocity_variance=1.0,
                                initial_velocity_noise_variance=0.25,
                                physical_rms=0.05, kernels=HEURISTIC_KERNELS,
                                length_scales=HEURISTIC_LENGTH_SCALES):
    """Build separate encoded GP perturbation factors, one per prior component.

    Condition each raw prior on v(0) + measurement_noise = 0. The observation
    has noise variance ``initial_velocity_noise_variance``; its zero value
    leaves the conditional mean zero. For each component, factor K_cond=C@C.T
    in host float64, then T=L.T@C. Scale T so two independent planar axes
    have E[time-mean squared Euclidean displacement] = physical_rms**2.

    Return an axis-factor bank (components, coordinates, coordinates). One
    component is selected uniformly for each trajectory and shared by its x/y
    axes. Column noise is T@normal; row noise is normal@T.T. Components are
    never averaged. Smooth RBF priors can be numerically rank deficient, so
    use an eigendecomposition, clipping only negative roundoff eigenvalues.
    No diagonal jitter is added.
    """
    import numpy as np

    for name, value in (("prior_velocity_variance", prior_velocity_variance),
                        ("initial_velocity_noise_variance", initial_velocity_noise_variance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(physical_rms) or physical_rms < 0:
        raise ValueError("physical_rms must be finite and nonnegative")
    kernels, length_scales = tuple(kernels), tuple(length_scales)
    if not kernels or not length_scales:
        raise ValueError("kernels and length_scales must be nonempty")
    L = np.asarray(encoding.L, dtype=np.float64)
    dimension = 2 * encoding.way_points + 1
    if L.shape != (dimension, dimension) or not np.isfinite(L).all():
        raise ValueError("encoding.L must be finite and match the prior coordinates")
    factors, components = [], []
    for kernel in kernels:
        for length_scale in length_scales:
            raw = trajectory_covariance(
                encoding.way_points, encoding.period_t, kernel=kernel,
                length_scale=length_scale, var=prior_velocity_variance,
            )
            # v(0) follows the n positions in [p_1,...,p_n,v_0,...,v_n].
            cross = raw[:, encoding.way_points]
            conditional = raw - np.outer(cross, cross) / (
                prior_velocity_variance + initial_velocity_noise_variance
            )
            conditional = 0.5 * (conditional + conditional.T)
            eigenvalues, eigenvectors = np.linalg.eigh(conditional)
            tolerance = 100 * np.finfo(np.float64).eps * max(1.0, float(eigenvalues[-1]))
            if eigenvalues[0] < -tolerance:
                raise ValueError(f"Conditioned {kernel}/{length_scale} covariance is not positive semidefinite")
            raw_factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
            encoded_factor = L.T @ raw_factor
            unscaled_mse = 2.0 * np.sum(np.square(encoded_factor))
            if not np.isfinite(unscaled_mse) or unscaled_mse <= 0:
                raise ValueError("Conditioned noise must have positive physical variance")
            scale = physical_rms / np.sqrt(unscaled_mse)
            factors.append(encoded_factor * scale)
            components.append({"kernel": kernel.lower(), "length_scale": float(length_scale),
                               "physical_scale": float(scale),
                               "minimum_raw_eigenvalue": float(eigenvalues[0])})
    return np.stack(factors), {
        "kind": "conditioned_integrated_velocity_gp_mixture",
        "prior_velocity_variance": float(prior_velocity_variance),
        "initial_velocity_noise_variance": float(initial_velocity_noise_variance),
        "initial_velocity_observation": 0.0,
        "conditional_initial_velocity_variance_before_scaling": float(
            prior_velocity_variance * initial_velocity_noise_variance /
            (prior_velocity_variance + initial_velocity_noise_variance)),
        "physical_rms": float(physical_rms), "expected_physical_mse": float(physical_rms**2),
        "component_selection": "uniform_per_trajectory", "components": components,
        "probabilities": [1.0 / len(components)] * len(components),
        "component_order": "kernel-major, length-scale-minor", "shared_component_across_axes": True,
        "axes_independent_given_component": True, "spatial_dimensions": 2,
        "factorization": "raw_eigh_with_negative_roundoff_clipped_then_L_transpose",
        "matrix_convention": "column: delta_z = T @ normal; row: delta_z = normal @ T.T",
        "construction_dtype": "float64", "storage_dtype": "float32",
    }


def gaussian_moments(initial_velocity, *, way_points=WAY_POINTS,
                     steps_per_way_point=STEPS_PER_WAY_POINT, var=VAR,
                     length_scale=LENGTH_SCALE, velocity_noise_var=VELOCITY_NOISE_VAR,
                     dt=DT, metric_samples=METRIC_SAMPLES):
    """Return per-axis means and a shared Cholesky in physical coordinates.

    Keep matrix construction and factorization in host float64, including the
    small modes of the physical distance metric. Spatial axes are independent.
    """
    import numpy as np
    from task_wrappers.trajectory_encoding import MaternTrajectoryEncoding

    encoding = MaternTrajectoryEncoding(
        way_points, steps_per_way_point * dt, length_scale, metric_samples,
    )
    prior_covariance = var * encoding.raw_covariance
    cross_initial_velocity = prior_covariance[:, way_points]
    observation_variance = var + velocity_noise_var
    raw_mean = np.asarray(initial_velocity)[:, None] * (
        cross_initial_velocity / observation_variance
    )
    raw_covariance = prior_covariance - np.outer(
        cross_initial_velocity, cross_initial_velocity,
    ) / observation_variance
    mean = raw_mean @ encoding.L
    covariance = encoding.L.T @ raw_covariance @ encoding.L
    return mean, np.linalg.cholesky(covariance)
