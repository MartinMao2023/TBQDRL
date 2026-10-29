"""Clone trajectory priors with safe NSF, learned affine, and fixed prior affine."""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import argparse
import math
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from brax import envs
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jax_rq_nsf.safe_flow import FlowConfig, NF_ARCHITECTURE, NormalizingFlow, init_flow
from jax_rq_nsf.fixed_affine_flow import (
    FixedAffineFlow, FixedAffineState, init_flow as init_fixed_flow,
)
from task_wrappers.trajectory_encoding import MaternTrajectoryEncoding
from task_wrappers.trajectory_prior import (
    WAY_POINTS, STEPS_PER_WAY_POINT, VAR, LENGTH_SCALE, VELOCITY_NOISE_VAR,
    DT, METRIC_SAMPLES, ENCODING_TYPE, gaussian_moments, make_heuristic_affine,
)


SEED = 0
NUM_STEPS = 10_000
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
MAX_GRADIENT_NORM = 5.0
BOUNDARY_LOSS_WEIGHT = 1.0
REPORT_EVERY = 100

OUTPUT_DIRECTORY = Path("output")
CHECKPOINT_PATH = OUTPUT_DIRECTORY / "trajectory_flow_variables.msgpack"
METADATA_PATH = OUTPUT_DIRECTORY / "trajectory_flow_metadata.json"
FLOW_ARCHITECTURE = NF_ARCHITECTURE
REGION_COVERAGE = 0.95

FLOW_CONFIG = FlowConfig(
    dimension=4 * WAY_POINTS + 2,
    num_layers=6,
    hidden_features=(128, 128),
    num_bins=8,
    tail_bound=4.0,
)


@dataclass(frozen=True)
class CloneModel:
    """Select another flow while sharing targets, optimization, and logging."""

    config: Any
    architecture: str
    initializer: Callable
    checkpoint_stem: Path
    has_spline_boundaries: bool = True
    fixed_affine_metadata: dict | None = None


def make_fixed_affine():
    """Build the shared prior affine T = L.T @ chol(K_average)."""
    encoding = MaternTrajectoryEncoding(
        WAY_POINTS, STEPS_PER_WAY_POINT * DT, LENGTH_SCALE, METRIC_SAMPLES,
    )
    # Equal weights: Matern 1/2, Matern 3/2, RBF, each at length scales 1..8.
    heuristic = make_heuristic_affine(encoding, var=4.0)
    affine = FixedAffineState.from_numpy(
        heuristic["matrix"], inverse_matrix=heuristic["inverse"],
        log_abs_det=heuristic["log_abs_det"],
    )
    return affine, heuristic["metadata"]


def make_fixed_affine_model():
    """Append the shared prior affine to the safe NSF core."""
    affine, prior_metadata = make_fixed_affine()

    def initialize(key, config):
        # Safe NSF already contains the learned final diagonal affine.
        return init_fixed_flow(key, NormalizingFlow(config), affine)

    return CloneModel(
        config=FLOW_CONFIG,
        architecture="safe_nsf_fixed_affine_v1",
        initializer=initialize,
        checkpoint_stem=OUTPUT_DIRECTORY / "trajectory_flow",
        fixed_affine_metadata={"core_architecture": NF_ARCHITECTURE,
                               "prior": prior_metadata},
    )


def make_target_distribution(
    key: jax.Array,
    *, include_constant: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return target moments and log normalizer, omitting 2*pi by default."""
    base_env = envs.create(
        env_name="ant",
        episode_length=4_096,
        backend="mjx",
        reset_noise_scale=0.0,
    )
    initial_env_state = base_env.reset(key)
    initial_velocity = np.asarray(jax.device_get(initial_env_state.obs[13:15]))
    mean, cholesky = gaussian_moments(
        initial_velocity, way_points=WAY_POINTS,
        steps_per_way_point=STEPS_PER_WAY_POINT, var=VAR,
        length_scale=LENGTH_SCALE, velocity_noise_var=VELOCITY_NOISE_VAR,
        dt=DT, metric_samples=METRIC_SAMPLES,
    )
    mean = jnp.asarray(mean, dtype=jnp.float32)
    # Factor and solve in host float64: the physical metric has small modes,
    # and rounding the Cholesky before solving unnecessarily loses precision.
    inverse_cholesky = jnp.asarray(np.linalg.solve(
        cholesky, np.eye(cholesky.shape[0]),
    ), dtype=jnp.float32)

    dimensions_per_axis = cholesky.shape[0]
    axis_log_normalizer = (
        (dimensions_per_axis * math.log(2.0 * math.pi) if include_constant else 0.0)
        + 2.0 * np.sum(np.log(np.diag(cholesky)))
    )
    log_normalizer = jnp.asarray(2.0 * axis_log_normalizer, dtype=jnp.float32)
    return mean, inverse_cholesky, log_normalizer


def target_log_prob(
    samples: jax.Array,
    mean: jax.Array,
    inverse_cholesky: jax.Array,
    log_normalizer: jax.Array,
) -> jax.Array:
    """Evaluate the two independent target Gaussian trajectory blocks."""
    return jnp.sum(axis_log_probs(samples, mean, inverse_cholesky, log_normalizer), axis=-1)


def axis_log_probs(samples, mean, inverse_cholesky, log_normalizer):
    """Gaussian log densities using the supplied normalizer's convention."""
    structured = samples.reshape(samples.shape[:-1] + mean.shape)
    whitened = jnp.einsum("ij,...aj->...ai", inverse_cholesky, structured - mean)
    squared_mahalanobis = jnp.sum(jnp.square(whitened), axis=-1)
    return -0.5 * (squared_mahalanobis + log_normalizer / mean.shape[0])


def region_parameters(log_normalizer, dimension, coverage=REGION_COVERAGE):
    """Gaussian density boundary enclosing `coverage` of its probability mass.

    Squared Mahalanobis radius follows chi-square(dimension). Density decreases
    with radius, so a 95% mass region uses the *5th* log-density percentile.
    The normalizer must use the same constants as the evaluated log density.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    radius_squared = float(chi2.ppf(coverage, df=dimension))
    cutoff = -0.5 * (float(log_normalizer) + radius_squared)
    return radius_squared, cutoff


def flattened_log_score(log_prob, cutoff):
    """Unnormalized log density with a soft plateau inside the Gaussian region.

    Both arguments may include or omit the Gaussian's constant terms, provided
    they use the same convention. The difference is invariant to that choice.
    """
    return -jax.nn.softplus(cutoff - log_prob)**2


def initialize_final_affine(variables, target_mean, inverse_cholesky):
    """Initialize output translation and isotropic scale from the reference GP.

    The Gaussian reference has independent spatial axes with equal covariance.
    Recover its covariance factor in host float64 and use its squared Frobenius
    norm to obtain the trace. Both cloning targets use this same initialization.
    Only the final affine parameters change; the spline initialization is kept.
    """
    mean = np.asarray(jax.device_get(target_mean), dtype=np.float64)
    inverse = np.asarray(jax.device_get(inverse_cholesky), dtype=np.float64)
    if mean.ndim != 2 or inverse.shape != (mean.shape[-1], mean.shape[-1]):
        raise ValueError("Expected per-axis means and a shared inverse Cholesky")
    if mean.size != variables["params"]["final_bias"].size:
        raise ValueError("Target mean dimension does not match the NF")
    factor = np.linalg.solve(inverse, np.eye(inverse.shape[0]))
    covariance_trace = float(mean.shape[0] * np.sum(np.square(factor)))
    average_variance = covariance_trace / mean.size
    if not np.isfinite(mean).all() or not math.isfinite(average_variance) or average_variance <= 0:
        raise ValueError("Target mean and average variance must be finite, with positive variance")
    scale = math.sqrt(average_variance)
    # _final_affine uses exp(raw) for contraction and 1+raw for expansion.
    raw_scale = math.log(scale) if scale < 1.0 else scale - 1.0
    if raw_scale < -5.0:
        raise ValueError("Target average scale is below the final affine's exp(-5) minimum")
    params = variables["params"]
    initialized = {
        **variables,
        "params": {
            **params,
            "final_bias": jnp.asarray(mean.reshape(-1), dtype=params["final_bias"].dtype),
            "final_raw_scale": jnp.full_like(params["final_raw_scale"], raw_scale),
        },
    }
    return initialized, {
        "method": "reference_gaussian_mean_and_trace",
        "reference_covariance_trace": covariance_trace,
        "dimension": int(mean.size),
        "isotropic_variance": average_variance,
        "isotropic_standard_deviation": scale,
    }


def make_train_chunk(
    flow: NormalizingFlow,
    optimizer: optax.GradientTransformation,
    target_mean: jax.Array,
    inverse_cholesky: jax.Array,
    target_log_normalizer: jax.Array,
    *,
    target_kind: str = "gaussian",
    coverage: float = REGION_COVERAGE,
    batch_size: int | None = None,
    report_every: int | None = None,
    boundary_loss_weight: float = BOUNDARY_LOSS_WEIGHT,
    include_constant: bool = False,
):
    """Create reverse-KL updates with a separate spline-boundary penalty.

    The boundary penalty measures internal spline inputs before either
    output affine transform.
    """
    if target_kind not in ("gaussian", "flat95"):
        raise ValueError(f"Unknown target kind: {target_kind}")
    if not math.isfinite(boundary_loss_weight) or boundary_loss_weight < 0:
        raise ValueError("boundary_loss_weight must be finite and nonnegative")
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    report_every = REPORT_EVERY if report_every is None else report_every
    dimension = target_mean.size
    _, cutoff = region_parameters(float(target_log_normalizer), dimension, coverage)
    use_core_entropy = isinstance(flow, FixedAffineFlow)

    def train_step(
        carry: tuple[Any, optax.OptState, jax.Array],
        _: None,
    ) -> tuple[
        tuple[Any, optax.OptState, jax.Array],
        dict[str, jax.Array],
    ]:
        variables, optimizer_state, key = carry
        key, sample_key = jax.random.split(key)

        def loss_fn(current_params: Any):
            current_variables = {**variables, "params": current_params}
            if use_core_entropy:
                samples, entropy_samples, sample_boundary_losses = flow.apply(
                    current_variables, sample_key, batch_size,
                    method=flow.sample_with_core, return_boundary_loss=True,
                )
                entropy_offset = flow.apply(current_variables, method=flow.entropy_offset)
                density_method = flow.log_prob_core
            else:
                samples, sample_boundary_losses = flow.apply(
                    current_variables, sample_key, batch_size,
                    method=flow.sample, return_boundary_loss=True,
                )
                entropy_samples = samples
                entropy_offset = jnp.float32(0)
                density_method = flow.log_prob
            frozen_variables = jax.tree.map(
                jax.lax.stop_gradient,
                current_variables,
            )
            # STL: stop density parameters, retaining the path through samples.
            # For fixed-affine flows evaluate directly on core outputs; never
            # transform to physical coordinates and back for the entropy term.
            log_q_core = flow.apply(
                frozen_variables,
                entropy_samples,
                method=density_method,
                include_constant=include_constant,
            )
            log_q = log_q_core - entropy_offset
            axis_log_p = axis_log_probs(
                samples,
                target_mean,
                inverse_cholesky,
                target_log_normalizer,
            )
            gaussian_log_p = jnp.sum(axis_log_p, axis=-1)
            inside_region = gaussian_log_p >= cutoff
            if target_kind == "flat95":
                log_p = flattened_log_score(gaussian_log_p, cutoff)
            else:
                log_p = gaussian_log_p
            # Keep log_q (negative entropy) even on the flat plateau. Dropping
            # it would not fit a distribution and would permit concentration.
            optimization_kl = jnp.mean(log_q_core - log_p)
            sample_kl = optimization_kl - entropy_offset
            boundary_loss = jnp.mean(sample_boundary_losses)
            weighted_boundary_loss = boundary_loss_weight * boundary_loss
            total_loss = optimization_kl + weighted_boundary_loss
            metrics = {
                "loss": sample_kl + weighted_boundary_loss,
                "optimization_loss": total_loss,
                "boundary_loss": boundary_loss,
                "weighted_boundary_loss": weighted_boundary_loss,
                ("reverse_kl" if target_kind == "gaussian"
                 else "reverse_kl_up_to_constant"): sample_kl,
                "mean_log_q": jnp.mean(log_q),
                "mean_log_p": jnp.mean(log_p),
                "mean_gaussian_log_prob": jnp.mean(gaussian_log_p),
                "inside_original_region": jnp.mean(inside_region),
                "entropy": -jnp.mean(log_q),
                "core_entropy": -jnp.mean(log_q_core),
                "fixed_affine_entropy_offset": entropy_offset,
                "mean_squared_mahalanobis_radius": jnp.mean(
                    -2.0 * gaussian_log_p - target_log_normalizer
                ),
            }
            return total_loss, metrics

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(variables["params"])
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            variables["params"],
        )
        variables = {**variables, "params": optax.apply_updates(variables["params"], updates)}
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return (variables, optimizer_state, key), metrics

    @partial(jax.jit, static_argnames=("num_updates",))
    def train_chunk(
        variables: Any,
        optimizer_state: optax.OptState,
        key: jax.Array,
        num_updates: int = report_every,
    ) -> tuple[
        Any,
        optax.OptState,
        jax.Array,
        dict[str, jax.Array],
    ]:
        (
            variables,
            optimizer_state,
            key,
        ), stacked_metrics = jax.lax.scan(
            train_step,
            (variables, optimizer_state, key),
            xs=None,
            length=num_updates,
        )
        averaged_metrics = jax.tree.map(
            lambda values: jnp.mean(values, axis=0),
            stacked_metrics,
        )
        return variables, optimizer_state, key, averaged_metrics

    return train_chunk


def checkpoint_paths(target_kind: str, *, model: CloneModel | None = None) -> tuple[Path, Path]:
    """Keep the Gaussian's established path and save the flat model separately."""
    if target_kind not in ("gaussian", "flat95"):
        raise ValueError(f"Unknown target kind: {target_kind}")
    if model is not None:
        stem = model.checkpoint_stem
        suffix = "_flat95" if target_kind == "flat95" else ""
        return (
            stem.with_name(f"{stem.name}{suffix}_variables.msgpack"),
            stem.with_name(f"{stem.name}{suffix}_metadata.json"),
        )
    if target_kind == "gaussian":
        return CHECKPOINT_PATH, METADATA_PATH
    if target_kind == "flat95":
        return (
            OUTPUT_DIRECTORY / "trajectory_flow_flat95_variables.msgpack",
            OUTPUT_DIRECTORY / "trajectory_flow_flat95_metadata.json",
        )
    raise ValueError(f"Unknown target kind: {target_kind}")


def target_metadata(target_kind, target_mean, log_normalizer, coverage, *, include_constant=False):
    """Record the density convention and the joint Gaussian reference region."""
    dimension = int(target_mean.size)
    result = {
        "kind": target_kind,
        "dimension": dimension,
        "base_gaussian_axes_factorized": True,
        "target_axes_factorized": target_kind == "gaussian",
        "gaussian_includes_2pi": include_constant,
        "region_coverage": coverage,
        "log_density_quantile_at_boundary": 1.0 - coverage,
        "score": ("joint_gaussian_log_prob" if target_kind == "gaussian"
                  else "-softplus(cutoff - joint_gaussian_log_prob)"),
        "normalized": target_kind == "gaussian" and include_constant,
    }
    if log_normalizer is not None:
        radius_squared, cutoff = region_parameters(
            float(log_normalizer), dimension, coverage,
        )
        result.update(
            gaussian_log_normalizer=float(log_normalizer),
            gaussian_peak_log_prob=-0.5 * float(log_normalizer),
            squared_mahalanobis_cutoff=radius_squared,
            gaussian_log_prob_cutoff=cutoff,
        )
    return result


def save_result(
    variables: Any,
    target_mean: jax.Array,
    *,
    target_kind: str = "gaussian",
    target_log_normalizer: jax.Array | None = None,
    inverse_cholesky: jax.Array | None = None,
    coverage: float = REGION_COVERAGE,
    training_config: dict | None = None,
    boundary_loss_weight: float = BOUNDARY_LOSS_WEIGHT,
    model: CloneModel | None = None,
    include_constant: bool = False,
) -> tuple[Path, Path]:
    """Save variables and enough metadata to distinguish both target densities."""
    checkpoint_path, metadata_path = checkpoint_paths(target_kind, model=model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(flax.serialization.to_bytes(variables))
    metadata = {
        "checkpoint": str(checkpoint_path),
        "seed": SEED,
        "log_prob_includes_2pi": include_constant,
        "objective": ("reverse KL: KL(flow || hand-engineered target)"
                      if target_kind == "gaussian"
                      else "reverse KL up to the soft-flat target normalizing constant"),
        "regularization": {
            "boundary_loss_weight": boundary_loss_weight,
            "boundary_loss": (
                "mean over samples and coupling layers of summed squared "
                "excess outside the spline tail bounds"
            ) if model is None or model.has_spline_boundaries else "none (no spline boundaries)",
            "total_loss": ("reverse_kl_objective + boundary_loss_weight * boundary_loss"
                           if model is None or model.has_spline_boundaries else "reverse_kl_objective"),
        },
        "target": target_metadata(
            target_kind, target_mean, target_log_normalizer, coverage,
            include_constant=include_constant,
        ),
        "architecture": FLOW_ARCHITECTURE if model is None else model.architecture,
        "encoding": {
            "type": ENCODING_TYPE,
            "metric_samples": METRIC_SAMPLES,
            "metric": "mean_over_time_of_squared_euclidean_position_distance",
            "time_grid": "linspace_inclusive",
            "transform": "L.T @ [positions_without_x0, velocities] per axis",
        },
        "flow": asdict(FLOW_CONFIG if model is None else model.config),
        "task": {
            "way_points": WAY_POINTS,
            "steps_per_way_point": STEPS_PER_WAY_POINT,
            "var": VAR,
            "length_scale": LENGTH_SCALE,
            "velocity_noise_var": VELOCITY_NOISE_VAR,
            "dt": DT,
        },
        "target_mean": np.asarray(jax.device_get(target_mean)).tolist(),
    }
    if inverse_cholesky is not None:
        metadata["gaussian_inverse_cholesky_per_axis"] = np.asarray(
            jax.device_get(inverse_cholesky)
        ).tolist()
    if training_config is not None:
        metadata["training"] = training_config
    if "fixed_affine" in variables:
        fixed = jax.device_get(variables["fixed_affine"]["state"])
        fixed_metadata = (model.fixed_affine_metadata or {}) if model is not None else {}
        metadata["fixed_affine"] = {
            **fixed_metadata,
            "matrix": np.asarray(fixed.matrix).tolist(),
            "inverse": np.asarray(fixed.inverse_matrix).tolist(),
            "log_abs_det": float(fixed.log_abs_det),
            "entropy_offset": float(fixed.log_abs_det),
            "log_prob_offset": -float(fixed.log_abs_det),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return checkpoint_path, metadata_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("gaussian", "flat95", "both"), default="both")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--report-every", type=int, default=REPORT_EVERY)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--boundary-loss-weight", type=float, default=BOUNDARY_LOSS_WEIGHT,
                        help="Weight of internal spline-boundary excursions (default: 1).")
    parser.add_argument("--coverage", type=float, default=REGION_COVERAGE,
                        help="Mass of the joint Gaussian region defining the soft plateau.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--include-constant", action="store_true",
                        help="Include d/2 * log(2*pi) in flow and Gaussian target log densities.")
    args = parser.parse_args(argv)
    if min(args.num_steps, args.batch_size, args.report_every) < 1:
        parser.error("num-steps, batch-size, and report-every must be positive")
    if args.learning_rate <= 0 or not 0 < args.coverage < 1:
        parser.error("learning-rate must be positive and coverage must lie in (0, 1)")
    if not math.isfinite(args.boundary_loss_weight) or args.boundary_loss_weight < 0:
        parser.error("boundary-loss-weight must be finite and nonnegative")
    return args


def require_gpu():
    """Never silently start a production cloning run on the CPU."""
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        raise SystemExit(
            "No usable JAX GPU is available. Run this script on the GPU workstation."
        ) from None
    if not devices:
        raise SystemExit("No JAX GPU devices found; cloning was not started.")


def train_target(target_kind, args, distribution, flow_key, training_key, *, model=None):
    target_mean, inverse_cholesky, target_log_normalizer = distribution
    include_constant = args.include_constant
    # Both targets start independently from the same initialization and RNG,
    # allowing comparison at the same architecture and training budget.
    flow_config = FLOW_CONFIG if model is None else model.config
    initializer = init_flow if model is None else model.initializer
    architecture = FLOW_ARCHITECTURE if model is None else model.architecture
    model_name = "trajectory-flow" if model is None else model.checkpoint_stem.name.replace("_", "-")
    flow, variables = initializer(flow_key, flow_config)
    if isinstance(flow, FixedAffineFlow):
        initialization = {"method": "default_core_with_fixed_prior_affine",
                          "fixed_affine": model.fixed_affine_metadata if model is not None else None}
    elif "final_raw_scale" in variables["params"] and "final_bias" in variables["params"]:
        variables, initialization = initialize_final_affine(
            variables, target_mean, inverse_cholesky,
        )
    else:
        initialization = {"method": "default_core_without_final_affine"}
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(args.learning_rate),
    )
    # Fixed matrices are checkpointed but never enter the optimizer state.
    optimizer_state = optimizer.init(variables["params"])
    train_chunk = make_train_chunk(
        flow, optimizer, target_mean, inverse_cholesky, target_log_normalizer,
        target_kind=target_kind, coverage=args.coverage,
        batch_size=args.batch_size, report_every=args.report_every,
        boundary_loss_weight=args.boundary_loss_weight,
        include_constant=include_constant,
    )
    config = {
        "seed": SEED,
        "log_prob_includes_2pi": include_constant,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_gradient_norm": MAX_GRADIENT_NORM,
        "boundary_loss_weight": args.boundary_loss_weight,
        "report_every": args.report_every,
        "architecture": architecture,
        "encoding_type": ENCODING_TYPE,
        "metric_samples": METRIC_SAMPLES,
        "flow": asdict(flow_config),
        "initialization": initialization,
        "target": target_metadata(
            target_kind, target_mean, target_log_normalizer, args.coverage,
            include_constant=include_constant,
        ),
    }
    run = wandb.init(
        entity="airl-lab", project="TBQDRL", group="Trajectory NF cloning",
        name=(f"{target_kind}-physical-encoding" if model is None
              else f"{model_name}-{target_kind}-physical-encoding"),
        mode=args.wandb_mode, config=config,
    )
    history = []
    try:
        completed = 0
        while completed < args.num_steps:
            updates = min(args.report_every, args.num_steps - completed)
            variables, optimizer_state, training_key, metrics = train_chunk(
                variables, optimizer_state, training_key, num_updates=updates,
            )
            host_metrics = {
                name: float(value)
                for name, value in jax.device_get(metrics).items()
            }
            if not all(math.isfinite(value) for value in host_metrics.values()):
                raise FloatingPointError(
                    f"{target_kind}: non-finite metrics after update {completed + updates}"
                )
            completed += updates
            history.append({"step": completed, **host_metrics})
            run.log({f"train/{name}": value for name, value in host_metrics.items()},
                    step=completed)
            loss_key = ("reverse_kl" if target_kind == "gaussian"
                        else "reverse_kl_up_to_constant")
            print(
                f"{target_kind} {completed}/{args.num_steps} | "
                f"loss {host_metrics['loss']:.6f} | "
                f"KL objective {host_metrics[loss_key]:.6f} | "
                f"boundary {host_metrics['boundary_loss']:.6f} | "
                f"entropy {host_metrics['entropy']:.6f} | "
                f"inside Gaussian region {host_metrics['inside_original_region']:.4f}",
                flush=True,
            )

        checkpoint_path, metadata_path = save_result(
            variables, target_mean, target_kind=target_kind,
            target_log_normalizer=target_log_normalizer,
            inverse_cholesky=inverse_cholesky, coverage=args.coverage,
            training_config=config,
            boundary_loss_weight=args.boundary_loss_weight,
            model=model,
            include_constant=include_constant,
        )
        history_path = checkpoint_path.with_name(
            checkpoint_path.stem.removesuffix("_variables") + "_training_history.json"
        )
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        artifact = wandb.Artifact(
            f"{model_name}-{target_kind}-variables", type="model",
            metadata={"architecture": architecture,
                      "encoding_type": ENCODING_TYPE, "target_kind": target_kind},
        )
        for path in (checkpoint_path, metadata_path, history_path):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        print(f"Saved {target_kind}: {checkpoint_path}", flush=True)
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    require_gpu()
    model = make_fixed_affine_model()
    key = jax.random.PRNGKey(SEED)
    _, target_key, flow_key, training_key = jax.random.split(key, 4)
    distribution = make_target_distribution(target_key, include_constant=args.include_constant)
    targets = ("gaussian", "flat95") if args.target == "both" else (args.target,)
    for target_kind in targets:
        train_target(target_kind, args, distribution, flow_key, training_key, model=model)


if __name__ == "__main__":
    main()
