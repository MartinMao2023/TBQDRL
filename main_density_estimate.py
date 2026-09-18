"""Fit a 2D RQ-NSF to a brightness-weighted image density."""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import flax.serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from jax_rq_nsf import FlowConfig, NormalizingFlow, init_flow


# ---------------------------------------------------------------------------
# Design choices
# ---------------------------------------------------------------------------

IMAGE_PATH = Path("image_512x384_mean_normalized.npy")
CHECKPOINT_PATH = Path("density_flow_params.msgpack")
METADATA_PATH = Path("density_flow_metadata.json")
DENSITY_PATH = Path("density_estimate_512x512.npy")

SEED = 0
NUM_STEPS = 10_000
BATCH_SIZE = 8192
LEARNING_RATE = 3e-4
MAX_GRADIENT_NORM = 5.0
REPORT_EVERY = 100
DENSITY_GRID_SIZE = 512
DENSITY_EVALUATION_BATCH_SIZE = 16_384
DENSITY_PLOT_BOUNDS = ((-2.5, 2.5), (-2.5, 2.5))

# Pixel coordinates are mapped to this rectangle before being passed to the flow.
# Keeping the data just inside the spline interval avoids placing it directly
# on the boundary, where the spline joins its identity tails.
COORDINATE_BOUNDS = jnp.array([2.0, 1.5], dtype=jnp.float32)

FLOW_CONFIG = FlowConfig(
    dimension=2,
    num_layers=8,
    hidden_features=(128, 128),
    num_bins=32,
    tail_bound=4.0,
)

description = {
    "image_path": str(IMAGE_PATH),
    "seed": SEED,
    "num_steps": NUM_STEPS,
    "effective_num_steps": math.ceil(NUM_STEPS / REPORT_EVERY) * REPORT_EVERY,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_gradient_norm": MAX_GRADIENT_NORM,
    "report_every": REPORT_EVERY,
    "coordinate_bounds": [2.0, 1.5],
    "density_grid_size": DENSITY_GRID_SIZE,
    "density_plot_bounds": DENSITY_PLOT_BOUNDS,
    "flow": asdict(FLOW_CONFIG),
    "objective": "brightness-weighted negative log-likelihood",
}


def load_brightness(path: Path = IMAGE_PATH) -> jax.Array:
    """Load and validate a nonnegative 2D brightness array."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run prepare_image.ipynb first."
        )

    image = np.load(path, allow_pickle=False)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(f"Expected a floating-point image, got {image.dtype}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Brightness array contains NaN or infinite values")
    if np.any(image < 0.0):
        raise ValueError("Brightness values must be nonnegative")
    if not np.any(image > 0.0):
        raise ValueError("Brightness array must contain at least one positive value")

    return jnp.asarray(image, dtype=jnp.float32)


def bilinear_brightness(
    image: jax.Array, pixel_coordinates: jax.Array
) -> jax.Array:
    """Interpolate image brightness at continuous ``(x, y)`` coordinates."""
    height, width = image.shape
    x = jnp.clip(pixel_coordinates[..., 0], 0.0, width - 1.0)
    y = jnp.clip(pixel_coordinates[..., 1], 0.0, height - 1.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y1 = jnp.minimum(y0 + 1, height - 1)

    x_fraction = x - x0
    y_fraction = y - y0

    top = image[y0, x0] * (1.0 - x_fraction) + image[y0, x1] * x_fraction
    bottom = image[y1, x0] * (1.0 - x_fraction) + image[y1, x1] * x_fraction
    return top * (1.0 - y_fraction) + bottom * y_fraction


def sample_uniform_image(
    key: jax.Array,
    image: jax.Array,
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw uniform continuous image positions and interpolate their weights."""
    height, width = image.shape
    unit_coordinates = jax.random.uniform(
        key, shape=(batch_size, 2), dtype=image.dtype
    )

    pixel_coordinates = jnp.stack(
        (
            unit_coordinates[:, 0] * (width - 1),
            unit_coordinates[:, 1] * (height - 1),
        ),
        axis=-1,
    )
    model_coordinates = (
        2.0 * unit_coordinates - 1.0
    ) * COORDINATE_BOUNDS
    brightness = bilinear_brightness(image, pixel_coordinates)
    return model_coordinates, pixel_coordinates, brightness


def create_train_state(
    key: jax.Array,
    config: FlowConfig = FLOW_CONFIG,
) -> tuple[NormalizingFlow, TrainState]:
    """Initialize the flow and optimizer."""
    flow, variables = init_flow(key, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(LEARNING_RATE),
    )
    state = TrainState.create(
        apply_fn=flow.apply,
        params=variables["params"],
        tx=optimizer,
    )
    return flow, state


def count_parameters(params: Any) -> int:
    """Count scalar trainable parameters in a JAX parameter pytree."""
    return sum(parameter.size for parameter in jax.tree.leaves(params))


def make_sample_chunk(
    *,
    batch_size: int = BATCH_SIZE,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Create a compiled sampler vectorized over one key per training step."""

    @jax.jit
    def sample_chunk(
        keys: jax.Array,
        image: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def sample_one(key: jax.Array) -> tuple[jax.Array, jax.Array]:
            coordinates, _, brightness = sample_uniform_image(
                key, image, batch_size
            )
            return coordinates, brightness

        return jax.vmap(sample_one)(keys)

    return sample_chunk


def make_train_scan(
    flow: NormalizingFlow,
) -> Callable[
    [TrainState, tuple[jax.Array, jax.Array]],
    tuple[TrainState, dict[str, jax.Array]],
]:
    """Create a compiled optimizer loop over pre-sampled training batches."""

    @jax.jit
    def train_scan(
        state: TrainState,
        sampled_batches: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState, dict[str, Any]]:
        def scan_step(
            current_state: TrainState,
            samples: tuple[jax.Array, jax.Array],
        ) -> tuple[TrainState, dict[str, jax.Array]]:
            coordinates, brightness = samples

            def loss_fn(params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
                log_prob = flow.apply({"params": params}, coordinates)
                negative_log_prob = -log_prob

                # Self-normalized importance weighting approximates expectation
                # under image brightness from uniform coordinate samples.
                total_weight = jnp.sum(brightness)
                loss = jnp.sum(brightness * negative_log_prob) / jnp.maximum(
                    total_weight, jnp.finfo(brightness.dtype).tiny
                )
                effective_sample_size = total_weight**2 / jnp.maximum(
                    jnp.sum(brightness**2),
                    jnp.finfo(brightness.dtype).tiny,
                )
                metrics = {
                    "weighted_nll": loss,
                    "effective_sample_fraction": (
                        effective_sample_size / brightness.size
                    ),
                }
                return loss, metrics

            (_, metrics), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(current_state.params)
            current_state = current_state.apply_gradients(grads=gradients)
            return current_state, metrics

        return jax.lax.scan(
            scan_step,
            state,
            sampled_batches,
        )

    return train_scan


def save_result(
    state: TrainState,
    image_shape: tuple[int, int],
    config: FlowConfig = FLOW_CONFIG,
) -> None:
    """Save trained parameters and the information needed to reconstruct them."""
    CHECKPOINT_PATH.write_bytes(flax.serialization.to_bytes(state.params))
    metadata = {
        "flow_config": asdict(config),
        "image_shape": list(image_shape),
        "coordinate_bounds": np.asarray(COORDINATE_BOUNDS).tolist(),
        "parameter_file": str(CHECKPOINT_PATH),
        "parameter_count": count_parameters(state.params),
        "density_file": str(DENSITY_PATH),
        "density_grid_size": DENSITY_GRID_SIZE,
        "density_plot_bounds": DENSITY_PLOT_BOUNDS,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def evaluate_density_grid(
    flow: NormalizingFlow,
    params: Any,
    *,
    grid_size: int = DENSITY_GRID_SIZE,
    evaluation_batch_size: int = DENSITY_EVALUATION_BATCH_SIZE,
) -> np.ndarray:
    """Evaluate density on a regular ``(y, x)`` grid over the plot bounds."""
    number_of_points = grid_size**2
    if number_of_points % evaluation_batch_size != 0:
        raise ValueError(
            "grid_size**2 must be divisible by evaluation_batch_size"
        )

    (x_min, x_max), (y_min, y_max) = DENSITY_PLOT_BOUNDS
    x_coordinates = jnp.linspace(x_min, x_max, grid_size, dtype=jnp.float32)
    y_coordinates = jnp.linspace(y_min, y_max, grid_size, dtype=jnp.float32)
    grid_x, grid_y = jnp.meshgrid(
        x_coordinates, y_coordinates, indexing="xy"
    )
    coordinates = jnp.stack((grid_x, grid_y), axis=-1)
    coordinate_batches = coordinates.reshape(
        number_of_points // evaluation_batch_size,
        evaluation_batch_size,
        2,
    )

    @jax.jit
    def evaluate_batches(
        current_params: Any, batches: jax.Array
    ) -> jax.Array:
        def evaluate_batch(
            carry: None, batch: jax.Array
        ) -> tuple[None, jax.Array]:
            log_prob = flow.apply(
                {"params": current_params},
                batch,
                method=flow.log_prob,
            )
            return carry, log_prob

        _, log_prob_batches = jax.lax.scan(
            evaluate_batch,
            None,
            batches,
        )
        return log_prob_batches

    log_density = evaluate_batches(params, coordinate_batches)
    density = jnp.exp(log_density).reshape(grid_size, grid_size)
    return np.asarray(density, dtype=np.float32)


def train(
    image: jax.Array,
    training_key: jax.Array,
    flow: Any,
    state: Any,
    num_steps: int,
    run: Any,
) -> TrainState:
    """Train the flow against the continuous brightness-weighted image."""

    sample_chunk = make_sample_chunk()
    train_scan = make_train_scan(flow)

    num_chunks = math.ceil(num_steps / REPORT_EVERY)
    completed_steps = 0
    for _ in range(num_chunks):
        training_key, chunk_key = jax.random.split(training_key)
        step_keys = jax.random.split(chunk_key, REPORT_EVERY)

        # `vmap` prepares the ys consumed by the compiled `lax.scan`.
        sampled_batches = sample_chunk(step_keys, image)
        state, metric_history = train_scan(state, sampled_batches)
        completed_steps += REPORT_EVERY

        metrics = jax.tree.map(jnp.mean, metric_history)
        logged_metrics = {
            name: float(value) for name, value in metrics.items()
        }
        if run is not None:
            run.log(logged_metrics, step=completed_steps)

    return state


def main() -> None:
    image = load_brightness()
    print(
        f"Loaded {IMAGE_PATH}: shape={image.shape}, dtype={image.dtype}, "
        f"mean={float(jnp.mean(image)):.6f}"
    )
    initialization_key, training_key = jax.random.split(jax.random.key(SEED))
    flow, state = create_train_state(initialization_key)
    description["parameter_count"] = count_parameters(state.params)

    run = wandb.init(
        entity="airl-lab",
        group="NF tests",
        project="TBQDRL",
        config=description,
    )
    try:
        state = train(image, training_key, flow, state, NUM_STEPS, run)
        save_result(state, tuple(image.shape))
        density = evaluate_density_grid(flow, state.params)
        np.save(DENSITY_PATH, density, allow_pickle=False)
        
        display_density = density / np.maximum(np.percentile(density, 99), np.finfo(density.dtype).tiny)
        display_density = np.minimum(display_density, 1.0)
        run.log(
            {
                "density_estimate": wandb.Image(
                    display_density,
                    caption="Density on [-2.5, 2.5] x [-2.5, 2.5]",
                )
            },
            step=int(state.step),
        )
    finally:
        run.finish()
    print(f"Saved parameters to {CHECKPOINT_PATH}")
    print(f"Saved reconstruction metadata to {METADATA_PATH}")
    print(f"Saved density grid to {DENSITY_PATH}")


if __name__ == "__main__":
    main()
