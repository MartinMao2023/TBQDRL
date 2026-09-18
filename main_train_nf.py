"""Train a normalizing flow to clone the trajectory task distribution."""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import wandb
from brax import envs

from jax_rq_nsf import FlowConfig, NormalizingFlow, init_flow
from task_wrappers.trajectory_following import AntFiniteMaternWrapper


SEED = 0
NUM_STEPS = 10_000
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
MAX_GRADIENT_NORM = 5.0
REPORT_EVERY = 100

WAY_POINTS = 12
STEPS_PER_WAY_POINT = 8
VAR = 4
LENGTH_SCALE = 1.0
VELOCITY_NOISE_VAR = 0.25
DT = 0.05

OUTPUT_DIRECTORY = Path("output")
CHECKPOINT_PATH = OUTPUT_DIRECTORY / "trajectory_flow_variables.msgpack"
METADATA_PATH = OUTPUT_DIRECTORY / "trajectory_flow_metadata.json"

FLOW_CONFIG = FlowConfig(
    dimension=4 * WAY_POINTS + 2,
    num_layers=8,
    hidden_features=(128, 128),
    num_bins=16,
    tail_bound=6.0,
)


def make_target_distribution(
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the target mean, inverse Cholesky, and log normalizer."""
    base_env = envs.create(
        env_name="ant",
        episode_length=4_096,
        backend="mjx",
        reset_noise_scale=0.0,
    )
    task_env = AntFiniteMaternWrapper(
        base_env,
        way_points=WAY_POINTS,
        steps_per_way_point=STEPS_PER_WAY_POINT,
        var=VAR,
        l=LENGTH_SCALE,
        v_noise_var=VELOCITY_NOISE_VAR,
        dt=DT,
    )
    initial_env_state = base_env.reset(key)
    initial_velocity = initial_env_state.obs[13:15]

    mean = task_env.posterior_mu_T * initial_velocity[:, None]
    cholesky = task_env.posterior_L
    inverse_cholesky = jsp.linalg.solve_triangular(
        cholesky,
        jnp.eye(cholesky.shape[0], dtype=cholesky.dtype),
        lower=True,
    )

    dimensions_per_axis = cholesky.shape[0]
    axis_log_normalizer = (
        dimensions_per_axis * math.log(2.0 * math.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    )
    log_normalizer = 2.0 * axis_log_normalizer
    return mean, inverse_cholesky, log_normalizer


def target_log_prob(
    samples: jax.Array,
    mean: jax.Array,
    inverse_cholesky: jax.Array,
    log_normalizer: jax.Array,
) -> jax.Array:
    """Evaluate the two independent target Gaussian trajectory blocks."""
    structured_samples = samples.reshape(
        samples.shape[0],
        2,
        2 * WAY_POINTS + 1,
    )
    centered = structured_samples - mean[None, ...]
    whitened = jnp.einsum("ij,baj->bai", inverse_cholesky, centered)
    squared_mahalanobis = jnp.sum(jnp.square(whitened), axis=(1, 2))
    return -0.5 * (squared_mahalanobis + log_normalizer)


def make_train_chunk(
    flow: NormalizingFlow,
    optimizer: optax.GradientTransformation,
    target_mean: jax.Array,
    inverse_cholesky: jax.Array,
    target_log_normalizer: jax.Array,
):
    """Create a compiled block of reverse-KL optimization steps."""

    def train_step(
        carry: tuple[Any, optax.OptState, jax.Array],
        _: None,
    ) -> tuple[
        tuple[Any, optax.OptState, jax.Array],
        dict[str, jax.Array],
    ]:
        variables, optimizer_state, key = carry
        key, sample_key = jax.random.split(key)

        def loss_fn(current_variables: Any):
            samples = flow.apply(
                current_variables,
                sample_key,
                BATCH_SIZE,
                method=flow.sample,
            )
            frozen_variables = jax.tree.map(
                jax.lax.stop_gradient,
                current_variables,
            )
            log_q = flow.apply(
                frozen_variables,
                samples,
                method=flow.log_prob,
            )
            log_p = target_log_prob(
                samples,
                target_mean,
                inverse_cholesky,
                target_log_normalizer,
            )
            reverse_kl_samples = log_q - log_p
            sample_kl = jnp.mean(reverse_kl_samples)
            metrics = {
                "reverse_kl": sample_kl,
                "mean_log_q": jnp.mean(log_q),
                "mean_log_p": jnp.mean(log_p),
            }
            return sample_kl, metrics

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(variables)
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            variables,
        )
        variables = optax.apply_updates(variables, updates)
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return (variables, optimizer_state, key), metrics

    @jax.jit
    def train_chunk(
        variables: Any,
        optimizer_state: optax.OptState,
        key: jax.Array,
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
            length=REPORT_EVERY,
        )
        averaged_metrics = jax.tree.map(
            lambda values: jnp.mean(values, axis=0),
            stacked_metrics,
        )
        return variables, optimizer_state, key, averaged_metrics

    return train_chunk


def save_result(
    variables: Any,
    target_mean: jax.Array,
) -> None:
    """Save the complete Flax variables tree and reconstruction metadata."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_bytes(flax.serialization.to_bytes(variables))
    metadata = {
        "checkpoint": str(CHECKPOINT_PATH),
        "seed": SEED,
        "objective": "reverse KL: KL(flow || hand-engineered target)",
        "flow": asdict(FLOW_CONFIG),
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
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    key, target_key, flow_key, training_key = jax.random.split(key, 4)

    (
        target_mean,
        inverse_cholesky,
        target_log_normalizer,
    ) = make_target_distribution(target_key)
    flow, variables = init_flow(flow_key, FLOW_CONFIG)

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(LEARNING_RATE),
    )
    optimizer_state = optimizer.init(variables)
    train_chunk = make_train_chunk(
        flow,
        optimizer,
        target_mean,
        inverse_cholesky,
        target_log_normalizer,
    )

    run = wandb.init(
        entity="airl-lab",
        project="TBQDRL",
        group="Trajectory NF cloning",
        config={
            "seed": SEED,
            "num_steps": NUM_STEPS,
            "effective_num_steps": (
                math.ceil(NUM_STEPS / REPORT_EVERY) * REPORT_EVERY
            ),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_gradient_norm": MAX_GRADIENT_NORM,
            "report_every": REPORT_EVERY,
            "objective": "reverse KL",
            "flow": asdict(FLOW_CONFIG),
            "task": {
                "way_points": WAY_POINTS,
                "steps_per_way_point": STEPS_PER_WAY_POINT,
                "var": VAR,
                "length_scale": LENGTH_SCALE,
                "velocity_noise_var": VELOCITY_NOISE_VAR,
                "dt": DT,
            },
        },
    )

    for report_index in range(math.ceil(NUM_STEPS / REPORT_EVERY)):
        variables, optimizer_state, training_key, training_metrics = train_chunk(
            variables,
            optimizer_state,
            training_key,
        )
        wandb.log(
            {
                f"train/{name}": float(value)
                for name, value in training_metrics.items()
            },
            step=(report_index + 1) * REPORT_EVERY,
        )

    save_result(variables, target_mean)
    artifact = wandb.Artifact(
        "trajectory-flow-variables",
        type="model",
        metadata={"flow": asdict(FLOW_CONFIG)},
    )
    artifact.add_file(str(CHECKPOINT_PATH))
    artifact.add_file(str(METADATA_PATH))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
