"""Filter a trajectory flow using a trained critic and variational inference."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import optax
import wandb
from brax import envs
from optax.losses import sigmoid_binary_cross_entropy

from jax_rq_nsf import FlowConfig, NormalizingFlow, init_flow
from networks import GCMLP


SEED = 0
NUM_STEPS = 10_000
BATCH_SIZE = 4_096
LEARNING_RATE = 3e-4
MAX_GRADIENT_NORM = 5.0
REPORT_EVERY = 100

WAY_POINTS = 12
FLOW_DIMENSION = 4 * WAY_POINTS + 2
Z_DIMENSION = FLOW_DIMENSION + 5

TARGET_VALUE = 0.95
TARGET_PROBABILITY = TARGET_VALUE * 31.0 / 64.0 + 0.5
TARGET_LOGIT = math.log(
    TARGET_PROBABILITY / (1.0 - TARGET_PROBABILITY)
)

OLD_FLOW_CHECKPOINT = Path("output/trajectory_flow_variables.msgpack")
CRITIC_CHECKPOINT = Path("output/nf_matern/saved_data/critic.msgpack")
OUTPUT_DIRECTORY = Path("output/vi_filtered")
NEW_FLOW_CHECKPOINT = OUTPUT_DIRECTORY / "trajectory_flow_variables.msgpack"
METADATA_PATH = OUTPUT_DIRECTORY / "trajectory_flow_metadata.json"

FLOW_CONFIG = FlowConfig(
    dimension=FLOW_DIMENSION,
    num_layers=8,
    hidden_features=(128, 128),
    num_bins=16,
    tail_bound=6.0,
)

CRITIC_HIDDEN_LAYERS = (128, 128)
INITIAL_Z_PREFIX = jnp.array(
    [0.0, 1.0, -1.0, 0.0, 0.0],
    dtype=jnp.float32,
)


def load_flow(
    key: jax.Array,
    checkpoint_path: Path,
) -> tuple[NormalizingFlow, Any]:
    """Construct a flow and restore its complete variables tree."""
    flow, variables_template = init_flow(key, FLOW_CONFIG)
    variables = flax.serialization.from_bytes(
        variables_template,
        checkpoint_path.read_bytes(),
    )
    return flow, variables


def create_critic_and_initial_observation(
    reset_key: jax.Array,
    critic_key: jax.Array,
) -> tuple[GCMLP, Any, jax.Array]:
    """Restore the critic and record the deterministic initial observation."""
    env = envs.create(
        env_name="ant",
        episode_length=4_096,
        backend="mjx",
        reset_noise_scale=0.0,
    )
    initial_observation = env.reset(reset_key).obs
    critic = GCMLP(
        layer_sizes=CRITIC_HIDDEN_LAYERS + (1,),
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        activation=nn.silu,
        kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    )
    critic_template = critic.init(
        critic_key,
        obs=initial_observation,
        z=jnp.zeros((Z_DIMENSION,), dtype=initial_observation.dtype),
    )
    critic_variables = flax.serialization.from_bytes(
        critic_template,
        CRITIC_CHECKPOINT.read_bytes(),
    )
    return critic, critic_variables, initial_observation


def make_train_chunk(
    flow: NormalizingFlow,
    old_flow_variables: Any,
    critic: GCMLP,
    critic_variables: Any,
    initial_observation: jax.Array,
    optimizer: optax.GradientTransformation,
):
    """Create a compiled block of critic-guided VI updates."""

    def train_step(
        carry: tuple[Any, optax.OptState, jax.Array],
        _: None,
    ) -> tuple[
        tuple[Any, optax.OptState, jax.Array],
        dict[str, jax.Array],
    ]:
        new_flow_variables, optimizer_state, key = carry
        key, sample_key = jax.random.split(key)

        def loss_fn(current_variables: Any):
            flow_samples = flow.apply(
                current_variables,
                sample_key,
                BATCH_SIZE,
                method=flow.sample,
            )

            z_prefix = jnp.broadcast_to(
                INITIAL_Z_PREFIX,
                (BATCH_SIZE, INITIAL_Z_PREFIX.shape[0]),
            )
            zs = jnp.concatenate([z_prefix, flow_samples], axis=-1)
            observations = jnp.broadcast_to(
                initial_observation,
                (BATCH_SIZE, initial_observation.shape[0]),
            )
            critic_logits = critic.apply(
                critic_variables,
                observations,
                zs,
            )[..., 0]
            clipped_logits = jnp.minimum(critic_logits, TARGET_LOGIT)
            log_likelihood = -sigmoid_binary_cross_entropy(
                clipped_logits,
                TARGET_PROBABILITY,
            )

            old_log_prob = flow.apply(
                old_flow_variables,
                flow_samples,
                method=flow.log_prob,
            )
            frozen_new_variables = jax.tree.map(
                jax.lax.stop_gradient,
                current_variables,
            )
            new_log_prob = flow.apply(
                frozen_new_variables,
                flow_samples,
                method=flow.log_prob,
            )

            elbo_samples = (
                log_likelihood
                + old_log_prob
                - new_log_prob
            )
            critic_values = jnp.clip(
                (jax.nn.sigmoid(critic_logits) * 64.0 - 32.0) / 31.0,
                0.0,
                1.0,
            )
            metrics = {
                "elbo": jnp.mean(elbo_samples),
                "log_likelihood": jnp.mean(log_likelihood),
                "log_prior": jnp.mean(old_log_prob),
                "entropy": -jnp.mean(new_log_prob),
                "kl_new_old": jnp.mean(new_log_prob - old_log_prob),
                "critic_value": jnp.mean(critic_values),
                "critic_logit": jnp.mean(critic_logits),
                "target_fraction": jnp.mean(
                    critic_logits >= TARGET_LOGIT
                ),
            }
            return -metrics["elbo"], metrics

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(new_flow_variables)
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            new_flow_variables,
        )
        new_flow_variables = optax.apply_updates(
            new_flow_variables,
            updates,
        )
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return (new_flow_variables, optimizer_state, key), metrics

    @jax.jit
    def train_chunk(
        new_flow_variables: Any,
        optimizer_state: optax.OptState,
        key: jax.Array,
    ) -> tuple[
        Any,
        optax.OptState,
        jax.Array,
        dict[str, jax.Array],
    ]:
        (
            new_flow_variables,
            optimizer_state,
            key,
        ), stacked_metrics = jax.lax.scan(
            train_step,
            (new_flow_variables, optimizer_state, key),
            xs=None,
            length=REPORT_EVERY,
        )
        averaged_metrics = jax.tree.map(
            lambda values: jnp.mean(values, axis=0),
            stacked_metrics,
        )
        return (
            new_flow_variables,
            optimizer_state,
            key,
            averaged_metrics,
        )

    return train_chunk


def save_result(new_flow_variables: Any) -> None:
    """Save the filtered flow and the configuration needed to restore it."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    NEW_FLOW_CHECKPOINT.write_bytes(
        flax.serialization.to_bytes(new_flow_variables)
    )
    metadata = {
        "checkpoint": str(NEW_FLOW_CHECKPOINT),
        "old_flow_checkpoint": str(OLD_FLOW_CHECKPOINT),
        "critic_checkpoint": str(CRITIC_CHECKPOINT),
        "seed": SEED,
        "objective": (
            "critic log-likelihood + old-flow log-prior "
            "+ new-flow entropy"
        ),
        "target_value": TARGET_VALUE,
        "target_probability": TARGET_PROBABILITY,
        "target_logit": TARGET_LOGIT,
        "flow": asdict(FLOW_CONFIG),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    (
        flow_key,
        reset_key,
        critic_key,
        training_key,
    ) = jax.random.split(key, 4)

    flow, old_flow_variables = load_flow(
        flow_key,
        OLD_FLOW_CHECKPOINT,
    )
    new_flow_variables = jax.tree.map(
        lambda value: jnp.array(value),
        old_flow_variables,
    )
    (
        critic,
        critic_variables,
        initial_observation,
    ) = create_critic_and_initial_observation(reset_key, critic_key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRADIENT_NORM),
        optax.adam(LEARNING_RATE),
    )
    optimizer_state = optimizer.init(new_flow_variables)
    train_chunk = make_train_chunk(
        flow,
        old_flow_variables,
        critic,
        critic_variables,
        initial_observation,
        optimizer,
    )

    run = wandb.init(
        entity="airl-lab",
        project="TBQDRL",
        group="Trajectory NF VI filtering",
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
            "target_value": TARGET_VALUE,
            "target_probability": TARGET_PROBABILITY,
            "target_logit": TARGET_LOGIT,
            "old_flow_checkpoint": str(OLD_FLOW_CHECKPOINT),
            "critic_checkpoint": str(CRITIC_CHECKPOINT),
            "flow": asdict(FLOW_CONFIG),
        },
    )

    for report_index in range(math.ceil(NUM_STEPS / REPORT_EVERY)):
        (
            new_flow_variables,
            optimizer_state,
            training_key,
            metrics,
        ) = train_chunk(
            new_flow_variables,
            optimizer_state,
            training_key,
        )
        wandb.log(
            {
                f"train/{name}": float(value)
                for name, value in metrics.items()
            },
            step=(report_index + 1) * REPORT_EVERY,
        )

    save_result(new_flow_variables)
    artifact = wandb.Artifact(
        "vi-filtered-trajectory-flow",
        type="model",
        metadata={"flow": asdict(FLOW_CONFIG)},
    )
    artifact.add_file(str(NEW_FLOW_CHECKPOINT))
    artifact.add_file(str(METADATA_PATH))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
