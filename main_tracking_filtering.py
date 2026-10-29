"""Filter a saved tracking NF with its critic, then continue PPO for 500 iterations.

Default input: output/tracking/flat95/saved_data.
Default output: output/tracking/flat95_filtered/saved_data.
The posterior is proportional to the saved NF density times
exp(-4.665 * relu(2.44235 - critic_logits)**2).
"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from types import SimpleNamespace

import main_tracking as tracking


FILTER_LOGIT_THRESHOLD = 2.44235
FILTER_LOSS_WEIGHT = 4.665
INITIAL_Z_PREFIX = (0.0, 1.0, -1.0, 0.0, 0.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=tracking.PROJECT_ROOT / "output/tracking/flat95/saved_data")
    parser.add_argument("--output-dir", type=Path,
                        default=tracking.PROJECT_ROOT / "output/tracking/flat95_filtered/saved_data")
    parser.add_argument("--vi-updates", type=int, default=5000)
    parser.add_argument("--vi-batch-size", type=int, default=4096)
    parser.add_argument("--vi-learning-rate", type=float, default=3e-4)
    parser.add_argument("--vi-max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--vi-log-period", type=int, default=50)
    tracking.add_training_arguments(parser)
    parser.set_defaults(num_iterations=500)
    args = parser.parse_args(argv)
    tracking.validate_training_arguments(parser, args)
    if min(args.vi_updates, args.vi_batch_size, args.vi_log_period) < 1:
        parser.error("VI update count, batch size, and log period must be positive")
    for name in ("vi_learning_rate", "vi_max_gradient_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.checkpoint_dir == args.output_dir:
        parser.error("Use a separate output directory to retain the source checkpoint")
    return args


def select_source(args):
    """Check the saved run and its NF without importing training dependencies."""
    directory = args.checkpoint_dir
    description_path = directory / "description.json"
    if not description_path.is_file():
        raise FileNotFoundError(f"Tracking description not found: {description_path}")
    description = json.loads(description_path.read_text())
    if not (directory / "ppo_training_state.msgpack").is_file():
        for name in ("policy.msgpack", "critic.msgpack"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Tracking checkpoint not found: {directory / name}")
    selection = tracking.select_flow(SimpleNamespace(
        flow=None, flow_checkpoint=directory / "flow_variables.msgpack",
        flow_metadata=directory / "flow_metadata.json", output_dir=args.output_dir,
        vec_env=args.vec_env, mini_batch_size=args.mini_batch_size,
    ))
    if selection.metadata["architecture"] not in tracking.FIXED_AFFINE_CORES:
        raise ValueError("Filtering expects a current fixed-affine NF checkpoint")
    for name in ("task", "encoding"):
        if description[name] != selection.metadata[name]:
            raise ValueError(f"Saved PPO {name} does not match its NF metadata")
    return selection


def filtering_config(args):
    return {
        "loss": "4.665 * relu(2.44235 - critic_logits)**2",
        "logit_threshold": FILTER_LOGIT_THRESHOLD,
        "loss_weight": FILTER_LOSS_WEIGHT,
        "objective": "KL(new NF || saved NF) + mean critic filtering loss",
        "vi_updates": args.vi_updates,
        "vi_batch_size": args.vi_batch_size,
        "vi_learning_rate": args.vi_learning_rate,
        "vi_max_gradient_norm": args.vi_max_gradient_norm,
        "initial_z_prefix": list(INITIAL_Z_PREFIX),
        "critic_value": "sigmoid(critic_logits)",
        "fixed_affine_trainable": False,
        "source_checkpoint_directory": str(args.checkpoint_dir),
    }


def critic_filter_loss(critic_logits):
    import jax

    return FILTER_LOSS_WEIGHT * jax.nn.relu(FILTER_LOGIT_THRESHOLD - critic_logits)**2


def make_vi_train_chunk(flow, critic, optimizer, *, batch_size=4096):
    """Freeze the prior and critic, optimizing only the NF's params collection.

    Both densities use core coordinates: their common fixed-affine Jacobian
    cancels from KL. STL stops the new density's parameters, retaining sample
    gradients. The critic sees physical encodings at time zero, zero tracking
    deviation, and the deterministic reset observation, as in tracking PPO.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def train_step(carry, old_variables, critic_variables, initial_observation):
        variables, optimizer_state, key = carry
        key, sample_key = jax.random.split(key)

        def loss_fn(params):
            current = {**variables, "params": params}
            physical_samples, core_samples = flow.apply(
                current, sample_key, batch_size, method=flow.sample_with_core,
            )
            prefix = jnp.broadcast_to(jnp.asarray(INITIAL_Z_PREFIX, physical_samples.dtype),
                                      (batch_size, len(INITIAL_Z_PREFIX)))
            zs = jnp.concatenate([prefix, physical_samples], axis=-1)
            observations = jnp.broadcast_to(initial_observation,
                                            (batch_size, initial_observation.shape[-1]))
            logits = critic.apply(critic_variables, observations, zs)[..., 0]
            penalties = critic_filter_loss(logits)
            frozen_current = jax.tree.map(jax.lax.stop_gradient, current)
            log_q = flow.apply(frozen_current, core_samples, method=flow.log_prob_core)
            log_prior = flow.apply(old_variables, core_samples, method=flow.log_prob_core)
            kl = jnp.mean(log_q - log_prior)
            penalty = jnp.mean(penalties)
            loss = kl + penalty
            entropy_offset = flow.apply(current, method=flow.entropy_offset)
            metrics = {
                "loss": loss, "elbo": -loss, "kl_new_old": kl,
                "filter_loss": penalty, "entropy": -jnp.mean(log_q) + entropy_offset,
                "core_entropy": -jnp.mean(log_q), "critic_logit": jnp.mean(logits),
                "critic_value": jnp.mean(jax.nn.sigmoid(logits)),
                "fraction_above_threshold": jnp.mean(logits >= FILTER_LOGIT_THRESHOLD),
            }
            return loss, metrics

        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(variables["params"])
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, variables["params"])
        variables = {**variables, "params": optax.apply_updates(variables["params"], updates)}
        metrics["gradient_norm"] = optax.global_norm(gradients)
        return (variables, optimizer_state, key), metrics

    @partial(jax.jit, static_argnames=("num_updates",))
    def train_chunk(variables, optimizer_state, key, old_variables, critic_variables,
                    initial_observation, num_updates):
        old_variables = jax.tree.map(jax.lax.stop_gradient, old_variables)
        critic_variables = jax.tree.map(jax.lax.stop_gradient, critic_variables)
        carry, metrics = jax.lax.scan(
            lambda carry, _: train_step(carry, old_variables, critic_variables, initial_observation),
            (variables, optimizer_state, key), xs=None, length=num_updates,
        )
        averaged = jax.tree.map(lambda values: jnp.mean(values, axis=0), metrics)
        return *carry, averaged

    return train_chunk


def make_distribution_preparer(args):
    def prepare(flow, variables, metadata, env, critic, critic_variables):
        import jax
        import numpy as np
        import optax
        import wandb
        from flax import serialization

        initial_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
        training_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 2)
        initial_observation = env.env.reset(initial_key).obs
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.vi_max_gradient_norm), optax.adam(args.vi_learning_rate),
        )
        optimizer_state = optimizer.init(variables["params"])
        old_variables = variables
        train_chunk = make_vi_train_chunk(flow, critic, optimizer, batch_size=args.vi_batch_size)
        config = filtering_config(args)
        run = wandb.init(
            entity="airl-lab", project="TBQDRL", group="Tracking critic-guided VI filtering",
            name="tracking-flat95-vi-filter", config=config, mode=args.wandb_mode,
        )
        history = []
        try:
            completed = 0
            while completed < args.vi_updates:
                updates = min(args.vi_log_period, args.vi_updates - completed)
                variables, optimizer_state, training_key, metrics = train_chunk(
                    variables, optimizer_state, training_key, old_variables,
                    critic_variables, initial_observation, num_updates=updates,
                )
                report = {name: float(value) for name, value in jax.device_get(metrics).items()}
                if not all(math.isfinite(value) for value in report.values()):
                    raise FloatingPointError(f"Non-finite VI metrics after update {completed + updates}")
                completed += updates
                history.append({"update": completed, **report})
                run.log(report, step=completed)
                print(f"VI {completed}/{args.vi_updates} | loss {report['loss']:.6f} | "
                      f"KL {report['kl_new_old']:.6f} | critic {report['critic_value']:.4f} | "
                      f"above threshold {report['fraction_above_threshold']:.4f}", flush=True)

            metadata = {**metadata, "filtering": config}
            directory = args.output_dir / "vi_filtered"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / "flow_variables.msgpack"
            checkpoint.write_bytes(serialization.to_bytes(variables))
            (directory / "flow_metadata.json").write_text(json.dumps(
                {**metadata, "checkpoint": str(checkpoint)}, indent=2,
            ) + "\n")
            (directory / "vi_history.json").write_text(json.dumps(history, indent=2) + "\n")
            (directory / "vi_optimizer.msgpack").write_bytes(serialization.to_bytes(optimizer_state))
            np.save(directory / "vi_rng_key.npy", np.asarray(jax.device_get(training_key)))
            print(f"Saved filtered NF to {checkpoint}", flush=True)
        finally:
            run.finish()
        return variables, metadata

    return prepare


def main(argv=None):
    args = parse_args(argv)
    try:
        selection = select_source(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Invalid tracking checkpoint: {error}") from None
    print(json.dumps({
        **tracking.selection_summary(selection), "filtering": filtering_config(args),
        "additional_ppo_iterations": args.num_iterations,
        "optimizer_initialization": ("restore_full_state" if
            (args.checkpoint_dir / "ppo_training_state.msgpack").is_file()
            else "saved_weights_with_fresh_optimizers"),
    }, indent=2), flush=True)
    if not args.check_only:
        tracking.run_training(
            args, selection, checkpoint_directory=args.checkpoint_dir,
            distribution_preparer=make_distribution_preparer(args),
        )


if __name__ == "__main__":
    main()
