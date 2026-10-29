"""Train tracking PPO using the combined flat95 skill NF by default."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_ROOT = Path(__file__).resolve().parent
ENCODING_TYPE = "physical_positions_velocities_v1"
FIXED_AFFINE_CORES = {
    "safe_nsf_fixed_affine_v1": "safe_flow_even_permutation_restore_conditional_offset_v6",
    "real_nvp_fixed_affine_v1": "real_nvp_affine_coupling_v1",
    "combined_nsf_real_nvp_fixed_affine_v2": "combined_nsf_real_nvp_v2",
}
SUPPORTED_ARCHITECTURES = set(FIXED_AFFINE_CORES) | set(FIXED_AFFINE_CORES.values())


@dataclass(frozen=True)
class FlowSelection:
    checkpoint: Path | None
    metadata_path: Path | None
    metadata: dict
    target_kind: str
    output_directory: Path

    @property
    def rollout_length(self):
        task = self.metadata["task"]
        return task["way_points"] * task["steps_per_way_point"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--flow", choices=("gaussian", "flat95"),
                        help="Select a combined-flow checkpoint (default: flat95).")
    source.add_argument("--flow-checkpoint", type=Path, help="Use a custom NF checkpoint.")
    parser.add_argument("--flow-metadata", type=Path,
                        help="Defaults to the selected checkpoint's sibling metadata.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to output/tracking/<flow kind>/saved_data.")
    add_training_arguments(parser)
    args = parser.parse_args(argv)
    validate_training_arguments(parser, args)
    return args


def add_training_arguments(parser):
    """Keep PPO budgets and seeds identical across learned and exact samplers."""
    parser.add_argument("--num-iterations", type=int, default=2000)
    parser.add_argument("--vec-env", type=int, default=4096)
    parser.add_argument("--mini-batch-size", type=int, default=16384)
    parser.add_argument("--log-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=8848)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--check-only", action="store_true",
                        help="Check paths and metadata without initializing JAX or training.")


def validate_training_arguments(parser, args):
    if min(args.num_iterations, args.vec_env, args.mini_batch_size, args.log_period) < 1:
        parser.error("Iteration counts, environment count, and minibatch size must be positive")


def select_flow(args):
    """Resolve either NF and validate its sampling contract without running it."""
    selected_kind = args.flow or "flat95"
    default_stem = "trajectory_combined_fixed"
    if selected_kind == "flat95":
        default_stem += "_flat95"
    checkpoint = args.flow_checkpoint or PROJECT_ROOT / "output" / f"{default_stem}_variables.msgpack"
    checkpoint = checkpoint.expanduser().resolve()
    metadata_path = args.flow_metadata or checkpoint.with_name(
        checkpoint.stem.removesuffix("_variables") + "_metadata.json"
    )
    metadata_path = metadata_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"NF checkpoint not found: {checkpoint}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NF metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    encoding = metadata.get("encoding", {})
    if encoding.get("type") != ENCODING_TYPE:
        raise ValueError("Tracking requires an NF trained with physical_positions_velocities_v1 encoding")
    if metadata.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported NF architecture: {metadata.get('architecture')!r}")
    target_kind = metadata.get("target", {}).get("kind", "gaussian")
    if target_kind not in ("gaussian", "flat95"):
        raise ValueError(f"Unsupported cloning target: {target_kind!r}")
    if args.flow is not None and target_kind != args.flow:
        raise ValueError(f"Selected {args.flow}, but the metadata describes {target_kind}")
    task = metadata["task"]
    for name in ("way_points", "steps_per_way_point"):
        if not isinstance(task[name], int) or isinstance(task[name], bool) or task[name] <= 0:
            raise ValueError(f"Task {name} must be a positive integer")
    if task["dt"] <= 0 or task["length_scale"] <= 0:
        raise ValueError("Task dt and length_scale must be positive")
    if metadata["flow"]["dimension"] != 4 * task["way_points"] + 2:
        raise ValueError("NF dimension does not match the task's position-velocity encoding")
    if metadata["architecture"] in FIXED_AFFINE_CORES:
        fixed = metadata["fixed_affine"]
        if fixed.get("core_architecture") != FIXED_AFFINE_CORES[metadata["architecture"]]:
            raise ValueError("Fixed-affine core architecture does not match the checkpoint architecture")
        dimension = metadata["flow"]["dimension"]
        for name in ("matrix", "inverse"):
            matrix = fixed[name]
            if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
                raise ValueError(f"Fixed-affine {name} must match the NF dimension")
        if not math.isfinite(float(fixed["log_abs_det"])):
            raise ValueError("Fixed-affine log determinant must be finite")
    metric_samples = encoding["metric_samples"]
    if not isinstance(metric_samples, int) or metric_samples <= 2 * task["way_points"] + 1:
        raise ValueError("Encoding metric_samples must exceed the per-axis encoding dimension")
    rollout_length = task["way_points"] * task["steps_per_way_point"]
    samples = args.vec_env * rollout_length
    if samples < args.mini_batch_size or samples % args.mini_batch_size:
        raise ValueError(
            f"vec-env * task horizon ({samples}) must be divisible by mini-batch-size "
            f"({args.mini_batch_size})"
        )
    output = args.output_dir or PROJECT_ROOT / "output" / "tracking" / target_kind / "saved_data"
    return FlowSelection(checkpoint, metadata_path, metadata, target_kind,
                         output.expanduser().resolve())


def selection_summary(selection):
    return {
        "flow": selection.target_kind,
        "checkpoint": str(selection.checkpoint) if selection.checkpoint is not None else None,
        "metadata": str(selection.metadata_path) if selection.metadata_path is not None else None,
        "output_directory": str(selection.output_directory),
        "rollout_length": selection.rollout_length,
        "nf_architecture": selection.metadata["architecture"],
        "nf_config": selection.metadata["flow"],
    }


def load_nf(selection):
    import jax
    import jax.numpy as jnp
    from flax import serialization

    metadata = selection.metadata
    flow_settings = dict(metadata["flow"])
    flow_settings["hidden_features"] = tuple(flow_settings["hidden_features"])
    architecture = metadata["architecture"]
    core_architecture = FIXED_AFFINE_CORES.get(architecture, architecture)
    if core_architecture == "combined_nsf_real_nvp_v2":
        from jax_rq_nsf.combined_flow import CombinedFlow as Core, CombinedFlowConfig as Config
    elif core_architecture == "real_nvp_affine_coupling_v1":
        from jax_rq_nsf.real_nvp import RealNVP as Core, RealNVPConfig as Config
    elif core_architecture == "safe_flow_even_permutation_restore_conditional_offset_v6":
        from jax_rq_nsf.safe_flow import NormalizingFlow as Core, FlowConfig as Config
    else:
        raise ValueError(f"Unsupported NF core architecture: {core_architecture!r}")
    core = Core(Config(**flow_settings))
    key = jax.random.PRNGKey(0)
    if architecture in FIXED_AFFINE_CORES:
        from jax_rq_nsf.fixed_affine_flow import FixedAffineState, init_flow

        fixed = metadata["fixed_affine"]
        # Metadata stores already-rounded float32 arrays. Restore the complete
        # state from the checkpoint after constructing its matching template.
        affine = FixedAffineState(
            matrix=jnp.asarray(fixed["matrix"], dtype=jnp.float32),
            inverse_matrix=jnp.asarray(fixed["inverse"], dtype=jnp.float32),
            log_abs_det=jnp.asarray(fixed["log_abs_det"], dtype=jnp.float32),
        )
        flow, template = init_flow(key, core, affine)
    else:
        flow = core
        template = flow.init(key, jnp.zeros((1, flow_settings["dimension"]), dtype=jnp.float32))
    variables = serialization.from_bytes(template, selection.checkpoint.read_bytes())
    variables = jax.tree.map(jnp.asarray, variables)
    for expected, restored in zip(jax.tree.leaves(template), jax.tree.leaves(variables), strict=True):
        if expected.shape != restored.shape:
            raise ValueError("NF checkpoint parameter shapes do not match its metadata")
    return flow, variables, {
        **metadata,
        "source_architecture": metadata["architecture"],
    }


def restore_ppo_training_state(ppo, template, directory):
    """Restore full PPO state, or warm-start saved weights with fresh optimizers."""
    import jax
    import jax.numpy as jnp
    from flax import serialization

    directory = Path(directory)
    state_path = directory / "ppo_training_state.msgpack"
    if state_path.is_file():
        state = serialization.from_bytes(template, state_path.read_bytes())
        mode = "restored_full_training_state"
    else:
        policy = serialization.from_bytes(template.policy_params, (directory / "policy.msgpack").read_bytes())
        critic = serialization.from_bytes(template.critic_params, (directory / "critic.msgpack").read_bytes())
        state = template.replace(
            policy_params=policy, critic_params=critic,
            policy_opt_state=ppo._policy_optimizer.init(policy),
            critic_opt_state=ppo._critic_optimizer.init(critic),
        )
        state = state.replace(policy_opt_state=ppo._update_learning_rate(state))
        mode = "saved_weights_with_fresh_optimizers"
    for expected, restored in zip(jax.tree.leaves(template), jax.tree.leaves(state), strict=True):
        if getattr(expected, "shape", ()) != getattr(restored, "shape", ()):
            raise ValueError("PPO checkpoint shapes do not match the tracking networks")
    return jax.tree.map(jnp.asarray, state), mode


def run_training(args, selection, *, distribution_loader=load_nf,
                 checkpoint_directory=None, distribution_preparer=None):
    # Imports and GPU allocation are deliberately deferred until training starts.
    import jax
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        raise SystemExit("No usable JAX GPU is available; run PPO on the GPU workstation.") from None
    if not devices:
        raise SystemExit("No JAX GPU devices found; PPO was not started.")

    from functools import partial
    import inspect
    import flax.linen as nn
    from flax import serialization
    import jax.numpy as jnp
    import numpy as np
    import wandb
    from brax import envs
    from algorithms.tracking_ppo import PPO, PPOConfigs
    from networks import GCMLP, PPO_Policy
    from task_wrappers.state_independent_tracking import StateIndependentTrackingWrapper, _tracking_reward

    flow, flow_variables, metadata = distribution_loader(selection)
    source_description = {}
    if checkpoint_directory is not None:
        source_description = json.loads((Path(checkpoint_directory) / "description.json").read_text())

    task = metadata["task"]
    env = StateIndependentTrackingWrapper(
        envs.create(env_name="ant", episode_length=4096, backend="mjx", reset_noise_scale=0.0),
        flow, way_points=task["way_points"], steps_per_way_point=task["steps_per_way_point"],
        l=task["length_scale"], dt=task["dt"],
        metric_samples=metadata["encoding"]["metric_samples"],
    )
    ppo_settings = dict(
        policy_learning_rate_per_std=8e-4, critic_learning_rate=5e-4,
        clip_ratio=0.2, entropy_gain=0.001, discount=0.99, gae_lambda=0.95,
        critic_epochs=4, policy_epochs=4,
    )
    for name in ("policy_learning_rate_per_std", "critic_learning_rate", "critic_epochs", "policy_epochs"):
        ppo_settings[name] = source_description.get(name, ppo_settings[name])
    ppo_settings.update(source_description.get("ppo_config", {}))
    ppo_settings.update(rollout_length=selection.rollout_length, vec_env=args.vec_env,
                        mini_batch_size=args.mini_batch_size)
    ppo_config = PPOConfigs(**ppo_settings)
    actor_layers = tuple(source_description.get("actor_hidden_layers", (256, 256)))
    critic_layers = tuple(source_description.get("critic_hidden_layers", (128, 128)))
    policy_network = PPO_Policy(
        hidden_layer_sizes=actor_layers, action_dim=env.action_size,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
        kernel_init_final=jax.nn.initializers.orthogonal(0.01),
        activation=nn.silu, final_activation=jnp.tanh,
    )
    critic_network = GCMLP(
        layer_sizes=critic_layers + (1,),
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
        activation=nn.silu, kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    )
    ppo = PPO(env=env, policy_network=policy_network,
              critic_network=critic_network, ppo_configs=ppo_config)
    loop_key = jax.random.PRNGKey(args.seed)
    loop_key, policy_key, reset_key = jax.random.split(loop_key, 3)
    training_state = ppo.init(policy_key)
    optimizer_initialization = "fresh_training"
    if checkpoint_directory is not None:
        training_state, optimizer_initialization = restore_ppo_training_state(
            ppo, training_state, checkpoint_directory,
        )
    if distribution_preparer is not None:
        flow_variables, metadata = distribution_preparer(
            flow, flow_variables, metadata, env, critic_network, training_state.critic_params,
        )
    initial_states = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, args.vec_env), flow_variables,
    )
    carry = (initial_states, training_state, loop_key)

    def training_iteration(carry, _):
        states, training_state, loop_key = carry
        (final_states, sampled_states, training_state, loop_key), metrics = ppo.train(
            states, training_state, loop_key,
        )
        velocities = jnp.linalg.norm(sampled_states.env_state.obs[:, 13:15], axis=-1)
        # Reset physics while advancing each environment's task RNG.
        reset_states = initial_states.replace(key=final_states.key)
        states = jax.vmap(env.resample_task_state, in_axes=(0, None))(
            reset_states, flow_variables,
        )
        return (states, training_state, loop_key), (metrics, jnp.mean(velocities))

    @partial(jax.jit, static_argnames=("iterations",))
    def train_chunk(carry, iterations):
        return jax.lax.scan(training_iteration, carry, xs=None, length=iterations)

    description = {
        **selection_summary(selection),
        "task": task,
        "encoding": metadata["encoding"],
        "cloning_target": metadata.get("target", {"kind": "gaussian"}),
        "seed": args.seed,
        "num_iterations": args.num_iterations,
        "vec_env": args.vec_env,
        "mini_batch_size": args.mini_batch_size,
        "policy_learning_rate_per_std": ppo_config.policy_learning_rate_per_std,
        "critic_learning_rate": ppo_config.critic_learning_rate,
        "policy_epochs": ppo_config.policy_epochs,
        "critic_epochs": ppo_config.critic_epochs,
        "actor_hidden_layers": list(actor_layers),
        "critic_hidden_layers": list(critic_layers),
        "ppo_config": asdict(ppo_config),
        "optimizer_initialization": optimizer_initialization,
        "tracking_reward_source": inspect.getsource(_tracking_reward),
    }
    if checkpoint_directory is not None:
        description["source_checkpoint_directory"] = str(checkpoint_directory)
    if "filtering" in metadata:
        description["filtering"] = metadata["filtering"]
    directory = selection.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "description.json").write_text(json.dumps(description, indent=2) + "\n")
    (directory / "description.log").write_text(
        "\n".join(f"{name}: {value}" for name, value in description.items()) + "\n"
    )
    # Snapshot the actual sampling distribution, including its fixed affine.
    (directory / "flow_variables.msgpack").write_bytes(serialization.to_bytes(flow_variables))
    snapshot_metadata = {
        **metadata,
        "checkpoint": str(directory / "flow_variables.msgpack"),
    }
    (directory / "flow_metadata.json").write_text(json.dumps(snapshot_metadata, indent=2) + "\n")
    run = wandb.init(
        entity="airl-lab", project="TBQDRL", group="State-independent tracking PPO",
        name=f"tracking-{selection.target_kind}" + ("-filtered" if "filtering" in metadata else ""),
        config=description, mode=args.wandb_mode,
    )
    try:
        completed = 0
        history = []
        while completed < args.num_iterations:
            iterations = min(args.log_period, args.num_iterations - completed)
            carry, (stacked_metrics, velocities) = train_chunk(carry, iterations=iterations)
            completed += iterations
            report = {
                "critic_RMSE": float(jnp.mean(stacked_metrics.critic_rmse)),
                "approx_kl": float(jnp.mean(stacked_metrics.policy_approx_kl)),
                "iteration mean return": float(jnp.mean(stacked_metrics.average_return)),
                "iteration mean reward": float(jnp.mean(stacked_metrics.average_reward)),
                "dones per episode": float(jnp.mean(stacked_metrics.done_count)) / args.vec_env,
                "iteration_mean_v": float(jnp.mean(velocities)),
            }
            history.append({"iteration": completed, **report})
            run.log(report, step=completed)

        final_training_state = carry[1]
        (directory / "policy.msgpack").write_bytes(
            serialization.to_bytes(final_training_state.policy_params)
        )
        (directory / "critic.msgpack").write_bytes(
            serialization.to_bytes(final_training_state.critic_params)
        )
        (directory / "ppo_training_state.msgpack").write_bytes(serialization.to_bytes(final_training_state))
        (directory / "ppo_history.json").write_text(json.dumps(history, indent=2) + "\n")
        np.save(directory / "ppo_rng_key.npy", np.asarray(jax.device_get(carry[2])))
        print(f"Saved {selection.target_kind} tracking policy and critic to {directory}", flush=True)
    finally:
        run.finish()


def main(argv=None):
    args = parse_args(argv)
    try:
        selection = select_flow(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Invalid NF input: {error}") from None
    print(json.dumps(selection_summary(selection), indent=2), flush=True)
    if not args.check_only:
        run_training(args, selection)


if __name__ == "__main__":
    main()
