"""Collect policy trajectories for the invertible-mapping experiment.

Each buffer row contains the static flow sample supplied to the policy, its
log density under q, the 96 realized robot x-y positions, and a validity scale
that is zero at and after the first tracking failure or environment
termination.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from flax import serialization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jax_rq_nsf import FlowConfig, NormalizingFlow, init_flow
from networks import PPO_Policy
from task_wrappers.nf_trajectory_following import AntFiniteMaternWrapper


DEFAULT_FLOW_CHECKPOINT = (
    PROJECT_ROOT / "output/vi_filtered/trajectory_flow_variables.msgpack"
)
DEFAULT_POLICY_CHECKPOINT = (
    PROJECT_ROOT / "output/vi_filtered_ppo/saved_data/policy.msgpack"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output/mapping_test/trajectory_buffer.npz"
)

WAY_POINTS = 12
STEPS_PER_WAY_POINT = 8
ROLLOUT_LENGTH = WAY_POINTS * STEPS_PER_WAY_POINT
TASK_DIMENSION = 4 * WAY_POINTS + 2

FLOW_CONFIG = FlowConfig(
    dimension=TASK_DIMENSION,
    num_layers=8,
    hidden_features=(128, 128),
    num_bins=16,
    tail_bound=6.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=DEFAULT_FLOW_CHECKPOINT,
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=DEFAULT_POLICY_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-iterations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8848)
    parser.add_argument("--log-period", type=int, default=8)
    parser.add_argument(
        "--deterministic-actions",
        action="store_true",
        help="Use policy means instead of PPO-style stochastic actions.",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.num_iterations < 1:
        parser.error("--num-iterations must be positive")
    if args.log_period < 1:
        parser.error("--log-period must be positive")
    return args


def load_flow(checkpoint_path: Path) -> tuple[NormalizingFlow, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"flow checkpoint not found: {checkpoint_path}")
    flow, variables_template = init_flow(jax.random.PRNGKey(0), FLOW_CONFIG)
    variables = serialization.from_bytes(
        variables_template,
        checkpoint_path.read_bytes(),
    )
    return flow, variables


def create_environment(flow: NormalizingFlow) -> AntFiniteMaternWrapper:
    base_env = envs.create(
        env_name="ant",
        episode_length=4096,
        backend="mjx",
        reset_noise_scale=0.0,
    )
    return AntFiniteMaternWrapper(
        base_env,
        flow,
        way_points=WAY_POINTS,
        steps_per_way_point=STEPS_PER_WAY_POINT,
        inner_radius=0.5,
        max_radius=2.0,
        tolerance_radius=0.1,
    )


def load_policy(
    checkpoint_path: Path,
    env: AntFiniteMaternWrapper,
) -> tuple[PPO_Policy, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {checkpoint_path}")
    policy = PPO_Policy(
        hidden_layer_sizes=(256, 256),
        action_dim=env.action_size,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        kernel_init_final=jax.nn.initializers.orthogonal(0.01),
        activation=nn.silu,
        final_activation=jnp.tanh,
    )
    variables_template = policy.init(
        jax.random.PRNGKey(1),
        obs=jnp.zeros((env.observation_size,)),
        z=jnp.zeros((env.z_size,)),
    )
    variables = serialization.from_bytes(
        variables_template,
        checkpoint_path.read_bytes(),
    )
    return policy, variables


def create_batch_collector(
    *,
    env: AntFiniteMaternWrapper,
    flow: NormalizingFlow,
    flow_variables: Any,
    policy: PPO_Policy,
    policy_variables: Any,
    num_envs: int,
    deterministic_actions: bool,
) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    std_logits = policy_variables["params"]["std_logits"]
    action_std = nn.sigmoid(std_logits)

    def collect_batch(key: jax.Array) -> tuple[jax.Array, ...]:
        reset_key, action_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        states = jax.vmap(env.reset, in_axes=(0, None))(
            reset_keys,
            flow_variables,
        )
        initial_xy = states.env_state.pipeline_state.x.pos[:, 0, :2]

        # The first five entries are dynamic conditioning features.  The
        # remaining 50 entries are the static q sample used by this test.
        z = states.z_state.z[:, 5:]
        log_q = flow.apply(
            flow_variables,
            z,
            method=flow.log_prob,
        )[..., None]

        def rollout_step(
            carry: tuple[Any, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, jax.Array, jax.Array], tuple[jax.Array, ...]]:
            current_states, alive, step_key = carry
            observations, conditioning = env.get_obs(current_states)
            action_mean, _ = policy.apply(
                policy_variables,
                observations,
                conditioning,
            )
            step_key, noise_key = jax.random.split(step_key)
            if deterministic_actions:
                actions = action_mean
            else:
                actions = action_mean + action_std * jax.random.normal(
                    noise_key,
                    action_mean.shape,
                )

            next_states, transition_info = jax.vmap(env.step)(
                current_states,
                jnp.clip(actions, -1.0, 1.0),
            )
            terminated = transition_info.done[..., 0] > 0.5
            alive = alive & ~terminated
            xy = (
                next_states.env_state.pipeline_state.x.pos[:, 0, :2]
                - initial_xy
            )
            scale = alive.astype(xy.dtype)[..., None]
            return (next_states, alive, step_key), (xy, scale)

        def rollout_period(
            carry: tuple[Any, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, jax.Array, jax.Array], tuple[jax.Array, ...]]:
            carry, step_data = jax.lax.scan(
                rollout_step,
                carry,
                xs=None,
                length=STEPS_PER_WAY_POINT,
            )
            current_states, alive, step_key = carry
            current_states = jax.vmap(env.shift)(current_states)
            return (current_states, alive, step_key), step_data

        initial_alive = jnp.ones((num_envs,), dtype=jnp.bool_)
        _, (xy_blocks, scale_blocks) = jax.lax.scan(
            rollout_period,
            (states, initial_alive, action_key),
            xs=None,
            length=WAY_POINTS,
        )

        # Convert (waypoint, step, environment, ...) to
        # (environment, rollout_step, ...).
        xy = xy_blocks.reshape(ROLLOUT_LENGTH, num_envs, 2)
        xy = jnp.swapaxes(xy, 0, 1)
        scale = scale_blocks.reshape(ROLLOUT_LENGTH, num_envs, 1)
        scale = jnp.swapaxes(scale, 0, 1)
        return z, log_q, xy, scale

    return jax.jit(collect_batch)


def allocate_buffer(
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((num_samples, TASK_DIMENSION), dtype=np.float32),
        np.empty((num_samples, 1), dtype=np.float32),
        np.empty((num_samples, ROLLOUT_LENGTH, 2), dtype=np.float32),
        np.empty((num_samples, ROLLOUT_LENGTH, 1), dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    flow_checkpoint = args.flow_checkpoint.resolve()
    policy_checkpoint = args.policy_checkpoint.resolve()
    output_path = args.output.resolve()

    flow, flow_variables = load_flow(flow_checkpoint)
    env = create_environment(flow)
    policy, policy_variables = load_policy(policy_checkpoint, env)
    collect_batch = create_batch_collector(
        env=env,
        flow=flow,
        flow_variables=flow_variables,
        policy=policy,
        policy_variables=policy_variables,
        num_envs=args.num_envs,
        deterministic_actions=args.deterministic_actions,
    )

    num_samples = args.num_envs * args.num_iterations
    z_buffer, log_q_buffer, xy_buffer, scale_buffer = allocate_buffer(
        num_samples
    )
    buffer_size_gib = sum(
        array.nbytes
        for array in (z_buffer, log_q_buffer, xy_buffer, scale_buffer)
    ) / 1024**3

    print("JAX devices:", jax.devices())
    print("Flow checkpoint:", flow_checkpoint)
    print("Policy checkpoint:", policy_checkpoint)
    print("Output:", output_path)
    print(
        f"Collecting {num_samples:,} trajectories "
        f"({args.num_iterations} x {args.num_envs})"
    )
    print(f"In-memory buffer size: {buffer_size_gib:.2f} GiB")
    print(
        "Action mode:",
        "deterministic" if args.deterministic_actions else "stochastic",
    )

    key = jax.random.PRNGKey(args.seed)
    start_time = time.monotonic()
    for iteration in range(args.num_iterations):
        key, batch_key = jax.random.split(key)
        z, log_q, xy, scale = jax.device_get(collect_batch(batch_key))
        start = iteration * args.num_envs
        stop = start + args.num_envs
        z_buffer[start:stop] = z
        log_q_buffer[start:stop] = log_q
        xy_buffer[start:stop] = xy
        scale_buffer[start:stop] = scale

        completed = iteration + 1
        if completed == 1 or completed % args.log_period == 0:
            elapsed = time.monotonic() - start_time
            valid_fraction = float(np.mean(scale))
            print(
                f"iteration {completed}/{args.num_iterations}: "
                f"{elapsed:.1f}s elapsed, "
                f"batch valid-step fraction={valid_fraction:.4f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        z=z_buffer,
        log_q=log_q_buffer,
        xy=xy_buffer,
        scale=scale_buffer,
    )
    print("Saved trajectory buffer:", output_path)
    print("z:", z_buffer.shape)
    print("log_q:", log_q_buffer.shape)
    print("xy:", xy_buffer.shape)
    print("scale:", scale_buffer.shape)


if __name__ == "__main__":
    main()
