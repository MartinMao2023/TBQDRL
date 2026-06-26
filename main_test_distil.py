import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import jax
import flax.linen as nn
import jax.numpy as jnp
from brax import envs
from flax import serialization
from typing import Tuple

from custom_types import RNGKey, Params
from networks import GC_GMM_PPO_Policy
from task_wrappers.ant_wrapper import AntWrapper
from data_struct.distillation_transitions import GMMDistillationTransition
from data_struct.states import GeneralizedState
from algorithms.intermediate_bc import GMMDistillationBC, GMMBCConfigs
from data_struct import PPOTransition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEC_ENV        = 4096
ROLLOUT_LENGTH = 128
N_AGG          = 16                                  # rollout iterations per aggregation
FRACTION       = 8                                   # keep 1/FRACTION of each rollout
EIGHTH         = VEC_ENV * ROLLOUT_LENGTH // FRACTION  # 65 536  samples per agg step
TOTAL          = N_AGG * EIGHTH                      # 1 048 576 total after aggregation

MINI_BATCH_SIZE = 4096
MINIBATCH_NUM   = 16   # inner scan steps  (gradient updates per bc-iteration)
BC_ITERATIONS   = TOTAL // MINI_BATCH_SIZE // MINIBATCH_NUM   # outer scan steps  (each emits one EMA loss value)
# sanity: BC_ITERATIONS * MINIBATCH_NUM * MINI_BATCH_SIZE == TOTAL  →  16*16*4096 == 1 048 576 ✓

DEMO_SIZE   = VEC_ENV * ROLLOUT_LENGTH   # 524 288  (full rollout, no fraction)
DEMO_REPEAT = math.ceil(TOTAL / DEMO_SIZE)  # tiles needed to reach TOTAL



actor_hidden_layers: Tuple[int, ...] = (128, 128)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
env = AntWrapper(env)

# ---------------------------------------------------------------------------
# Teacher network — seeds must match main_ant_GMM.py / test_GMM.ipynb exactly.
# component_means are a network *attribute* (not saved in params), so the same
# PRNGKey must be used to reproduce them.
# ---------------------------------------------------------------------------
seed = 6666
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
component_means = jnp.concatenate([
    jnp.zeros(env.action_size),
    jax.random.normal(subkey, shape=(3 * env.action_size)) * 0.25,
])

teacher_network = GC_GMM_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=4,
    component_means=component_means,
)

seed = 4242
random_key = jax.random.PRNGKey(seed)
fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs  = jnp.zeros(shape=(env.z_size,))
policy_template = teacher_network.init(random_key, obs=fake_obs, z=fake_zs)

with open("output/GMM/policy.msgpack", "rb") as f:
    teacher_params = serialization.from_bytes(policy_template, f.read())

teacher_std_logits = teacher_params["params"]["std_logits"]
teacher_std = jax.nn.sigmoid(teacher_std_logits)
print(f"Teacher std: {teacher_std}")

# ---------------------------------------------------------------------------
# Student network  (MoE, twice hidden size and mixture components as teacher)
# ---------------------------------------------------------------------------
loop_random_key, subkey = jax.random.split(loop_random_key)
component_means = jnp.concatenate([
    jnp.zeros(env.action_size),
    jax.random.normal(subkey, shape=(7 * env.action_size)) * 0.25,
])
student_hidden_layers = (256, 256)
student_network = GC_GMM_PPO_Policy(
    hidden_layer_sizes=student_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=8,
    component_means=component_means,
)


bc_configs = GMMBCConfigs(
    learning_rate=1e-3,
    bc_epochs=1,          # unused by state_update_scan; kept for API compat
    mini_batch_size=MINI_BATCH_SIZE,
    num_mini_batches=MINIBATCH_NUM,   # sets ema_alpha for the inner scan
)

bc = GMMDistillationBC(
    env=env,
    policy_network=student_network,
    teacher_std_logits=teacher_std_logits,
    bc_configs=bc_configs,
)

random_key, subkey = jax.random.split(random_key)
bc_training_state = bc.init(subkey)


@jax.jit
def aggregate_data(
    policy_params: Params,
    starting_states: GeneralizedState,
    key: RNGKey,
) -> Tuple[GeneralizedState, RNGKey, GMMDistillationTransition, jax.Array]:

    def agg_step(
        carry: Tuple[GeneralizedState, RNGKey],
        _,
    ) -> Tuple[Tuple, Tuple[GMMDistillationTransition, jax.Array]]:
        states, key = carry

        # Per-environment rollout step
        def play_step_fn(
            carry: Tuple[GeneralizedState, RNGKey, jax.Array, jax.Array],
        ) -> Tuple[Tuple, GMMDistillationTransition]:
            state, step_key, cum_r, survive = carry

            obs, z = env.get_obs(state)
            action_means, weight_logits, _ = teacher_network.apply(
                policy_params, obs, z
            )
            # action_std = nn.sigmoid(std_logits) * 2
            action_std = teacher_std

            component_weights = nn.softmax(weight_logits) # (k,)
            step_key, subkey = jax.random.split(step_key)
            action_mean = jax.random.choice(subkey, a=action_means, p=component_weights)

            # action_mean = jnp.sum(action_means * component_weights[:, None], axis=0)

            step_key, subkey = jax.random.split(step_key)
            noise  = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + noise, -1.0, 1.0)
            # action = jnp.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

            state, tinfo = env.step(state, action)

            cum_r = cum_r + survive * tinfo.reward
            survive = survive * (1.0 - tinfo.done)

            transition = GMMDistillationTransition(
                obs=obs,
                zs=z,
                action_means=action_means,       # (k, action_dim)
                component_logits=weight_logits,  # (k,)
            )
            return (state, step_key, cum_r, survive), transition

        key, subkey = jax.random.split(key)
        rollout_keys = jax.random.split(subkey, num=VEC_ENV)

        (final_states, _, cum_rewards, _), transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (states, rollout_keys, jnp.zeros((VEC_ENV, 1)), jnp.ones((VEC_ENV, 1))),
            length=ROLLOUT_LENGTH,
        )
        # transitions fields: (ROLLOUT_LENGTH, VEC_ENV, ...)

        # Shuffle across the full (ROLLOUT_LENGTH × VEC_ENV) pool, take 1/8.
        key, subkey = jax.random.split(key)
        transitions = transitions.shuffle(subkey)            # (ROLLOUT_LENGTH*VEC_ENV, ...)
        transitions = jax.tree.map(lambda x: x[:EIGHTH], transitions)  # (EIGHTH, ...)

        return (final_states, key), (transitions, cum_rewards)

    (final_states, final_key), (agg_transitions, all_cum_rewards) = jax.lax.scan(
        agg_step,
        (starting_states, key),
        length=N_AGG,
    )
    # agg_transitions fields: (N_AGG, EIGHTH, ...)
    # all_cum_rewards:         (N_AGG, VEC_ENV)
    return final_states, final_key, agg_transitions, all_cum_rewards


@jax.jit
def collect_demonstrations(
    policy_params: Params,
    starting_states: GeneralizedState,
    key: RNGKey,
) -> Tuple[GeneralizedState, RNGKey, PPOTransition]:
    """Collect pseudo-expert demonstrations by perturbing the teacher with O-U noise.

    The teacher selects an action (component sample + Gaussian noise) and then a
    correlated residual is added via an Ornstein-Uhlenbeck process:

        x_{t+1} = 0.9 * x_t + 0.25 * N(0, I)

    The O-U state is initialised at zero and carried across the full rollout
    without resetting at episode boundaries.

    Only obs, zs, actions, dones, and truncations are meaningful in the returned
    PPOTransition; all other scalar fields are zero-filled.

    Returns:
        (final_states, key, transitions) where transitions has flat leading
        shape (DEMO_SIZE, ...) = (VEC_ENV * ROLLOUT_LENGTH, ...).
    """

    def play_step_fn(
        carry: Tuple[GeneralizedState, RNGKey, jax.Array, jax.Array, jax.Array],
    ) -> Tuple[Tuple, PPOTransition]:
        state, step_key, cum_r, survive, ou_noise = carry

        obs, z = env.get_obs(state)
        action_means, weight_logits, _ = teacher_network.apply(policy_params, obs, z)

        component_weights = nn.softmax(weight_logits)
        step_key, subkey = jax.random.split(step_key)
        action_mean = jax.random.choice(subkey, a=action_means, p=component_weights)

        step_key, subkey = jax.random.split(step_key)
        # ou_noise = 0.9 * ou_noise + 0.109 * jax.random.normal(subkey, ou_noise.shape)
        ou_noise = 0.975 * ou_noise + 0.04444 * jax.random.normal(subkey, ou_noise.shape)

        action = jnp.clip(action_mean + ou_noise, -1.0, 1.0)
        state, tinfo = env.step(state, action)

        cum_r   = cum_r   + survive * tinfo.reward
        survive = survive * (1.0 - tinfo.done)

        transition = PPOTransition(
            obs=obs,
            actions=action,
            zs=z,
            log_likelihood=jnp.zeros((1,)),
            rewards=jnp.zeros((1,)),
            td_lambda_returns=jnp.zeros((1,)),
            gaes=jnp.zeros((1,)),
            dones=tinfo.done,
            truncations=tinfo.truncation,
            weights=jnp.zeros((1,)),
        )

        return (state, step_key, cum_r, survive, ou_noise), transition

    key, subkey = jax.random.split(key)
    rollout_keys = jax.random.split(subkey, num=VEC_ENV)

    (final_states, _, _, _, _), transitions = jax.lax.scan(
        lambda x, _: jax.vmap(play_step_fn)(x),
        (
            starting_states,
            rollout_keys,
            jnp.zeros((VEC_ENV, 1)),
            jnp.ones((VEC_ENV, 1)),
            jnp.zeros((VEC_ENV, env.action_size)),  # O-U state initialised at 0
        ),
        length=ROLLOUT_LENGTH,
    )
    # transitions: (ROLLOUT_LENGTH, VEC_ENV, ...) → (DEMO_SIZE, ...)
    transitions = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1).reshape(DEMO_SIZE, *x.shape[2:]),
        transitions,
    )
    return final_states, key, transitions


# ---------------------------------------------------------------------------
# Initial env reset  (seed 114514 matches main_ant_GMM.py)
# ---------------------------------------------------------------------------
loop_random_key = jax.random.PRNGKey(114514)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=VEC_ENV)
states = jax.vmap(env.reset)(subkeys)



@jax.jit
def scan_bc(
    carry: Tuple,
    data: Tuple[GMMDistillationTransition, PPOTransition],
) -> Tuple[Tuple, jax.Array]:
    training_state, key = carry
    key, subkey = jax.random.split(key)
    new_training_state, metrics = bc.state_update(training_state, data, subkey)
    carry = (new_training_state, key)
    return carry, metrics.bc_loss  # (3,): [kl, nll, average_diff]


# ------------------------------------------------------------------
# Stage 1: aggregate teacher rollouts into one large dataset
# ------------------------------------------------------------------
states, loop_random_key, agg_transitions, all_cum_rewards = aggregate_data(
    teacher_params, states, loop_random_key
)
teacher_avg_reward = jnp.mean(all_cum_rewards) / ROLLOUT_LENGTH

mixed = jax.tree.map(
    lambda x: jnp.swapaxes(x, 0, 1).reshape(TOTAL, *x.shape[2:]),
    agg_transitions,
)
batched = jax.tree.map(
    lambda x: x.reshape(BC_ITERATIONS, MINIBATCH_NUM, MINI_BATCH_SIZE, *x.shape[1:]),
    mixed,
)

# ------------------------------------------------------------------
# Stage 2: collect pseudo-expert demonstrations (O-U perturbed teacher)
# and tile to match the teacher buffer size.
# ------------------------------------------------------------------
states, loop_random_key, demo_transitions = collect_demonstrations(
    teacher_params, states, loop_random_key
)

# Shuffle then tile DEMO_REPEAT times, keep exactly TOTAL rows.
loop_random_key, subkey = jax.random.split(loop_random_key)
demo_transitions = demo_transitions.shuffle(subkey)
demo_tiled = jax.tree.map(
    lambda x: jnp.tile(x, (DEMO_REPEAT,) + (1,) * (x.ndim - 1))[:TOTAL],
    demo_transitions,
)
demo_batched = jax.tree.map(
    lambda x: x.reshape(BC_ITERATIONS, MINIBATCH_NUM, MINI_BATCH_SIZE, *x.shape[1:]),
    demo_tiled,
)

batched_combined = (batched, demo_batched)

# ------------------------------------------------------------------
# Stage 3: BC training over the combined dataset.
#   • outer scan → BC_ITERATIONS steps, one EMA loss vector per step
#   • inner scan → MINIBATCH_NUM gradient updates per step
# ------------------------------------------------------------------
print(f"teacher_avg_reward = {float(teacher_avg_reward):.4f}")


for i in range(512):
    (bc_training_state, loop_random_key), losses = jax.lax.scan(
        scan_bc,
        (bc_training_state, loop_random_key),
        batched_combined,
    )
    # losses: (BC_ITERATIONS, 3)  — last row is the most recent EMA values
    print(
        f"KL = {float(losses[-1, 0]):.6f}  "
        f"NLL = {float(losses[-1, 1]):.6f}  "
        f"weight_diff = {float(losses[-1, 2]):.6f}"
    )



@jax.jit
def teacher_rollout_fn(
    teacher_params: Params,
    starting_states: GeneralizedState,
    keys: RNGKey,
) -> jax.Array:
    """Returns cumulative rewards per env over one rollout with the teacher."""

    def play_step_fn(
        carry: Tuple[GeneralizedState, RNGKey, jax.Array, jax.Array],
    ) -> Tuple[Tuple, None]:
        state, key, cum_r, survive = carry

        obs, z = env.get_obs(state)
        action_means, weight_logits, std_logits = teacher_network.apply(teacher_params, obs, z)
        component_weights = nn.softmax(weight_logits) # (k,)

        key, subkey = jax.random.split(key)
        action_mean = jax.random.choice(subkey, a=action_means, p=component_weights)

        key, subkey = jax.random.split(key)
        noise  = teacher_std * jax.random.normal(subkey, action_mean.shape)
        action = jnp.clip(action_mean + noise, -1.0, 1.0)

        state, tinfo = env.step(state, action)

        # cum_r   = cum_r   + survive * tinfo.reward
        cum_r   = cum_r  +  tinfo.reward
        survive = survive * (1.0 - tinfo.done)

        return (state, key, cum_r, survive), None

    (_, _, cum_rewards, _), _ = jax.lax.scan(
        lambda x, _: jax.vmap(play_step_fn)(x),
        (starting_states, keys, jnp.zeros((VEC_ENV, 1)), jnp.ones((VEC_ENV, 1))),
        length=1024,
    )
    return cum_rewards


@jax.jit
def student_rollout_fn(
    student_params: Params,
    starting_states: GeneralizedState,
    keys: RNGKey,
) -> jax.Array:
    """Returns cumulative rewards per env over one rollout with the teacher."""

    def play_step_fn(
        carry: Tuple[GeneralizedState, RNGKey, jax.Array, jax.Array],
    ) -> Tuple[Tuple, None]:
        state, key, cum_r, survive = carry

        obs, z = env.get_obs(state)
        action_means, weight_logits, _ = student_network.apply(student_params, obs, z)
        component_weights = nn.softmax(weight_logits[:4]) # (k,)

        key, subkey = jax.random.split(key)
        action_mean = jax.random.choice(subkey, a=action_means[:4], p=component_weights)

        key, subkey = jax.random.split(key)
        noise  = teacher_std * jax.random.normal(subkey, action_mean.shape)
        action = jnp.clip(action_mean + noise, -1.0, 1.0)

        state, tinfo = env.step(state, action)

        # cum_r   = cum_r   + survive * tinfo.reward
        cum_r   = cum_r  +  tinfo.reward
        survive = survive * (1.0 - tinfo.done)

        return (state, key, cum_r, survive), None

    (_, _, cum_rewards, _), _ = jax.lax.scan(
        lambda x, _: jax.vmap(play_step_fn)(x),
        (starting_states, keys, jnp.zeros((VEC_ENV, 1)), jnp.ones((VEC_ENV, 1))),
        length=1024,
    )
    return cum_rewards


print("\n" + "=" * 60)
print("Distillation complete. Evaluating student policy ...")

loop_random_key, subkey = jax.random.split(loop_random_key)
eval_keys = jax.random.split(subkey, num=VEC_ENV)

# Fresh env reset for a clean evaluation episode.
loop_random_key, subkey = jax.random.split(loop_random_key)
eval_subkeys = jax.random.split(subkey, num=VEC_ENV)
eval_states = jax.vmap(env.reset)(eval_subkeys)

student_cum_rewards = student_rollout_fn(
    bc_training_state.policy_params, eval_states, eval_keys
)
student_avg_reward = float(jnp.mean(student_cum_rewards) / 1024)

# Re-use the teacher on the same starting states for a fair comparison.
# _, _, _, teacher_cum_rewards = aggregate_data(
#     teacher_params, eval_states, loop_random_key
# )
teacher_cum_rewards = teacher_rollout_fn(
    teacher_params, eval_states, eval_keys
)
teacher_eval_avg_reward = float(jnp.mean(teacher_cum_rewards) / 1024)

print(f"  Teacher avg reward : {teacher_eval_avg_reward:.4f}")
print(f"  Student avg reward : {student_avg_reward:.4f}")
print(f"  Ratio  (student/teacher) : {student_avg_reward / (teacher_eval_avg_reward + 1e-8):.3f}")

print("=" * 60)

# ------------------------------------------------------------------
# Finally: Save the student policy network parameters.
# ------------------------------------------------------------------
print("Saving student policy")

model_bytes = serialization.to_bytes(bc_training_state.policy_params)
folder_path = f"./output/GMM"

with open(folder_path + f"/student_policy.msgpack", "wb") as f:
    f.write(model_bytes)

print("=" * 60)



