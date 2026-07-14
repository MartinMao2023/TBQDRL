"""
TBQDRL unified test pipeline.

Stage 0  Load pre-trained Gaussian PPO teacher (policy + critic) and define
         the GMM student policy.
Stage 1  Collect rollouts with the teacher into a NEW buffer class. The
         collected data feeds three downstream uses: distilling the teacher,
         (TBD) fine-tuning the critic, and computing the relocation advantage.
Stage 2  (Optional) Apply dropout on the pre-trained teacher policy to obtain
         more diverse trajectories for distillation and relocation.
Stage 3  Relocation optimization: maximize the relocation advantage and select
         which segments to relocate. Implemented in a separate new file
         (placeholder for now).
Stage 4  Distill both the teacher policy and the relocated trajectories into
         the GMM student, building on top of GMMDistillationBC. The teacher
         term uses KL divergence (already implemented in GMMDistillationBC),
         not BC-NLL.
Stage 5  Continue PPO on the distilled GMM policy and log the learning curve
         to WandB.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from datetime import datetime
from brax import envs
from flax import serialization

from typing import Tuple
from custom_types import Params, RNGKey
from networks import GC_PPO_Policy, GC_PPO_Policy_Dropout, GC_GMM_PPO_Policy, GCMLP
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from tools import (
    build_on_policy_rollout,
    build_dropout_rollout,
    with_policy_action_mean,
)
from algorithms.relocation import relocate
from data_struct.states import GeneralizedState
from data_struct import (
    MORelocationTransition,
    GMMDistillationTransition,
    PPOTransition,
)

# Teacher / student PPO and distillation building blocks.
from algorithms.gmm_ppo import PPO as GMMPPO, PPOConfigs as GMMPPOConfigs
from algorithms.intermediate_bc import GMMDistillationBC, GMMBCConfigs







# NOTE: Stage 3 lives in a NEW file; import the relocation optimizer here once
# it exists, e.g.
#   from algorithms.<relocation_module> import <relocate_fn>


# ===========================================================================
# Configuration
# ===========================================================================
# TODO: central hyperparameters (seeds, vec_env, rollout length, distillation
# batching, PPO iters, checkpoint paths, wandb group/project).

# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 1e-3 # unified
critic_learning_rate = 5e-4
ppo_rollout_length = 32
# collection configs
rollout_length = 64
num_collect_iterations = 4
# Stage 2 (optional) dropout-policy diversity configs
use_dropout_rollout = True
dropout_rate = 0.1
num_dropout_iterations = 4
# Stage 3 relocation configs
relocation_config = {
    # TODO: relocation hyperparameters (candidate preferences, advantage
    # threshold, segment length, etc.)
}
# Stage 4 distillation configs
bc_mini_batch_size = 4096
bc_num_mini_batches = 16
num_bc_epochs = 200


env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
env = AntMOWrapper(env)

critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
policy_network = GC_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.softplus,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
)

seed = 42
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

with open("output/MORL/test/policy.msgpack", "rb") as f:
    encoded_bytes = f.read()

fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs = jnp.zeros(shape=(env.z_size,))

policy_template = policy_network.init(subkey, obs=fake_obs, z=fake_zs)
policy_params = serialization.from_bytes(policy_template, encoded_bytes)

with open("output/MORL/test/critic.msgpack", "rb") as f:
    encoded_bytes = f.read()

critic_template = critic_network.init(subkey, obs=fake_obs, z=fake_zs)
critic_params = serialization.from_bytes(critic_template, encoded_bytes)

moving_mean = jnp.load("output/MORL/test/mean.npy")
moving_std = jnp.sqrt(jnp.load("output/MORL/test/var.npy"))

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

teacher_std_logits = policy_params["params"]["std_logits"]
teacher_std = jax.nn.sigmoid(teacher_std_logits)


on_policy_rollout_fn = build_on_policy_rollout(
    env=env,
    policy_network=policy_network,
    policy_params=policy_params,
    rollout_length=rollout_length,
)


@jax.jit
def collect_on_policy_data(starting_states, key):
    """Outer scan: run a rollout then resample the preference per env, repeat.

    Returns (final_states, key, all_transitions, final_info) where final_info is
    a stacked tuple (final_obs, final_last_actions) of shape
    (num_collect_iterations, vec_env, ...) recorded at the end of each rollout
    (before the preference resample) — needed downstream by `relocate`.
    """

    def outer_step(carry, _):
        states, key = carry
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=vec_env)
        final_states, transitions = on_policy_rollout_fn(states, subkeys)
        # final obs / last_action of this rollout (before preference resample)
        final_obs, _ = jax.vmap(env.get_obs)(final_states)
        final_last_actions = final_states.z_state.last_action
        final_info = (final_obs, final_last_actions)
        # resample preference for the next iteration (keeps last_action)
        final_states = jax.vmap(env.resample_task_state)(final_states)
        return (final_states, key), (transitions, final_info)

    (final_states, key), (all_transitions, all_final_info) = jax.lax.scan(
        outer_step,
        (starting_states, key),
        length=num_collect_iterations,
    )
    return final_states, key, all_transitions, all_final_info


loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)

states, loop_random_key, on_policy_transitions, on_policy_final_info = (
    collect_on_policy_data(states, loop_random_key)
)
# print(on_policy_transitions.mo_rewards.shape)

# on_policy_transitions fields: (num_collect_iterations, rollout_length, vec_env, ...)
# on_policy_final_info: (final_obs, final_last_actions), each
#   (num_collect_iterations, vec_env, ...)

dropout_policy_network = GC_PPO_Policy_Dropout(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    dropout_rate=dropout_rate,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
)

dropout_rollout_fn = build_dropout_rollout(
    env=env,
    policy_network=dropout_policy_network,
    policy_params=policy_params,
    rollout_length=rollout_length,
)


@jax.jit
def collect_dropout_data(starting_states, key):
    def outer_step(carry, _):
        states, key = carry
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=vec_env)
        final_states, transitions = dropout_rollout_fn(states, subkeys)
        final_obs, _ = jax.vmap(env.get_obs)(final_states)
        final_last_actions = final_states.z_state.last_action
        final_info = (final_obs, final_last_actions)
        final_states = jax.vmap(env.resample_task_state)(final_states)
        return (final_states, key), (transitions, final_info)

    (final_states, key), (all_transitions, all_final_info) = jax.lax.scan(
        outer_step,
        (starting_states, key),
        length=num_dropout_iterations,
    )
    return final_states, key, all_transitions, all_final_info




if use_dropout_rollout:
    loop_random_key, subkey = jax.random.split(loop_random_key)
    # subkeys = jax.random.split(subkey, num=vec_env)
    # dropout_start_states = jax.vmap(env.reset)(subkeys)

    states, loop_random_key, dropout_transitions, dropout_final_info = (
        collect_dropout_data(states, loop_random_key)
    )
    # print(dropout_transitions.mo_rewards.shape)


if use_dropout_rollout:
    combined_transitions = jax.tree.map(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        on_policy_transitions, dropout_transitions,
    ) # (8, 64, 4096, ...)
    combined_final_info = (
        jnp.concatenate(
            [on_policy_final_info[0], dropout_final_info[0]], axis=0
        ),  # final_obs
        jnp.concatenate(
            [on_policy_final_info[1], dropout_final_info[1]], axis=0
        ),  # final_last_actions
    )
    
else:
    combined_transitions = on_policy_transitions
    combined_final_info = on_policy_final_info

print(combined_transitions.obs.shape)
print(combined_final_info[0].shape, combined_final_info[1].shape)


# distillation dataset: copy with teacher action mean, then shuffle (i.i.d.)
distillation_transitions = with_policy_action_mean(
    combined_transitions, policy_network, policy_params,
)
loop_random_key, subkey = jax.random.split(loop_random_key)
distillation_transitions = distillation_transitions.shuffle(subkey)
# distillation_transitions fields: (N_distil, ...)


loop_random_key, subkey = jax.random.split(loop_random_key)
relocated_demonstrations = relocate(
    combined_transitions,
    combined_final_info,
    critic_network=critic_network,
    critic_params=critic_params,
    config=relocation_config,
    key=subkey,
)



_distil_batch = distillation_transitions.obs.shape[:-1]
gmm_distillation_transitions = GMMDistillationTransition(
    obs=distillation_transitions.obs,
    zs=distillation_transitions.zs,
    action_means=distillation_transitions.actions[..., None, :],  # (..., 1, d)
    component_logits=jnp.zeros((*_distil_batch, 1)),               # (..., 1) -> softmax 1
)


loop_random_key, subkey = jax.random.split(loop_random_key)


demo_flat = relocated_demonstrations.shuffle(subkey)
gmm_flat = gmm_distillation_transitions


_bc_chunk = bc_num_mini_batches * bc_mini_batch_size
_n_common = min(gmm_flat.obs.shape[0], demo_flat.obs.shape[0])
bc_iterations = _n_common // _bc_chunk
_n_used = bc_iterations * _bc_chunk

def _batch(x):
    return x[:_n_used].reshape(
        bc_iterations, bc_num_mini_batches, bc_mini_batch_size, *x.shape[1:]
    )

gmm_batched = jax.tree.map(_batch, gmm_flat)
demo_batched = jax.tree.map(_batch, demo_flat)
batched_combined = (gmm_batched, demo_batched)

# """
# --- Trainer ---------------------------------------------------------------
bc_configs = GMMBCConfigs(
    learning_rate=1e-3,
    bc_epochs=1,
    mini_batch_size=bc_mini_batch_size,
    num_mini_batches=bc_num_mini_batches,
)
bc = GMMDistillationBC(
    env=env,
    policy_network=student_network,
    teacher_std_logits=teacher_std_logits,
    bc_configs=bc_configs,
    k=4,
)
loop_random_key, subkey = jax.random.split(loop_random_key)
bc_training_state = bc.init(subkey)


@jax.jit
def scan_bc(carry, data):
    training_state, key = carry
    key, subkey = jax.random.split(key)
    new_training_state, metrics = bc.state_update(training_state, data, subkey)
    return (new_training_state, key), metrics.bc_loss  # (3,): [kl, nll, average_diff]



print(
    f"BC: {_n_used} samples | {bc_iterations} iter/epoch | "
    f"{bc_num_mini_batches} mini-batches x {bc_mini_batch_size}"
)
for i in range(num_bc_epochs):
    (bc_training_state, loop_random_key), losses = jax.lax.scan(
        scan_bc,
        (bc_training_state, loop_random_key),
        batched_combined,
    )
    print(
        f"[BC epoch {i + 1:3d}]  "
        f"KL = {float(losses[-1, 0]):.6f}  "
        f"NLL = {float(losses[-1, 1]):.6f}  "
        f"weight_diff = {float(losses[-1, 2]):.6f}"
    )


wandb_config = {
    "task": "relocation PPO test",
    "vec_env": vec_env,
    "mini_batch_size": mini_batch_size,
    "num_iterations": num_iterations,
    "policy_epochs": policy_epochs,
    "critic_epochs": critic_epochs,
    "policy_learning_rate_per_std": policy_learning_rate_per_std,
    "critic_learning_rate": critic_learning_rate,
    "ppo_rollout_length": ppo_rollout_length,
    "num_bc_epochs": num_bc_epochs,
    "use_dropout_rollout": use_dropout_rollout,
    "dropout_rate": dropout_rate,
}
wandb.init(
    entity="airl-lab",
    group="relocation PPO test",
    project="TBQDRL",
    config=wandb_config,
)
# record the final distillation loss on the same run
wandb.log({
    "bc/kl": float(losses[-1, 0]),
    "bc/nll": float(losses[-1, 1]),
    "bc/weight_diff": float(losses[-1, 2]),
})

gmm_ppo_configs = GMMPPOConfigs(
    policy_learnng_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.001,
    discount=0.99,
    td_lambda_discount=0.95,
    rollout_length=ppo_rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
)
ppo = GMMPPO(
    env=env,
    policy_network=student_network,
    critic_network=critic_network,
    ppo_configs=gmm_ppo_configs,
    std_anneal_fn=lambda x: jnp.maximum(0.05, 0.5 - x * 1e-4),
)

loop_random_key, subkey = jax.random.split(loop_random_key)
ppo_training_state = ppo.init(subkey)

# warm-start: distilled student policy + pre-trained teacher critic + running
# reward stats from the loaded mean.npy / var.npy. Opt states from init are
# kept (Adam moments are zero, structure matches the warm-started params);
# current_std / lr are recomputed by train() every iteration.
ppo_training_state = ppo_training_state.replace(
    policy_params=bc_training_state.policy_params,
    critic_params=critic_params,
    moving_mean=moving_mean,
    moving_squared_diff=jnp.square(moving_std),
    iteration_num=5000,
)

# fresh env reset for the PPO phase (seed matches main_ant_GMM.py convention)
seed = 114514
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
# states = jax.vmap(env.reset)(subkeys)
carry = (states, ppo_training_state, loop_random_key)


@jax.jit
def training_loop(carry, _):
    states, ppo_training_state, loop_random_key = carry

    (final_states, sampled_states, ppo_training_state, loop_random_key), aux_data = ppo.train(
        states, ppo_training_state, loop_random_key,
    )
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 13:15] ** 2, axis=-1))


    resampled_states = jax.vmap(env.resample_task_state)(final_states)
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.25, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        resampled_states,
        final_states,
    )

    new_carry = (new_states, ppo_training_state, loop_random_key)
    return new_carry, (
        aux_data.training_data.critic_error,
        aux_data.training_data.approx_kl,
        aux_data.training_data.clip_fraction,
        aux_data.rollout_data.average_return,
        jnp.mean(vs),
    )


log_period = 10
for i in range(int(num_iterations // log_period)):
    (states, ppo_training_state, loop_random_key), (
        iteration_critic_error,
        iteration_approx_kl,
        iteration_clip_fraction,
        iteration_mean_return,
        iteration_mean_v,
    ) = jax.lax.scan(training_loop, carry, length=log_period)

    wandb.log({
        "critic_RMSE": jnp.mean(iteration_critic_error),
        "approx_kl": jnp.mean(iteration_approx_kl),
        "clip_fraction": jnp.mean(iteration_clip_fraction),
        "iteration mean return": jnp.mean(iteration_mean_return),
        "iteration_mean_v": jnp.mean(iteration_mean_v),
    })
    print("v", jnp.mean(iteration_mean_v), "\t", "return", jnp.mean(iteration_mean_return))
    carry = (states, ppo_training_state, loop_random_key)

wandb.finish()
# """