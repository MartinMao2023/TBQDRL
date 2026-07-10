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


# ===========================================================================
# Stage 0 — load pre-trained teacher, define networks
# ===========================================================================
# TODO:
#   - create env (brax ant + AntWrapper)
#   - define Gaussian teacher policy (GC_PPO_Policy) and critic (GCMLP)
#   - define GMM student policy (GC_GMM_PPO_Policy)
#   - load pre-trained teacher policy + critic params from msgpack
#   - (no task/goal sampling in this stage)

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


# ===========================================================================
# Stage 1 — collect rollouts into a new buffer class
# ===========================================================================
# Structure:
#   play_step_fn  -> builds one MORelocationTransition
#   jax.vmap      -> over vec_env parallel envs
#   jax.lax.scan  -> over rollout_length           (inner, lives in tools.py)
#   jax.lax.scan  -> over num_collect_iterations   (outer, resamples preference)
# Final transition shape: (num_collect_iterations, rollout_length, vec_env, ...)
#
# Collected data is consumed by:
#   1. teacher-policy distillation (Stage 4)
#   2. (TBD) critic fine-tuning
#   3. relocation advantage computation (Stage 3)

on_policy_rollout_fn = build_on_policy_rollout(
    env=env,
    policy_network=policy_network,
    policy_params=policy_params,
    rollout_length=rollout_length,
)


@jax.jit
def collect_on_policy_data(starting_states, key):
    """Outer scan: run a rollout then resample the preference per env, repeat."""

    def outer_step(carry, _):
        states, key = carry
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=vec_env)
        final_states, transitions = on_policy_rollout_fn(states, subkeys)
        # resample preference for the next iteration (keeps last_action)
        final_states = jax.vmap(env.resample_task_state)(final_states)
        return (final_states, key), transitions

    (final_states, key), all_transitions = jax.lax.scan(
        outer_step,
        (starting_states, key),
        length=num_collect_iterations,
    )
    return final_states, key, all_transitions


loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)

states, loop_random_key, on_policy_transitions = (
    collect_on_policy_data(states, loop_random_key)
)
# print(on_policy_transitions.mo_rewards.shape)

# on_policy_transitions fields: (num_collect_iterations, rollout_length, vec_env, ...)

# ===========================================================================
# Stage 2 — (optional) dropout on the pre-trained policy for diversity
# ===========================================================================
# Apply inverted dropout (mask a fraction of hidden neurons, upscale by
# 1/(1-rate)) to the teacher policy during rollout, producing more diverse
# trajectories.  The dropout policy shares the teacher's parameter tree, so
# the loaded weights are reused directly.  Gated by `use_dropout_rollout`;
# when enabled, the collected data is concatenated onto the Stage 1 buffer
# along the iteration axis and feeds distillation (Stage 4) and relocation
# (Stage 3).

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
        final_states = jax.vmap(env.resample_task_state)(final_states)
        return (final_states, key), transitions

    (final_states, key), all_transitions = jax.lax.scan(
        outer_step,
        (starting_states, key),
        length=num_dropout_iterations,
    )
    return final_states, key, all_transitions




if use_dropout_rollout:
    loop_random_key, subkey = jax.random.split(loop_random_key)
    # subkeys = jax.random.split(subkey, num=vec_env)
    # dropout_start_states = jax.vmap(env.reset)(subkeys)

    states, loop_random_key, dropout_transitions = collect_dropout_data(
        states, loop_random_key
    )
    # print(dropout_transitions.mo_rewards.shape)


# ===========================================================================
# Stage 3 — relocation optimization
# ===========================================================================
# (a) Build the teacher-distillation dataset: concatenate the on-policy and
#     (optional) dropout transitions, then replace the executed action with
#     the teacher policy's deterministic action mean.  Sub-sampling is
#     skipped here because we only collect ~2M steps; if the dataset grows,
#     sub-sample before this step (TODO).  The distillation copy is shuffled
#     (i.i.d. cloning); the combined transitions keep their trajectory
#     structure for relocation.
# (b) Pass the combined transitions (trajectory structure preserved) to
#     `relocate` — a placeholder for now that returns the input unchanged.

if use_dropout_rollout:
    combined_transitions = jax.tree.map(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        on_policy_transitions, dropout_transitions,
    )
else:
    combined_transitions = on_policy_transitions

# distillation dataset: copy with teacher action mean, then shuffle (i.i.d.)
distillation_transitions = with_policy_action_mean(
    combined_transitions, policy_network, policy_params,
)
loop_random_key, subkey = jax.random.split(loop_random_key)
distillation_transitions = distillation_transitions.shuffle(subkey)
# distillation_transitions fields: (N_distil, ...)

# relocation: pass the combined transitions (trajectories preserved)
loop_random_key, subkey = jax.random.split(loop_random_key)
relocated_transitions = relocate(
    combined_transitions,
    critic_network=critic_network,
    critic_params=critic_params,
    config=relocation_config,
    key=subkey,
)
# relocated_transitions: placeholder returns combined_transitions unchanged


# ===========================================================================
# Stage 4 — distill teacher policy + relocated trajectories into the GMM
# ===========================================================================
# The trainer (GMMDistillationBC, from algorithms/intermediate_bc.py) expects a
# pair (GMMDistillationTransition, PPOTransition):
#   * the GMMDistillationTransition feeds the KL term — it carries the teacher's
#     component means and selection logits per state;
#   * the PPOTransition feeds the (advantage-weighted) NLL term on the relocated
#     demonstrations.
# Because the teacher here is a *single Gaussian* (not a GMM), we represent it
# as a 1-component GMM: action_means gets a new component axis and
# component_logits is zeros (softmax -> 1.0). The existing KL code then works
# unchanged (the teacher axis has size 1 and the logsumexp collapses trivially).
# For the PPOTransition we only need obs/zs/actions (+ dones/truncations); the
# log_likelihood / rewards / gaes entries are dummy, and `weights` carries the
# relocation advantage (placeholder ones for now).

# (a) Teacher-distillation set -> 1-component GMMDistillationTransition.
#     `distillation_transitions.actions` already holds the teacher action mean
#     (set by with_policy_action_mean), so it becomes the single component mean.
_distil_batch = distillation_transitions.obs.shape[:-1]
gmm_distillation_transitions = GMMDistillationTransition(
    obs=distillation_transitions.obs,
    zs=distillation_transitions.zs,
    action_means=distillation_transitions.actions[..., None, :],  # (..., 1, d)
    component_logits=jnp.zeros((*_distil_batch, 1)),               # (..., 1) -> softmax 1
)

# (b) Relocated set -> PPOTransition with dummy log_likelihood/rewards/gaes and
#     advantage = ones (placeholder until `relocate` fills in real advantages).
_dummy = jnp.zeros_like(relocated_transitions.dones)  # (..., 1)
demonstrate_transitions = PPOTransition(
    obs=relocated_transitions.obs,
    actions=relocated_transitions.actions,
    zs=relocated_transitions.zs,
    log_likelihood=_dummy,
    rewards=_dummy,
    td_lambda_returns=relocated_transitions.td_lambda_returns,
    gaes=_dummy,
    dones=relocated_transitions.dones,
    truncations=relocated_transitions.truncations,
    weights=jnp.ones_like(relocated_transitions.weights), # <---- TO DO
)

# TODO (next):
#   - instantiate GMMDistillationBC(env, student_network, teacher_std_logits,
#     GMMBCConfigs(...)) and init() the student params (warm-start from teacher
#     where desired);
#   - batch gmm_distillation_transitions / demonstrate_transitions into
#     (num_mini_batches, mini_batch_size, ...) and run state_update in a
#     jax.lax.scan training loop (as in main_test_distil.py).


# ===========================================================================
# Stage 5 — continue PPO on the distilled GMM, log to WandB
# ===========================================================================
# TODO:
#   - instantiate GMMPPO with the GMM student network
#   - warm-start policy params from the distilled student
#   - run the PPO training loop (jax.lax.scan, as in main_ant_GMM.py)
#   - log the learning curve to WandB
