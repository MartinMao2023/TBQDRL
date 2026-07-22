"""
TBQDRL ablation: relocated-only distillation + selector PPO.

This is a single-file ablation copied from ``main_test_relocation_PPO.py`` and
reusing most of its code.  Differences:

Stage 0-3  Same as the main pipeline: load the Gaussian PPO teacher (policy +
           critic), define the GMM student, collect on-policy (and optional
           dropout) rollouts, and run the relocation optimizer to produce
           ``relocated_demonstrations``.
Stage 4    Ablation: distill ONLY the relocated demonstrations into the GMM
           student via the two-stage trainer ``TwoStageBC``. Stage 1 fits all
           8 components with advantage-weighted NLL (means + shared learnable
           std floored at PPO_std); stage 2 runs importance-sampling PPO on
           the stage-1 network with the std frozen and annealed toward PPO_std.
           The resulting student policy is kept in memory (NOT saved to
           msgpack) for the next stage.
Stage 5    Train a binary "selector" policy with PPO (``SelectorPPO``) that, at
           each state, decides whether to follow the frozen Gaussian teacher
           (decision = 0) or the frozen relocated-only GMM student (decision =
           1). A fresh critic is trained, warm-started from the teacher's
           moving mean / squared diff. Log the gated return and P(relocate) to
           WandB.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from brax import envs
from flax import serialization

from typing import Tuple
from custom_types import Params, RNGKey
from networks import GC_PPO_Policy, GC_PPO_Policy_Dropout, GC_GMM_PPO_Policy, GCMLP
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from tools import (
    build_on_policy_rollout,
    build_dropout_rollout,
)
from algorithms.relocation import relocate
from data_struct.states import GeneralizedState
from data_struct import (
    MORelocationTransition,
    PPOTransition,
)

# Distillation (two-stage: NLL then importance-sampling PPO) and selector PPO.
from algorithms.two_stage_bc import TwoStageBC, TwoStageBCConfigs
# from algorithms.selector_ppo import SelectorPPO, SelectorPPOConfigs
from algorithms.binary_selector_ppo import SelectorPPO, SelectorPPOConfigs
from algorithms.gmm_ppo import PPO, PPOConfigs
# from algorithms.test_ppo import PPO, PPOConfigs


# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 5e-4  # unified (used for the GMM student distil lr)
critic_learning_rate = 5e-4
ppo_rollout_length = 32
# collection configs
rollout_length = 64
num_collect_iterations = 18 # <----------------    change to 8
relocation_config = {
    # TODO: relocation hyperparameters (candidate preferences, advantage
    # threshold, segment length, etc.)
}
# Stage 4 distillation configs
bc_mini_batch_size = 4096
bc_num_mini_batches = 16
num_stage1_epochs = 50
num_stage2_epochs = 50
# Stage 2 std anneal: alpha = min(epoch / stage2_anneal_epochs, 1) interpolates
# the frozen std from the stage-1 (clipped) std toward PPO_std.
# Stage 5 selector PPO configs
selector_learning_rate = 3e-4


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

seed = 15038
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

folder_path = "output/MORL/test2"
# folder_path = "output/MORL/test"

with open(folder_path + "/policy.msgpack", "rb") as f:
    encoded_bytes = f.read()

fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs = jnp.zeros(shape=(env.z_size,))

policy_template = policy_network.init(subkey, obs=fake_obs, z=fake_zs)
policy_params = serialization.from_bytes(policy_template, encoded_bytes)

with open(folder_path + "/critic.msgpack", "rb") as f:
    encoded_bytes = f.read()

critic_template = critic_network.init(subkey, obs=fake_obs, z=fake_zs)
critic_params = serialization.from_bytes(critic_template, encoded_bytes)

moving_mean = jnp.load(folder_path + "/mean.npy")
moving_std = jnp.sqrt(jnp.load(folder_path + "/var.npy"))

loop_random_key, subkey = jax.random.split(loop_random_key)
component_num = 4
if component_num > 1:
    component_means = jnp.concatenate([
        jnp.zeros(env.action_size),
        jax.random.normal(subkey, shape=((component_num - 1) * env.action_size)) * 0.01,
    ])
else:
    component_means = jnp.zeros(env.action_size)

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
    component_num=component_num,
    component_means=component_means,
)

teacher_std_logits = policy_params["params"]["std_logits"]
teacher_std = jax.nn.sigmoid(teacher_std_logits)

# ===========================================================================
# New change, we load the demonstrator
# ===========================================================================

# with open("output/MORL/test/policy.msgpack", "rb") as f:
#     encoded_bytes = f.read()
# expert_params = serialization.from_bytes(policy_template, encoded_bytes)
expert_params = policy_params



# ===========================================================================
# Stage 1: on-policy rollout collection with the teacher
# ===========================================================================

on_policy_rollout_fn = build_on_policy_rollout(
    env=env,
    policy_network=policy_network,
    # policy_params=policy_params,
    policy_params=expert_params, # <-------- we change this to expert
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
        final_obs, _ = jax.vmap(env.get_obs)(final_states)
        final_last_actions = final_states.z_state.last_action
        final_info = (final_obs, final_last_actions)
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

if num_collect_iterations > 0:
    states, loop_random_key, on_policy_transitions, on_policy_final_info = (
        collect_on_policy_data(states, loop_random_key)
    )

need_distill = True

if num_collect_iterations > 0:
    combined_transitions = on_policy_transitions
    combined_final_info = on_policy_final_info
else:
    need_distill = False

if need_distill:
    print(combined_transitions.obs.shape)
    # print(combined_final_info[0].shape, combined_final_info[1].shape)
    # """

    # ===========================================================================
    # Stage 3: relocation optimization -> relocated_demonstrations
    # ===========================================================================

    # loop_random_key, subkey = jax.random.split(loop_random_key)
    relocated_demonstrations = relocate(
        combined_transitions,
        combined_final_info,
        critic_network=critic_network,
        critic_params=critic_params,
        moving_mean=moving_mean,
        moving_std=moving_std,
        config=relocation_config,
        key=subkey,
        max_data_size=2097152,
    )

    # dummy = jnp.ones_like(combined_transitions.td_lambda_returns) # distill teacher

    # relocated_demonstrations = PPOTransition(
    #     obs=combined_transitions.obs,
    #     actions=combined_transitions.actions,
    #     zs=combined_transitions.zs,
    #     log_likelihood=combined_transitions.log_likelihood,
    #     rewards=dummy,
    #     td_lambda_returns=dummy,
    #     gaes=dummy,
    #     dones=dummy,
    #     truncations=dummy,
    #     weights=dummy,
    # )



bc_configs = TwoStageBCConfigs(
    learning_rate=1e-3,
    bc_epochs=1,
    mini_batch_size=bc_mini_batch_size,
    num_mini_batches=bc_num_mini_batches,
    clip_log_ratio=0.2,
)
bc = TwoStageBC(
    env=env,
    policy_network=student_network,
    teacher_std_logits=teacher_std_logits,
    bc_configs=bc_configs,
    k=component_num,
)

if need_distill:
    loop_random_key, subkey = jax.random.split(loop_random_key)
    demo_flat = relocated_demonstrations.shuffle(subkey)  # (N_demo, ...)

    _bc_chunk = bc_num_mini_batches * bc_mini_batch_size
    bc_iterations = demo_flat.obs.shape[0] // _bc_chunk
    _n_used = bc_iterations * _bc_chunk

    def _batch(x):
        return x[:_n_used].reshape(
            bc_iterations, bc_num_mini_batches, bc_mini_batch_size, *x.shape[1:]
        )

    demo_batched = jax.tree.map(_batch, demo_flat)

    loop_random_key, subkey = jax.random.split(loop_random_key)
    bc_training_state = bc.init(subkey)

    print(
        f"Two-stage BC: {_n_used} samples | {bc_iterations} iter/epoch | "
        f"{bc_num_mini_batches} mini-batches x {bc_mini_batch_size}"
    )


@jax.jit
def scan_bc_stage1(carry, data):
    training_state, key = carry
    key, subkey = jax.random.split(key)
    new_training_state, metrics = bc.stage1_update(training_state, data, subkey)
    return (new_training_state, key), metrics.bc_loss  # (3,): [nll, 0, 0]


@jax.jit
def scan_bc_stage2(carry, data):
    # `alpha` is threaded through the carry (constant per epoch) so it stays a
    # traced scalar and does not trigger recompilation across epochs.
    training_state, key, alpha = carry
    key, subkey = jax.random.split(key)
    new_training_state, metrics = bc.stage2_update(
        training_state, data, subkey, alpha
    )
    return (new_training_state, key, alpha), metrics.bc_loss  # (3,): [0, advantage, ratio]


if need_distill:
    # ---- Stage 1: advantage-weighted NLL (means + shared learnable std floored
    #      at PPO_std). metrics layout: [nll, 0, 0].
    for i in range(num_stage1_epochs):
        (bc_training_state, loop_random_key), losses = jax.lax.scan(
            scan_bc_stage1,
            (bc_training_state, loop_random_key),
            demo_batched,
        )
        print(
            f"[Stage1 epoch {i + 1:3d}]  "
            f"NLL = {float(losses[-1, 0]):.6f}"
        )

    # Stamp the stage-1 effective std so the stage-2 anneal starts from it.
    stage1_std = bc.stage1_final_std(bc_training_state)
    bc_training_state = bc_training_state.replace(stage1_std=stage1_std)

    print("stage1_std:", stage1_std)

    # ---- Stage 2: importance-sampling PPO with the std frozen + annealed by
    #      alpha toward PPO_std. metrics layout: [0, advantage, ratio].
    for i in range(num_stage2_epochs):
        alpha = jnp.clip(jnp.array(i + 1) / num_stage2_epochs, 0.0, 1.0)
        (bc_training_state, loop_random_key, _), losses = jax.lax.scan(
            scan_bc_stage2,
            (bc_training_state, loop_random_key, alpha),
            demo_batched,
        )
        print(
            f"[Stage2 epoch {i + 1:3d}]  "
            f"advantage = {float(losses[-1, 1]):.6f}  "
            f"ratio = {float(losses[-1, 2]):.6f}"
        )

    # Relocated-only distilled student policy (frozen hereafter).
    relocated_only_policy_params = bc_training_state.policy_params
    relocated_only_policy_params = {
                **relocated_only_policy_params,
                "params": {
                    **relocated_only_policy_params["params"],
                    "std_logits": teacher_std_logits,
                },
            }
"""

wandb_config = {
    "task": "relocation ablation: relocated-only distil + selector",
    "vec_env": vec_env,
    "mini_batch_size": mini_batch_size,
    "num_iterations": num_iterations,
    "policy_epochs": policy_epochs,
    "critic_epochs": critic_epochs,
    "selector_learning_rate": selector_learning_rate,
    "policy_learning_rate_per_std": policy_learning_rate_per_std,
    "critic_learning_rate": critic_learning_rate,
    "ppo_rollout_length": ppo_rollout_length,
    "use_dropout_rollout": False,
}
wandb.init(
    entity="airl-lab",
    group="relocation ablation selector",
    project="TBQDRL",
    config=wandb_config,
)

ppo_config = PPOConfigs(
    policy_learnng_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.001,
    discount=0.99,
    td_lambda_discount=0.95,
    rollout_length=rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
)


if need_distill:
    used_network = student_network
    used_params = relocated_only_policy_params
else:
    used_network = policy_network
    used_params = policy_params

selector = PPO(
    env=env,
    policy_network=used_network,
    # policy_network=policy_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
    std_anneal_fn=lambda x: jnp.maximum(0.05, 0.5 - x * 1e-4),
)

loop_random_key, subkey = jax.random.split(loop_random_key)
selector_training_state = selector.init(subkey)


selector_training_state = selector_training_state.replace(
    policy_params=used_params,
    # policy_params=policy_params,
    critic_params=critic_params,
    moving_mean=moving_mean,
    moving_squared_diff=jnp.square(moving_std),
    iteration_num=5000,
)

# fresh key for the selector PPO phase (matches main_ant_GMM.py convention)
seed = 15866
loop_random_key = jax.random.PRNGKey(seed)
carry = (states, selector_training_state, loop_random_key)


@jax.jit
def training_loop(carry, _):
    states, selector_training_state, loop_random_key = carry

    (final_states, _, selector_training_state, loop_random_key), aux_data = selector.train(
        states, selector_training_state, loop_random_key,
    )

    resampled_states = jax.vmap(env.resample_task_state)(final_states)
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.25, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        resampled_states,
        final_states,
    )

    new_carry = (new_states, selector_training_state, loop_random_key)
    return new_carry, (
        aux_data.training_data.critic_error,
        aux_data.training_data.approx_kl,
        aux_data.training_data.clip_fraction,
        aux_data.rollout_data.average_return,
    )


log_period = 10
for i in range(int(num_iterations // log_period)):
    (states, selector_training_state, loop_random_key), (
        iteration_critic_error,
        iteration_approx_kl,
        iteration_clip_fraction,
        iteration_mean_return,
    ) = jax.lax.scan(training_loop, carry, length=log_period)

    wandb.log({
        "critic_RMSE": jnp.mean(iteration_critic_error),
        "approx_kl": jnp.mean(iteration_approx_kl),
        "clip_fraction": jnp.mean(iteration_clip_fraction),
        "gated_return": jnp.mean(iteration_mean_return),
    })
    print("return", jnp.mean(iteration_mean_return))
    carry = (states, selector_training_state, loop_random_key)

wandb.finish()
"""