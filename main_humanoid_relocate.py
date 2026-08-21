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
from networks import PPO_Policy, GCMLP, Multi_Action_PPO_Policy, Selector
from task_wrappers.humanoid_mo_wrapper import HumanoidMOWrapper
from tools import (
    build_on_policy_rollout,
    with_policy_action_mean,
)
# from algorithms.relocation import relocate
from algorithms.cem_relocation import relocate, eval_return_and_GAE
from data_struct.states import GeneralizedState
from data_struct import (
    MORelocationTransition,
    GMMDistillationTransition,
    PPOTransition,
)

from algorithms.combined_distill import CombinedDistill, CombinedDistillConfigs
from algorithms.gmm_ppo import PPO, PPOConfigs
from algorithms.critic_fine_tuning import CriticFineTuning, CriticFineTuningConfigs


# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 2e-4  # unified (used for the GMM student distil lr)
selector_learning_rate = 1e-4
critic_learning_rate = 5e-4
ppo_rollout_length = 32
# collection configs
rollout_length = 64
num_collect_iterations = 16 # <----------------    change to 8
relocation_config = {
}

bc_mini_batch_size = 2048
num_stage1_epochs = 32
num_stage2_epochs = 8
# data_size = 2097152
data_size = 1572864


env = envs.create(env_name="humanoid", episode_length=4096, backend="mjx", auto_reset=True, action_repeat=2)
env = HumanoidMOWrapper(env)

critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
policy_network = PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.silu,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
)

seed = 38258
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

# folder_path = "output/MORL/test5"
folder_path = "output/MORL/huamnoid_mo_corrected"
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
moving_mse = jnp.load(folder_path + "/mse.npy")

critic_tuning_configs = CriticFineTuningConfigs(critic_epochs=2)
critic_tuner = CriticFineTuning(critic_network, critic_tuning_configs)

critic_tuning_state = critic_tuner.init(
    critic_params,
    moving_mean=moving_mean,
    moving_std=moving_std,
    moving_mse=moving_mse,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
k1 = 0
k2 = 4
component_num = k1 + k2
if component_num > 1:
    component_means = jnp.concatenate([
        jnp.zeros(env.action_size),
        jax.random.normal(subkey, shape=((component_num - 1) * env.action_size)) * 0.01,
    ])
else:
    component_means = jnp.zeros(env.action_size)

student_hidden_layers = (256, 256)
student_action_network = Multi_Action_PPO_Policy(
    hidden_layer_sizes=(256, 256),
    action_dim=env.action_size,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
    component_num=component_num,
    component_means=component_means,   
)


student_selection_network = Selector(
    hidden_layer_sizes=(128, 128),
    component_num=component_num,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
)



teacher_std_logits = policy_params["params"]["std_logits"]
teacher_std = jax.nn.sigmoid(teacher_std_logits)


expert_demo = False
if expert_demo:
    with open("output/MORL/test6/policy.msgpack", "rb") as f:
        encoded_bytes = f.read()
    expert_params = serialization.from_bytes(policy_template, encoded_bytes)
else:
    expert_params = policy_params


on_policy_rollout_fn = build_on_policy_rollout(
    env=env,
    policy_network=policy_network,
    policy_params=expert_params, 
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


    loop_random_key, subkey = jax.random.split(loop_random_key)
    flattened = combined_transitions.flatten()
    index = jax.random.permutation(subkey, flattened.shape[0])[: data_size]
    distillation_transitions = combined_transitions.from_flatten(flattened[index], combined_transitions)
else:
    need_distill = False


dummy = jnp.ones_like(on_policy_transitions.td_lambda_returns)
on_policy_PPO_transitions = PPOTransition(
    obs=on_policy_transitions.obs,
    actions=on_policy_transitions.actions,
    zs=on_policy_transitions.zs,
    log_likelihood=on_policy_transitions.log_likelihood,
    rewards=jnp.sum(on_policy_transitions.mo_rewards * on_policy_transitions.zs[..., -5:], axis=-1, keepdims=True),
    td_lambda_returns=dummy,
    gaes=dummy,
    dones=on_policy_transitions.dones,
    truncations=on_policy_transitions.truncations,
    weights=dummy,
)

print("start tuning")
loop_random_key, subkey = jax.random.split(loop_random_key)
final_obs, final_last_actions = on_policy_final_info
critic_tuning_state, _ = critic_tuner.train(
    critic_tuning_state,
    on_policy_PPO_transitions,
    final_obs,
    jnp.concatenate([final_last_actions, on_policy_transitions.zs[:, -1, :, -5:]], axis=-1),
    subkey,
    )
print("tuning complete")

need_relocate = True

if need_distill:
    print(combined_transitions.obs.shape)

    if need_relocate:
        loop_random_key, subkey = jax.random.split(loop_random_key)
        relocated_demonstrations = relocate(
            combined_transitions,
            combined_final_info,
            critic_network=critic_network,
            critic_params=critic_params,
            moving_mean=moving_mean,
            moving_std=moving_std,
            config=relocation_config,
            key=subkey,
            max_data_size=data_size,
        )

        lambda_returns, gaes = eval_return_and_GAE(
            relocated_demonstrations,
            critic_network=critic_network,
            critic_params=critic_tuning_state.critic_params,
            moving_mean=moving_mean,
            moving_std=moving_std,
            config=relocation_config,
            )
        mean_gae = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        clipped_gaes = jnp.clip(gaes, 0.0)

        print("New average GAEs:", mean_gae)

        relocated_demonstrations = relocated_demonstrations.replace(
            td_lambda_returns=lambda_returns,
            gaes=gaes,
            weights=jnp.clip(clipped_gaes / (jnp.mean(clipped_gaes) + 1e-6), max=10),
        )
        

    else:
        dummy = jnp.ones_like(combined_transitions.td_lambda_returns) # distill teacher

        loop_random_key, subkey = jax.random.split(loop_random_key)

        # # # random sample
        # preference_part_1 = jax.random.normal(subkey, (num_collect_iterations, 64, vec_env, 2)) # <--- all independent
        # loop_random_key, subkey = jax.random.split(loop_random_key)
        # preference_part_2 = jnp.abs(jax.random.normal(subkey, (num_collect_iterations, 64, vec_env, 3))) # <--- all independent
        # # preferences = combined_transitions.zs[..., 8:] * 0.0 + jnp.concatenate([preference_part_1, preference_part_2], axis=-1)
        # preferences = jnp.concatenate([preference_part_1, preference_part_2], axis=-1)

        # preference_noise = jax.random.normal(subkey, (num_collect_iterations, 1, vec_env, 5)) * 0.4
        # preferences = combined_transitions.zs[..., -5:] # (18, 64, 4096, 5)
        # preferences = jnp.clip(preferences + preference_noise, min=jnp.array([-100, -100, 0, 0, 0]))

        # preferences = preferences / jnp.linalg.norm(preferences, axis=-1, keepdims=True)

        # new_zs = jnp.concatenate([combined_transitions.zs[..., :-5], preferences], axis=-1)


        relocated_demonstrations = PPOTransition(
            obs=combined_transitions.obs,
            actions=combined_transitions.actions,
            zs=combined_transitions.zs,
            # zs=new_zs,
            log_likelihood=combined_transitions.log_likelihood,
            rewards=dummy,
            td_lambda_returns=dummy,
            gaes=dummy,
            dones=dummy,
            truncations=dummy,
            weights=dummy,
        )



bc_configs = CombinedDistillConfigs(
    action_learning_rate=3e-4,
    selector_learning_rate=3e-4,
    stage1_epochs=num_stage1_epochs,
    stage2_epochs=num_stage2_epochs,
    mini_batch_size=bc_mini_batch_size,
    clip_log_ratio=0.2,
    k1=k1,
)
bc = CombinedDistill(
    env=env,
    policy_network=student_action_network,
    selector_network=student_selection_network,
    teacher_std_logits=teacher_std_logits,
    bc_configs=bc_configs,
)

if need_distill:
    loop_random_key, subkey = jax.random.split(loop_random_key)
    demo_flat = relocated_demonstrations.shuffle(subkey)            # (N_demo, ...)
    distill_flat = distillation_transitions

    # Both buffers must end up the same size so the per-epoch scan lines up.
    _n = min(distill_flat.obs.shape[0], demo_flat.obs.shape[0])
    _n = (_n // bc_mini_batch_size) * bc_mini_batch_size

    def _batch(x):
        return x[:_n].reshape(-1, bc_mini_batch_size, *x.shape[1:])

    demo_batched = jax.tree.map(_batch, demo_flat)
    distill_batched = jax.tree.map(_batch, distill_flat)

    _distil_batch = distill_batched.obs.shape[:-1]
    gmm_distillation_transitions = GMMDistillationTransition(
        obs=distill_batched.obs,
        zs=distill_batched.zs,
        action_means=distill_batched.actions[..., None, :],  # (..., 1, d)
        component_logits=jnp.zeros((*_distil_batch, 1)),     # (..., 1) -> softmax 1
    )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    bc_training_state = bc.init(subkey)

    print(
        f"Combined distill: {_n} samples | "
        f"{_n // bc_mini_batch_size} mini-batches x {bc_mini_batch_size} | "
        f"stage1 {num_stage1_epochs} epochs + stage2 {num_stage2_epochs} epochs"
    )

    bc_training_state, _metrics = bc.distill(
        bc_training_state,
        (gmm_distillation_transitions, demo_batched),
        subkey,
    )

    print(_metrics.bc_loss)

    del distillation_transitions, demo_flat, relocated_demonstrations 
    del gmm_distillation_transitions, demo_batched

    # Relocated-only distilled student policy (std fixed to teacher by distill).
    relocated_only_policy_params = bc_training_state.policy_params

# """


group_name = "relocation humanoid"


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


print(f"k1: {k1}, component num: {component_num}")

ppo_config = PPOConfigs(
    policy_learning_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    selector_learning_rate=selector_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.001,
    discount=0.99,
    gae_lambda=0.95,
    rollout_length=ppo_rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
)


if need_distill:
    used_action_network = student_action_network
    used_selector_network = student_selection_network
    used_action_params = relocated_only_policy_params
    used_selector_params = bc_training_state.selector_params
else:
    used_action_network = policy_network
    used_selector_network = None
    used_action_params = policy_params
    used_selector_params = None

selector = PPO(
    env=env,
    policy_network=used_action_network,
    selector_network=used_selector_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
selector_training_state = selector.init(subkey)


selector_training_state = selector_training_state.replace(
    policy_params=used_action_params,
    selector_params=used_selector_params,
    critic_params=critic_params,
    moving_mean=moving_mean,
    moving_squared_diff=jnp.square(moving_std),
    iteration_num=5000,
    moving_mse=300,
)

# fresh key for the selector PPO phase (matches main_ant_GMM.py convention)
# seed = 8848
# loop_random_key = jax.random.PRNGKey(seed)
carry = (states, selector_training_state, loop_random_key, 0)

wandb.init(
    entity="airl-lab",
    group=group_name,
    project="TBQDRL",
    config=wandb_config,
)


@jax.jit
def training_loop(carry, _):
    states, selector_training_state, loop_random_key, num = carry

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

    new_carry = (new_states, selector_training_state, loop_random_key, num + 1)
    return new_carry, aux_data
    


log_period = 10
for i in range(int(num_iterations // log_period)):
    carry,  stacked_aux_data = jax.lax.scan(training_loop, carry, length=log_period)

    wandb.log({
        "critic_RMSE": jnp.mean(stacked_aux_data.critic_rmse),
        "approx_kl": jnp.mean(stacked_aux_data.policy_approx_kl),
        "gated_return": jnp.mean(stacked_aux_data.average_return),
    })
    print("return", jnp.mean(stacked_aux_data.average_return))

wandb.finish()


# """