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
)
# from algorithms.relocation import relocate
from algorithms.cem_relocation import relocate, eval_return_and_GAE
from data_struct.states import GeneralizedState
from data_struct import (
    PPOTransition,
)
from algorithms.gmm_ppo import PPO, PPOConfigs
from algorithms.critic_fine_tuning import CriticFineTuning, CriticFineTuningConfigs
from algorithms.awr import (
    AWR,
    AWRConfigs,
    AWRGMMPolicyConfigs,
    AWRGMMPolicyExtractor,
)



# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 500
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
num_stage2_epochs = 4
# data_size = 2097152
data_size = 1572864

expert_demo = True
need_relocate = True
need_distill = True

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
folder_path = "output/MORL/huamnoid_150"
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
component_num = 4
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


if expert_demo:
    with open("output/MORL/humanoid_long/policy.msgpack", "rb") as f:
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
final_zs = jnp.concatenate([final_last_actions, on_policy_transitions.zs[:, -1, :, -5:]], axis=-1)
critic_tuning_state, _ = critic_tuner.train(
    critic_tuning_state,
    on_policy_PPO_transitions,
    final_obs,
    jnp.concatenate([final_last_actions, on_policy_transitions.zs[:, -1, :, -5:]], axis=-1),
    subkey,
    )
print("tuning complete")



def to_traj(x):
    x = jnp.swapaxes(x, -3, -2)  # (..., vec_env, rollout_length, d)
    return x.reshape(-1, rollout_length, x.shape[-1])

# flatten all
on_policy_PPO_transitions = jax.tree.map(to_traj, on_policy_PPO_transitions)
final_obs = final_obs.reshape(-1, final_obs.shape[-1])
final_zs = final_zs.reshape(-1, final_zs.shape[-1])
on_policy_PPO_transitions = critic_tuner.calculate_td_lambda_returns(
    critic_tuning_state,
    on_policy_PPO_transitions,
    final_obs,
    final_zs,
)
on_policy_PPO_transitions = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), on_policy_PPO_transitions)


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

        del on_policy_transitions
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
        print("New average GAEs:", mean_gae)

        relocated_demonstrations = relocated_demonstrations.replace(
            td_lambda_returns=lambda_returns,
        )

        relocated_demonstrations = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), relocated_demonstrations)
        all_transitions = relocated_demonstrations

        del relocated_demonstrations, on_policy_PPO_transitions
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


        # relocated_demonstrations = PPOTransition(
        #     obs=combined_transitions.obs,
        #     actions=combined_transitions.actions,
        #     zs=combined_transitions.zs,
        #     # zs=new_zs,
        #     log_likelihood=combined_transitions.log_likelihood,
        #     rewards=dummy,
        #     td_lambda_returns=dummy,
        #     gaes=dummy,
        #     dones=dummy,
        #     truncations=dummy,
        #     weights=dummy,
        # )
        # relocated_demonstrations = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), relocated_demonstrations)
        # all_transitions = relocated_demonstrations
        # del relocated_demonstrations, on_policy_PPO_transitions

        all_transitions = on_policy_PPO_transitions
        del on_policy_PPO_transitions



if need_distill:
    awr_configs = AWRConfigs(
        critic_learning_rate=critic_learning_rate,
        critic_epochs=critic_epochs,
        mini_batch_size=bc_mini_batch_size,
    )
    awr = AWR(
        critic_network=critic_network,
        configs=awr_configs,
    )
    awr_training_state = awr.init(
        critic_params=critic_tuning_state.critic_params,
        moving_mean=moving_mean,
        moving_std=moving_std,
    )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    (awr_training_state, all_transitions), awr_metrics = awr.train(
        awr_training_state,
        all_transitions,
        subkey,
    )
    critic_params = awr_training_state.critic_params

    awr_policy_configs = AWRGMMPolicyConfigs(
        policy_learning_rate=3e-4,
        selector_learning_rate=selector_learning_rate,
        policy_epochs=num_stage1_epochs,
        stage2_epochs=num_stage2_epochs,
        mini_batch_size=bc_mini_batch_size,
        temperature=7.5,
    )
    awr_policy_extractor = AWRGMMPolicyExtractor(
        env=env,
        policy_network=student_action_network,
        selector_network=student_selection_network,
        configs=awr_policy_configs,
    )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    awr_policy_training_state = awr_policy_extractor.init(subkey)
    awr_policy_training_state, awr_policy_metrics = (
        awr_policy_extractor.train(
            awr_policy_training_state,
            all_transitions,
            target_std_logits=teacher_std_logits,
        )
    )

    print("AWR critic RMSE:", awr_metrics.critic_rmse)
    print(
        "AWR stage 1/2 loss:",
        awr_policy_metrics.stage1_loss,
        awr_policy_metrics.stage2_loss,
    )

    extracted_policy_params = awr_policy_training_state.policy_params
    extracted_selector_params = awr_policy_training_state.selector_params


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
    used_action_params = extracted_policy_params
    used_selector_params = extracted_selector_params
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
    moving_mse=moving_mse,
)

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
        "selection entropy": jnp.mean(stacked_aux_data.selection_entropy),
    })
    print("return", jnp.mean(stacked_aux_data.average_return))

wandb.finish()

