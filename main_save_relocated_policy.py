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
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from task_wrappers.humanoid_mo_wrapper import HumanoidMOWrapper
from tools import (
    build_on_policy_rollout,
)
from algorithms.cem_relocation import relocate, eval_return_and_GAE
from data_struct.states import GeneralizedState
from data_struct import (
    PPOTransition,
)
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
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
selector_learning_rate = 1e-4
critic_learning_rate = 5e-4
rollout_length = 64
num_collect_iterations = 16 # <----------------    change to 8
relocation_config = {}

bc_mini_batch_size = 2048
num_stage1_epochs = 32
num_stage2_epochs = 4
data_size = 1572864


# env_name = "ant"
env_name = "humanoid"

need_distill = True
need_relocate = True
expert_demo = False




if env_name == "ant":
    num_collect_iterations = 16
    env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
    env = AntMOWrapper(env)
else:
    num_collect_iterations = 16 # <----------------    change to 8
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

critic_tuning_configs = CriticFineTuningConfigs(critic_epochs=2)
critic_tuner = CriticFineTuning(critic_network, critic_tuning_configs)

seed = 4455
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)


if env_name == "ant":
    folder_path = "output/MORL/test3"
else:
    folder_path = "output/MORL/huamnoid_150" # <--- humanoid

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
    if env_name == "ant":
        with open("output/MORL/test4/policy.msgpack", "rb") as f:
            encoded_bytes = f.read()
    else:
        with open("output/MORL/huamnoid_long/policy.msgpack", "rb") as f:
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
    index = jax.random.permutation(subkey, flattened.shape[0])[:data_size]
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
    final_zs,
    subkey,
    )
print("tuning complete")





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
        dummy = jnp.ones_like(combined_transitions.td_lambda_returns) 
        relocated_demonstrations = PPOTransition(
            obs=combined_transitions.obs,
            actions=combined_transitions.actions,
            zs=combined_transitions.zs,
            log_likelihood=combined_transitions.log_likelihood,
            rewards=dummy,
            td_lambda_returns=dummy,
            gaes=dummy,
            dones=dummy,
            truncations=dummy,
            weights=dummy,
        )
        relocated_demonstrations = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), relocated_demonstrations)
        all_transitions = relocated_demonstrations
        del relocated_demonstrations, on_policy_PPO_transitions



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



policy_model_bytes = serialization.to_bytes(extracted_policy_params)
selector_model_bytes = serialization.to_bytes(extracted_selector_params)
if expert_demo:
    if env_name == "ant":
        if need_relocate:
            folder_path = f"./output/MORL/ant_relocated_expert"
        else:
            folder_path = f"./output/MORL/ant_distill_expert"
    else:
        if need_relocate:
            folder_path = f"./output/MORL/humanoid_relocated_expert"
        else:
            folder_path = f"./output/MORL/humanoid_distill_expert"
else:
    if env_name == "ant":
        if need_relocate:
            folder_path = f"./output/MORL/ant_relocated"
        else:
            folder_path = f"./output/MORL/ant_no_relocated"
    else:
        if need_relocate:
            folder_path = f"./output/MORL/humanoid_relocated"
        else:
            folder_path = f"./output/MORL/humanoid_no_relocated"


# if not os.path.exists(folder_path):
#     os.makedirs(folder_path, exist_ok=True)
#     print(f"new folder <{folder_path}> created")

# with open(folder_path + f"/policy.msgpack", "wb") as f:
#     f.write(policy_model_bytes)
# with open(folder_path + f"/selector.msgpack", "wb") as f:
#     f.write(selector_model_bytes)

