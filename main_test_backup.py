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
from networks import PPO_Policy, GCMLP, Selector
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from task_wrappers.humanoid_mo_wrapper import HumanoidMOWrapper
from data_struct.states import GeneralizedState
# from algorithms.binary_option_critic_ppo import BinaryOptionCriticPPO, BinaryOptionCriticPPOConfigs
from algorithms.frozen_backup_option_critic_ppo import FrozenBackupOptionCriticPPO, FrozenBackupOptionCriticPPOConfigs



# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 500
policy_epochs = 4
critic_epochs = 4
selector_epochs = 4
policy_learning_rate_per_std = 2e-4 # <-- humanoid
# policy_learning_rate_per_std = 1e-3 # <--- ant
critic_learning_rate = 5e-4
selector_learning_rate = 3e-4
rollout_length = 32



# env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
# env = AntMOWrapper(env)
env = envs.create(env_name="humanoid", episode_length=4096, backend="mjx", auto_reset=True, action_repeat=2)
env = HumanoidMOWrapper(env)

critic_hidden_layers: Tuple[int, ...] = (128, 128)
selector_hidden_layers: Tuple[int, ...] = (128, 128)
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

selector_network = GCMLP(
    layer_sizes=selector_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.silu,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    final_activation=lambda x: x - 2.85,
)


# policy_0_path = "output/MORL/test5/policy.msgpack"
# policy_1_path = "output/MORL/test6/policy.msgpack"
# policy_1_path = "output/MORL/humanoid_relocated/policy.msgpack" # <---- back to relocation
# policy_1_path = "output/MORL/humanoid_relocated_expert/policy.msgpack" # <---- relocated expert demo
# policy_1_path = "output/MORL/humanoid_distill_expert/policy.msgpack" # <---- expert demo

# policy_1_path = "output/MORL/test4/policy.msgpack"
# policy_1_path = "output/MORL/humanoid_no_relocated/policy.msgpack"
# critic_path = "output/MORL/test5/critic.msgpack"

# policy_1_path = "output/MORL/ant_relocated/policy.msgpack" # <---- ant relocation
# policy_1_path = "output/MORL/humanoid_relocated/policy.msgpack"
policy_1_path = "output/MORL/humanoid_no_relocated/policy.msgpack"


seed = 454353
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs = jnp.zeros(shape=(env.z_size,))

policy_template = policy_network.init(subkey, obs=fake_obs, z=fake_zs)
critic_template = critic_network.init(subkey, obs=fake_obs, z=fake_zs)

# folder_path = "output/MORL/test3"
# folder_path = "output/MORL/test5"
folder_path = "output/MORL/huamnoid_mo_corrected"

with open(folder_path + "/policy.msgpack", "rb") as f:
    encoded_bytes = f.read()
policy_0_params = serialization.from_bytes(policy_template, encoded_bytes)

with open(policy_1_path, "rb") as f:
    encoded_bytes = f.read()
policy_1_params = serialization.from_bytes(policy_template, encoded_bytes)
# policy_1_params = jax.tree.map(lambda x: -x, policy_1_params)

with open(folder_path + "/critic.msgpack", "rb") as f:
    encoded_bytes = f.read()
critic_params = serialization.from_bytes(critic_template, encoded_bytes)

moving_mean = jnp.load(folder_path + "/mean.npy")
moving_std = jnp.sqrt(jnp.load(folder_path + "/var.npy"))
moving_mse = jnp.load(folder_path + "/mse.npy")


ppo_config = FrozenBackupOptionCriticPPOConfigs(
    policy_learning_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    selector_learning_rate=selector_learning_rate,
    entropy_gain=0.001,
    selector_deviation_limit=0.05,
    discount=0.99,
    gae_lambda=0.95,
    rollout_length=rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
    selector_epochs=selector_epochs,
    moving_mse_ema_learning_rate=0.05,
)

option_critic_ppo = FrozenBackupOptionCriticPPO(
    env=env,
    policy_network=policy_network,
    critic_network=critic_network,
    selector_network=selector_network,
    configs=ppo_config,
    persistence_anneal_fn=lambda x: jnp.clip(0.975 - x * 0.0005, 0.9, 0.95),
    # persistence_anneal_fn=lambda x: 1.0,
)



loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)
loop_random_key, subkey = jax.random.split(loop_random_key)

training_state = option_critic_ppo.init(
    subkey,
    policy_params=policy_1_params,
    frozen_policy_params=policy_0_params,
    frozen_critic_params=critic_params,
    critic_mean=moving_mean,
    critic_std=moving_std,
    moving_mse=moving_mse,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
carry = (states, training_state, subkey)


@jax.jit
def training_loop(carry, _):
    states, training_state, loop_random_key = carry

    (final_states, training_state, loop_random_key), aux_data = option_critic_ppo.train(
        states, training_state, loop_random_key, 
    )

    resampled_states = jax.vmap(env.resample_task_state)(final_states)
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.25, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        resampled_states,
        final_states,
    )

    new_carry = (new_states, training_state, loop_random_key)
    return new_carry, aux_data



wandb_config = {
    "task": "test option critic PPO",
    "vec_env": vec_env,
    "mini_batch_size": mini_batch_size,
    "num_iterations": num_iterations,
    "policy_epochs": policy_epochs,
    "critic_epochs": critic_epochs,
    "selector_learning_rate": selector_learning_rate,
    "policy_learning_rate_per_std": policy_learning_rate_per_std,
    "critic_learning_rate": critic_learning_rate,
    "ppo_rollout_length": rollout_length,
    "use_dropout_rollout": False,
}
wandb.init(
    entity="airl-lab",
    group="backup option critic",
    project="TBQDRL",
    config=wandb_config,
)

log_period = 5
for i in range(int(num_iterations // log_period)):
    carry, stacked_aux_data = jax.lax.scan(training_loop, carry, length=log_period)

    # print("policy 0 return", jnp.mean(stacked_aux_data.option_returns[:, 0]),)
    # print("policy 1 return", jnp.mean(stacked_aux_data.option_returns[:, 1]),)
    # print("selector_preference", jnp.mean(stacked_aux_data.selector_preference))
    # print("selector_SNR", jnp.mean(stacked_aux_data.selector_SNR))

    wandb.log({
        "critic RMSE": jnp.mean(stacked_aux_data.critic_rmse),
        "approx kl": jnp.mean(stacked_aux_data.policy_approx_kl),
        "policy return": jnp.mean(stacked_aux_data.policy_return),
        "selector return": jnp.mean(stacked_aux_data.selector_value),
        "selector_preference": jnp.mean(stacked_aux_data.selector_preference),
        "selector_advantage": jnp.mean(stacked_aux_data.selector_advantage),
        "selector_certainty": jnp.mean(stacked_aux_data.selector_certainty),
        "selector_SNR": jnp.mean(stacked_aux_data.selector_SNR),
        # "done count": jnp.mean(stacked_aux_data.total_dones),
    })

wandb.finish()
