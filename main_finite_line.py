import jax
import flax.linen as nn
import jax.numpy as jnp
# from wrappers import AutoResetWrapper
from brax import envs
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datetime import datetime
from custom_types import RNGKey, Params
from typing import Any, Tuple, List
from algorithms.trajectory_ppo import PPO, PPOConfigs, PPOTrainingState
# from data_struct.transitions import PPOTransition
from networks import GCMLP, PPO_Policy
# from functools import partial
from flax import serialization
from task_wrappers.trajectory_following import AntLineMaternWrapper
from data_struct.states import GeneralizedState


vec_env = 4096
mini_batch_size = 16384
num_iterations = 2000
policy_epochs = 4
critic_epochs = 4
fitness_critic_epochs = 4
policy_learning_rate_per_std = 1e-3 # unified
critic_learning_rate = 5e-4
rollout_length = 96

description = {
        "task": "Test new trajectory PPO with Ant",
        "v_var": 2.25,
        "policy_learning_rate": policy_learning_rate_per_std,
        "critic_learning_rate": critic_learning_rate,
        "architecture": "Simple MLP for both networks",
        "learnable std": True,
        "vec_env": vec_env,
        "batchsize": mini_batch_size,
        "rollout_length": rollout_length,
        "iterations": num_iterations,
        "policy epoch": policy_epochs,
        "critic_epochs": critic_epochs,
        "fitness_critic_epochs": fitness_critic_epochs
    }

description_text = "\n".join(
    [f"{i}: {j}" for i, j in description.items()]
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f"./output/matern/output_{timestamp}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print(f"new folder <{folder_path}> created")


with open(folder_path + "/description.log", "w") as f:
    f.write(description_text)


ppo_config = PPOConfigs(
    policy_learning_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.001,
    discount=0.99,
    gae_lambda=0.95,
    rollout_length=rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
)


seed = 8848
loop_random_key = jax.random.PRNGKey(seed)

# # creat environment (Ant)
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", reset_noise_scale=0.0)
env = AntLineMaternWrapper(env, inner_radius=1.0, max_radius=2, tolerance_radius=0.2) # for horizon = 4.8 seconds


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


ppo = PPO(
    env=env,
    policy_network=policy_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
ppo_training_state = ppo.init(subkey)

seed = 8848
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
initial_states = jax.vmap(env.reset)(subkeys)


carry = (initial_states, ppo_training_state, loop_random_key)


@jax.jit
def training_loop(
    carry: Tuple[GeneralizedState, PPOTrainingState, RNGKey], 
    _: None,
    ) -> Tuple[Tuple, Tuple]:

    states, ppo_training_state, loop_random_key = carry

    (final_states, sampled_states, ppo_training_state, loop_random_key), aux_data = ppo.train(
        states,
        ppo_training_state,
        loop_random_key,
    )
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 13: 15]**2, axis=-1))
    states = jax.vmap(env.resample_task_state)(initial_states)
    
    new_carry = (
        states,
        ppo_training_state,
        loop_random_key,
    )

    return new_carry, (aux_data, jnp.mean(vs))


wandb.init(
    entity="airl-lab",
    project="TBQDRL",
    group="Trajectory PPO test",
    config=description,
)


log_period = 10

for i in range(int(num_iterations / log_period)):

    (
        states, 
        ppo_training_state, 
        loop_random_key,
        ), (stacked_aux_data, iteration_mean_v) = jax.lax.scan(
        training_loop,
        carry,
        length=log_period,
    )

    wandb.log({
        "critic_RMSE": jnp.mean(stacked_aux_data.critic_rmse),
        "approx_kl": jnp.mean(stacked_aux_data.policy_approx_kl),
        "iteration mean return": jnp.mean(stacked_aux_data.average_return), 
        "iteration mean reward": jnp.mean(stacked_aux_data.average_reward),
        "dones per episode": jnp.mean(stacked_aux_data.done_count) / vec_env,
        # "gae mean": jnp.mean(stacked_aux_data.gae_mean),
        "iteration_mean_v": jnp.mean(iteration_mean_v), 
        })
    
    carry = (states, ppo_training_state, loop_random_key)

# (
#     _, 
#     final_ppo_training_state, 
#     loop_random_key,
# ) = carry

# model_bytes = serialization.to_bytes(final_ppo_training_state.policy_params)
# critic_bytes = serialization.to_bytes(final_ppo_training_state.critic_params)

# with open(folder_path + f"/model_{structure}.msgpack", "wb") as f:
#     f.write(model_bytes)

# with open(folder_path + f"/critic_{structure}.msgpack", "wb") as f:
#     f.write(critic_bytes)

# with open(folder_path + f"/fitness_critic_{structure}.msgpack", "wb") as f:
#     f.write(fitness_critic_bytes)

wandb.finish()

