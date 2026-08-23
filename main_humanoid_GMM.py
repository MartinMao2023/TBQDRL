import jax
import flax.linen as nn
import jax.numpy as jnp
# from wrappers import AutoResetWrapper
from brax import envs
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from custom_types import RNGKey, Params
from typing import Any, Tuple, List
from flax import serialization
from task_wrappers.humanoid_wrapper import HumanoidWrapper
from data_struct.states import GeneralizedState
from algorithms.gmm_ppo import PPO, PPOConfigs, PPOTrainingState
from networks import GCMLP, Multi_Action_PPO_Policy, Selector



vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 3e-4 # unified
critic_learning_rate = 5e-4
rollout_length = 32

description = {
        "task": "Test simple ant PPO",
        "policy_learning_rate": policy_learning_rate_per_std,
        "critic_learning_rate": critic_learning_rate,
        "architecture": "Simple MLP for both networks",
        "learnable std": False,
        "vec_env": vec_env,
        "batchsize": mini_batch_size,
        "rollout_length": rollout_length,
        "iterations": num_iterations,
        "policy epoch": policy_epochs,
        "critic_epochs": critic_epochs,
    }

description_text = "\n".join(
    [f"{i}: {j}" for i, j in description.items()]
)


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


# seed = 8848
seed = 4242
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

# # creat environment (Humanoid)
env = envs.create(env_name="humanoid", episode_length=4096, backend="mjx", auto_reset=True, action_repeat=2)
env = HumanoidWrapper(env)
component_means = jnp.concatenate([
    jnp.zeros(env.action_size), jax.random.normal(subkey, shape=(3 * env.action_size)) * 0.25
])


structure = "simple"
critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
selector_hidden_layers: Tuple[int, ...] = (128, 128)
policy_network = Multi_Action_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
    component_num=4,
    component_means=component_means,   
)

selector_network = Selector(
    hidden_layer_sizes=selector_hidden_layers,
    component_num=4,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
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
    selector_network=selector_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
ppo_training_state = ppo.init(subkey)

seed = 114514
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)
carry = (states, ppo_training_state, loop_random_key)


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
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 22: 23]**2, axis=-1))

    
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.25, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        sampled_states,
        final_states,
    )

    new_carry = (
        new_states,
        ppo_training_state,
        loop_random_key,
    )

    return new_carry, (aux_data, jnp.mean(vs))


wandb.init(
    entity="airl-lab",
    group="GMM tests",
    project="TBQDRL",
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
        "selection entropy": jnp.mean(stacked_aux_data.selection_entropy),
        "iteration_mean_v": jnp.mean(iteration_mean_v), 
        })

    carry = (states, ppo_training_state, loop_random_key)

# (
#     final_states, 
#     final_ppo_training_state, 
#     loop_random_key,
# ) = carry

# model_bytes = serialization.to_bytes(final_ppo_training_state.policy_params)
# critic_bytes = serialization.to_bytes(final_ppo_training_state.critic_params)
# fitness_critic_bytes = serialization.to_bytes(final_ppo_training_state.fitness_critic_params)

# with open(folder_path + f"/model_{structure}.msgpack", "wb") as f:
#     f.write(model_bytes)

# with open(folder_path + f"/critic_{structure}.msgpack", "wb") as f:
#     f.write(critic_bytes)

# with open(folder_path + f"/fitness_critic_{structure}.msgpack", "wb") as f:
#     f.write(fitness_critic_bytes)

wandb.finish()

