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
from algorithms.test_gmm_ppo import PPO, PPOConfigs, PPOTrainingState
# from data_struct.transitions import PPOTransition
from networks import GCMLP, GC_multi_Policy, GC_Selector
# from functools import partial
from flax import serialization
from task_wrappers.ant_wrapper import AntWrapper
from data_struct.states import GeneralizedState

vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 8e-4 # unified
critic_learning_rate = 5e-4
selector_learning_rate = 3e-4
rollout_length = 32

description = {
        "task": "Decoupled GMM PPO test",
        "policy_learning_rate": policy_learning_rate_per_std,
        "critic_learning_rate": critic_learning_rate,
        "selector_learning_rate": selector_learning_rate,
        "architecture": "Separate MLPs for action means and selector",
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


wandb.init(
    entity="airl-lab",
    group="GMM tests",
    project="TBQDRL",
    config=description,
)


# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f"./output/GMM_decoupled"

if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print(f"new folder <{folder_path}> created")

with open(folder_path + "/description.log", "w") as f:
    f.write(description_text)


ppo_config = PPOConfigs(
    policy_learnng_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    selector_learning_rate=selector_learning_rate,
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


seed = 6677
# seed = 42
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)

# # creat environment (Ant)
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
env = AntWrapper(env)
component_means = jnp.concatenate([
    jnp.zeros(env.action_size), jax.random.normal(subkey, shape=(3 * env.action_size)) * 0.25
])

structure = "simple"
critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
selector_hidden_layers: Tuple[int, ...] = (128, 128)
policy_network = GC_multi_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=4,
    component_means=component_means,   
)

selector_network = GC_Selector(
    hidden_layer_sizes=selector_hidden_layers,
    component_num=4,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.softplus,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    # final_activation=lambda x: x,
)

ppo = PPO(
    env=env,
    policy_network=policy_network,
    selector_network=selector_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
    std_anneal_fn=lambda x: jnp.maximum(0.05, 0.5 - x * 1e-4),
)

loop_random_key, subkey = jax.random.split(loop_random_key)
ppo_training_state = ppo.init(subkey)

seed = 8848
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
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 13: 15]**2, axis=-1))

    
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.5, shape=(vec_env,))

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

    return new_carry, (
        aux_data.training_data.critic_error,
        aux_data.training_data.approx_kl,
        aux_data.training_data.clip_fraction,
        # aux_data.rollout_data.average_reward, 
        aux_data.rollout_data.average_return,
        # aux_data.rollout_data.average_lifespan,
        jnp.mean(vs),
        )

log_period = 10

for i in range(int(num_iterations / log_period)):

    (
        states, 
        ppo_training_state, 
        loop_random_key,
        ), (
            iteration_critic_error,
            iteration_approx_kl,
            iteration_clip_fraction,
            iteration_mean_return,
            # iteration_mean_lifespan,
            iteration_mean_v,
            ) = jax.lax.scan(
        training_loop,
        carry,
        length=log_period,
    )


    wandb.log({
        "critic_RMSE": jnp.mean(iteration_critic_error),
        "approx_kl": jnp.mean(iteration_approx_kl),
        "clip_fraction": jnp.mean(iteration_clip_fraction),
        "iteration mean return": jnp.mean(iteration_mean_return), 
        "iteration_mean_v": jnp.mean(iteration_mean_v), 
        })
    
    print("v", jnp.mean(iteration_mean_v), "\t", "return", jnp.mean(iteration_mean_return))

    carry = (states, ppo_training_state, loop_random_key)

(
    final_states, 
    final_ppo_training_state, 
    loop_random_key,
) = carry

model_bytes = serialization.to_bytes(final_ppo_training_state.policy_params)
critic_bytes = serialization.to_bytes(final_ppo_training_state.critic_params)
selector_bytes = serialization.to_bytes(final_ppo_training_state.selector_params)


# with open(folder_path + f"/policy.msgpack", "wb") as f:
#     f.write(model_bytes)

# with open(folder_path + f"/critic.msgpack", "wb") as f:
#     f.write(critic_bytes)

# with open(folder_path + f"/selector.msgpack", "wb") as f:
#     f.write(selector_bytes)

# with open(folder_path + f"/fitness_critic_{structure}.msgpack", "wb") as f:
#     f.write(fitness_critic_bytes)

wandb.finish()
