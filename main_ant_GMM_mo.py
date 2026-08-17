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
from algorithms.gmm_ppo import PPO, PPOConfigs, PPOTrainingState
# from data_struct.transitions import PPOTransition
from networks import GCMLP, Multi_Action_PPO_Policy, Selector, ComplexGCMLP
# from functools import partial
from flax import serialization
# from task_wrappers.ant_wrapper import AntWrapper
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from data_struct.states import GeneralizedState


vec_env = 4096
mini_batch_size = 8192
num_iterations = 2000
policy_epochs = 4
critic_epochs = 4
component_num = 4
policy_learning_rate_per_std = 1e-3
selector_learning_rate = 1e-4
critic_learning_rate = 5e-4
rollout_length = 32

description = {
        "task": "Test Ant MO GMM PPO",
        "policy_learning_rate": policy_learning_rate_per_std,
        "selector_learning_rate": selector_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "architecture": "Separate action, selector, and critic MLPs",
        "learnable std": True,
        "component_num": component_num,
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
    group="MORL GMM tests",
    project="TBQDRL",
    config=description,
)


# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# folder_path = f"./output/matern/output_{timestamp}"

# if not os.path.exists(folder_path):
#     os.makedirs(folder_path, exist_ok=True)
#     print(f"new folder <{folder_path}> created")


# with open(folder_path + "/description.log", "w") as f:
#     f.write(description_text)


ppo_config = PPOConfigs(
    policy_learning_rate_per_std=policy_learning_rate_per_std,
    selector_learning_rate=selector_learning_rate,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    approx_kl_threshold=0.0125,
    entropy_gain=0.001,
    selection_entropy_gain=0.0,
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

# # create environment (Ant)
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
env = AntMOWrapper(env)

structure = "simple"
critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
selector_hidden_layers: Tuple[int, ...] = (128, 128)
policy_network = Multi_Action_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    component_num=component_num,
    initial_std_logits=None,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
)

selector_network = Selector(
    hidden_layer_sizes=selector_hidden_layers,
    component_num=component_num,
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

    (final_states, sampled_states, ppo_training_state, loop_random_key), metrics = ppo.train(
        states,
        ppo_training_state,
        loop_random_key,
    )
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 13: 15]**2, axis=-1))

    resampled_states = jax.vmap(env.resample_task_state)(final_states)
    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.25, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        resampled_states,
        final_states,
    )

    new_carry = (
        new_states,
        ppo_training_state,
        loop_random_key,
    )

    return new_carry, (
        metrics.critic_rmse,
        metrics.policy_approx_kl,
        metrics.average_reward,
        metrics.average_return,
        metrics.done_count,
        jnp.mean(vs),
        metrics.gae_mean,
        metrics.gae_std,
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
            iteration_average_reward,
            iteration_mean_return,
            iteration_done_count,
            iteration_mean_v,
            gae_means,
            gae_stds,
            ) = jax.lax.scan(
        training_loop,
        carry,
        length=log_period,
    )


    wandb.log({
        "critic_RMSE": jnp.mean(iteration_critic_error),
        "approx_kl": jnp.mean(iteration_approx_kl),
        "average_reward": jnp.mean(iteration_average_reward),
        "iteration mean return": jnp.mean(iteration_mean_return),
        "done_count": jnp.mean(iteration_done_count),
        "iteration_mean_v": jnp.mean(iteration_mean_v),
        "GAE mean": jnp.mean(gae_means),
        "GAE std": jnp.mean(gae_stds),
        })

    carry = (states, ppo_training_state, loop_random_key)

    # if jnp.mean(iteration_mean_return) > 70:
    #     print("early break!")
    #     break


# =================================
#      Save model parameters
# =================================

(
    final_states,
    final_ppo_training_state,
    loop_random_key,
) = carry

model_bytes = serialization.to_bytes(final_ppo_training_state.policy_params)
selector_bytes = serialization.to_bytes(final_ppo_training_state.selector_params)
critic_bytes = serialization.to_bytes(final_ppo_training_state.critic_params)


# folder_path = f"./output/MORL/ant_gmm_mo_long"

# if not os.path.exists(folder_path):
#     os.makedirs(folder_path, exist_ok=True)
#     print(f"new folder <{folder_path}> created")

# with open(folder_path + "/policy.msgpack", "wb") as f:
#     f.write(model_bytes)

# with open(folder_path + "/selector.msgpack", "wb") as f:
#     f.write(selector_bytes)

# with open(folder_path + "/critic.msgpack", "wb") as f:
#     f.write(critic_bytes)

# jnp.save(folder_path + "/mean.npy", final_ppo_training_state.moving_mean)
# jnp.save(folder_path + "/var.npy", final_ppo_training_state.moving_squared_diff)
# jnp.save(folder_path + "/mse.npy", final_ppo_training_state.moving_mse)

wandb.finish()
