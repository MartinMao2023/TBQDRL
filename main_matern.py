import jax
import flax.linen as nn
import jax.numpy as jnp
from wrappers import AutoResetWrapper
from brax import envs
import os
from datetime import datetime
import wandb
from custom_types import RNGKey, EnvState, Params
from typing import Any, Tuple, List
from algorithms import TrajectoryPPO, PPOConfigs, PPOTrainingState
from data_struct.transitions import PPOTransition
from networks import GCMLP, GC_PPO_Policy, ComplexGCMLP, ComplexGCPPO_Policy
# from functools import partial
from trajectory_encoder import TaskState, MaternEncoder, CircularEncoder, LineEncoder
from flax import serialization

vec_env = 4096
mini_batch_size= 8192
l = 2
num_iterations = 20000
policy_epochs = 4
critic_epochs = 6
horizon = 2
policy_learning_rate = 3e-4
critic_learning_rate = 5e-4

description = {
        "task": "MaternEncoder",
        "l": l,
        "horizon": horizon,
        "var": 2.25,
        "policy_learning_rate": policy_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "architecture": "Simple MLP for both networks",
        "learnable std": False,
        "vec_env": vec_env,
        "batchsize": mini_batch_size,
        "rollout_length": 64,
        "iterations": num_iterations,
        "policy epoch": policy_epochs,
        "critic_epochs": critic_epochs,
    }

description_text = "\n".join(
    [f"{i}: {j}" for i, j in description.items()]
)


wandb.init(
    entity="airl-lab",
    project="TBQDRL",
    config=description,
)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = f"./output/matern/output_{timestamp}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print(f"new folder <{folder_path}?> created")


with open(folder_path + "/description.log", "w") as f:
    f.write(description_text)


ppo_config = PPOConfigs(
    policy_learnng_rate=policy_learning_rate,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.005,
    discount=0.99,
    td_lambda_discount=0.95,
    rollout_length=64,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
    initial_std=0.5,
    std_decay_rate=0.0001,
    min_std=0.1,
)

# seed = 8848
seed = 42
loop_random_key = jax.random.PRNGKey(seed)

# creat environment (Ant)
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)


structure = "simple"
critic_hidden_layers: Tuple[int, ...] = (64, 64)
actor_hidden_layers: Tuple[int, ...] = (64, 64)
policy_network = GC_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=False,
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.softplus,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    final_activation=lambda x: x - 3,
)


# structure = "complex"
# critic_hidden_layers: Tuple[int, ...] = ((64, 16), (64, 16))
# actor_hidden_layers: Tuple[int, ...] = ((64, 16), (64, 16))
# policy_network = ComplexGCPPO_Policy(
#     hidden_layer_sizes=actor_hidden_layers,
#     action_dim=env.action_size,
#     kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
#     kernel_init_final=jax.nn.initializers.orthogonal(0.01),
#     activation=nn.softplus,
#     final_activation=jnp.tanh,
#     learnable_std=False,
# )

# critic_network = ComplexGCMLP(
#     hidden_layer_sizes=critic_hidden_layers,
#     output_size=1,
#     kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
#     activation=nn.softplus,
#     kernel_init_final=jax.nn.initializers.orthogonal(0.01),
#     final_activation=lambda x: x - 3,
# )


task_encoder = MaternEncoder(l=l, horizon=horizon)

ppo = TrajectoryPPO(
    env=env,
    policy_network=policy_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
    encoder=task_encoder,
)

loop_random_key, subkey = jax.random.split(loop_random_key)
ppo_training_state = ppo.init(subkey)

seed = 42
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)

loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
task_states = jax.vmap(task_encoder.reset)(states.obs, subkeys)


carry = (states, task_states, ppo_training_state, loop_random_key)

def training_loop(
    carry: Tuple[EnvState, TaskState, PPOTrainingState, RNGKey], 
    _: None,
    ) -> Tuple[Tuple, Tuple]:

    states, task_states, ppo_training_state, loop_random_key = carry

    (states, task_states, ppo_training_state, loop_random_key), aux_data, transitions = ppo.train(
        states,
        task_states,
        ppo_training_state,
        loop_random_key,
    )

    vs = jnp.sqrt(jnp.sum(states.obs[:, 13:15]**2, axis=-1))

    initial_states = states.replace(
        pipeline_state=states.info['first_pipeline_state'], 
        obs=states.info['first_obs'],
        )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, 1/8, shape=(vec_env,))
    states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        initial_states,
        states,
    )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    subkeys = jax.random.split(subkey, num=vec_env)
    candidate_task_states = jax.vmap(task_encoder.reset)(states.obs, subkeys)
    
    # loop_random_key, subkey = jax.random.split(loop_random_key)
    # ps = ps | jax.random.bernoulli(subkey, 1/7, shape=(vec_env,))
    task_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        candidate_task_states,
        task_states,
    )

    new_carry = (
        states,
        task_states,
        ppo_training_state,
        loop_random_key,
    )

    critic_error, approx_kl, clip_fraction = aux_data


    return new_carry, (
        critic_error,
        approx_kl,
        clip_fraction,
        jnp.mean(transitions.rewards), 
        jnp.mean(transitions.td_lambda_returns), 
        jnp.mean(vs))


iteration_mean_returns = []
iteration_mean_rewards = []
iteration_mean_vs = []

log_period = 20

for i in range(int(num_iterations / log_period)):
    (
        states, 
        task_states, 
        ppo_training_state, 
        loop_random_key,
        ), (
            iteration_critic_error,
            iteration_approx_kl,
            iteration_clip_fraction,
            iteration_mean_reward, 
            iteration_mean_return,
            iteration_mean_v,
            ) = jax.lax.scan(
        training_loop,
        carry,
        length=log_period,
    )

    iteration_mean_returns.append(iteration_mean_return)
    iteration_mean_rewards.append(iteration_mean_reward)
    iteration_mean_vs.append(iteration_mean_v)

    wandb.log({
        "critic_RMSE": iteration_critic_error[-1],
        "approx_kl": iteration_approx_kl[-1],
        "clip_fraction": iteration_clip_fraction[-1],
        "iteration mean return": iteration_mean_return[-1], 
        "iteration_mean_reward": iteration_mean_reward[-1],
        "iteration_mean_v": iteration_mean_v[-1], 
        # "iteartion num": (i + 1) * log_period
        }
        )

    if ((i + 1) * log_period) % 500 == 0:
        loop_random_key, subkey = jax.random.split(loop_random_key)
        subkeys = jax.random.split(subkey, num=vec_env)
        states = jax.vmap(env.reset)(subkeys)

        loop_random_key, subkey = jax.random.split(loop_random_key)
        subkeys = jax.random.split(subkey, num=vec_env)
        task_states = jax.vmap(task_encoder.reset)(states.obs, subkeys)

    carry = (states, task_states, ppo_training_state, loop_random_key)

(
    final_states, 
    final_task_states, 
    final_ppo_training_state, 
    loop_random_key,
) = carry

iteration_mean_returns = jnp.concatenate(iteration_mean_returns)
iteration_mean_rewards = jnp.concatenate(iteration_mean_rewards)
iteration_mean_vs = jnp.concatenate(iteration_mean_vs)


jnp.save(folder_path + "/matern_reward_curve.npy", iteration_mean_rewards)
jnp.save(folder_path + "/matern_return_curve.npy", iteration_mean_returns)
jnp.save(folder_path + "/matern_vs.npy", iteration_mean_vs)

model_bytes = serialization.to_bytes(final_ppo_training_state.policy_params)
critic_bytes = serialization.to_bytes(final_ppo_training_state.critic_params)

with open(folder_path + f"/matern_model_{structure}.msgpack", "wb") as f:
    f.write(model_bytes)

with open(folder_path + f"/matern_critic_{structure}.msgpack", "wb") as f:
    f.write(critic_bytes)

wandb.finish()

