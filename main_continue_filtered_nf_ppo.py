"""Continue trajectory PPO using the critic-filtered flow distribution."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from brax import envs
from flax import serialization

from algorithms.trajectory_ppo import PPO, PPOConfigs, PPOTrainingState
from custom_types import RNGKey
from data_struct.states import GeneralizedState
from jax_rq_nsf import FlowConfig, init_flow
from networks import GCMLP, PPO_Policy
from task_wrappers.nf_trajectory_following import AntFiniteMaternWrapper


vec_env = 4096
mini_batch_size = 16384
num_iterations = 2000
policy_epochs = 4
critic_epochs = 4
fitness_critic_epochs = 4
policy_learning_rate_per_std = 8e-4
critic_learning_rate = 5e-4
rollout_length = 96

flow_checkpoint_path = Path(
    "output/vi_filtered/trajectory_flow_variables.msgpack"
)
actor_checkpoint_path = Path(
    "output/nf_matern/saved_data/policy.msgpack"
)
critic_checkpoint_path = Path(
    "output/nf_matern/saved_data/critic.msgpack"
)
output_directory = Path("output/vi_filtered_ppo/saved_data")

flow_config = FlowConfig(
    dimension=50,
    num_layers=8,
    hidden_features=(128, 128),
    num_bins=16,
    tail_bound=6.0,
)

description = {
    "task": "Continue trajectory PPO with critic-filtered NF",
    "flow_checkpoint": str(flow_checkpoint_path),
    "actor_checkpoint": str(actor_checkpoint_path),
    "critic_checkpoint": str(critic_checkpoint_path),
    "optimizer_state": "fresh warm-start optimizer state",
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
    "fitness_critic_epochs": fitness_critic_epochs,
}

description_text = "\n".join(
    [f"{key}: {value}" for key, value in description.items()]
)

output_directory.mkdir(parents=True, exist_ok=True)
(output_directory / "description.log").write_text(description_text)

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


flow, flow_variables_template = init_flow(
    jax.random.PRNGKey(0),
    flow_config,
)
flow_variables = serialization.from_bytes(
    flow_variables_template,
    flow_checkpoint_path.read_bytes(),
)


seed = 8848
loop_random_key = jax.random.PRNGKey(seed)

env = envs.create(
    env_name="ant",
    episode_length=4096,
    backend="mjx",
    reset_noise_scale=0.0,
)
env = AntFiniteMaternWrapper(
    env,
    flow,
    inner_radius=0.5,
    max_radius=2,
    tolerance_radius=0.1,
)

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
policy_params = serialization.from_bytes(
    ppo_training_state.policy_params,
    actor_checkpoint_path.read_bytes(),
)
critic_params = serialization.from_bytes(
    ppo_training_state.critic_params,
    critic_checkpoint_path.read_bytes(),
)
ppo_training_state = ppo_training_state.replace(
    policy_params=policy_params,
    critic_params=critic_params,
)

seed = 8848
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
initial_states = jax.vmap(env.reset, in_axes=(0, None))(
    subkeys,
    flow_variables,
)

carry = (initial_states, ppo_training_state, loop_random_key)


@jax.jit
def training_loop(
    carry: Tuple[GeneralizedState, PPOTrainingState, RNGKey],
    _: None,
) -> Tuple[Tuple, Tuple]:
    states, ppo_training_state, loop_random_key = carry

    (
        final_states,
        sampled_states,
        ppo_training_state,
        loop_random_key,
    ), aux_data = ppo.train(
        states,
        ppo_training_state,
        loop_random_key,
    )
    vs = jnp.sqrt(
        jnp.sum(
            sampled_states.env_state.obs[:, 13:15] ** 2,
            axis=-1,
        )
    )
    states = jax.vmap(
        env.resample_task_state,
        in_axes=(0, None),
    )(initial_states, flow_variables)

    new_carry = (
        states,
        ppo_training_state,
        loop_random_key,
    )
    return new_carry, (aux_data, jnp.mean(vs))


run = wandb.init(
    entity="airl-lab",
    project="TBQDRL",
    group="Trajectory filtered-NF PPO continuation",
    config=description,
)

log_period = 10

for _ in range(int(num_iterations / log_period)):
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
        "iteration mean return": jnp.mean(
            stacked_aux_data.average_return
        ),
        "iteration mean reward": jnp.mean(
            stacked_aux_data.average_reward
        ),
        "dones per episode": (
            jnp.mean(stacked_aux_data.done_count) / vec_env
        ),
        "iteration_mean_v": jnp.mean(iteration_mean_v),
    })

    carry = (states, ppo_training_state, loop_random_key)

(
    _,
    final_ppo_training_state,
    loop_random_key,
) = carry

(output_directory / "policy.msgpack").write_bytes(
    serialization.to_bytes(final_ppo_training_state.policy_params)
)
(output_directory / "critic.msgpack").write_bytes(
    serialization.to_bytes(final_ppo_training_state.critic_params)
)

run.finish()
