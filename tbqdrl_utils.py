from buffer import PPOTransition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState, Env
import jax
import jax.numpy as jnp
import math


def heuristic_sample_task(
    obs: jnp.ndarray, 
    key: RNGKey,
    ) -> jnp.ndarray:
    
    return


def sample_task(
    task: jnp.ndarray, 
    key: RNGKey,
    ) -> jnp.ndarray:
    
    return


def trajectory_reward(
    old_pipeline_state: jnp.ndarray,
    new_pipline_state: jnp.ndarray,
    task: jnp.ndarray,
    ) -> jnp.ndarray:

    return # shape of (1, 1)



def trajectory_transition_fn( 
    state: EnvState, 
    task: jnp.ndarray,
    action_mean: jnp.ndarray, 
    action_std: jnp.ndarray, 
    key: RNGKey,
    env: Env,
    ) -> Tuple[EnvState, PPOTransition]:

    key, subkey = jax.random.split(key)
    candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
    action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
    action_noise = action - action_mean
    next_state = env.step(state, action)
    fitness_rewards = jnp.ones((1, 1)) * (next_state.reward - next_state.metrics["x_velocity"])

    reward = trajectory_reward(state.pipeline_state, next_state.pipeline_state, task)
    next_task = sample_task(task, key)
    
    transition = PPOTransition(
        obs=state.obs,
        actions=action,
        action_noises=action_noise,
        tasks=task, 
        rewards=reward,
        fitness_rewards=fitness_rewards,
        td_lambda_returns=jnp.zeros((1, 1)),
        gaes=jnp.zeros(shape=(1, 1)),
        dones=jnp.where(next_state.done, jnp.ones(shape=(1, 1)), jnp.zeros(shape=(1, 1))),
        truncations=jnp.zeros(shape=(1, 1)),
        weights=jnp.zeros(shape=(1, 1)),
        )

    return (next_state, next_task), transition





