from buffer import PPOTransition
from typing import Tuple, Any, Callable
from custom_types import RNGKey, Params, EnvState, Env
import jax
import jax.numpy as jnp
import math



def mo_ppo_exploraive_rollout(
    policy_fn: Callable,
    env_fn: Callable,
    rollout_length: int,
) -> Callable:

    
    @jax.jit
    def explorative_rollout_fn(
        policy_params: Params,
        starting_states: EnvState,
        last_action_means: jnp.ndarray,
        keys: RNGKey,
        ) -> PPOTransition:

        def play_step_fn(
            carry: Tuple[EnvState, jnp.ndarray, int, RNGKey],
            ) -> Tuple[Tuple, PPOTransition]:
            
            state, last_action_mean, key = carry
            key, subkey = jax.random.split(key)
            obs = jnp.concatenate([state.obs, last_action_mean], axis=-1)
            # obs = state.obs

            action_mean, action_std = policy_fn(policy_params, obs)
            candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
            action_noise = action - action_mean
            action_log_std = jnp.log(action_std + 1e-6)

            next_state = env_fn(state, action)
            # rewards=jnp.ones((1, )) * state.reward
            rewards = jnp.array([
                state.metrics["reward_forward"] + 1, 
                state.metrics["reward_ctrl"] + 0.25, 
                # 0.2 * jnp.mean(action_log_std) + 0.6, 
                state.pipeline_state.x.pos[0, 2],
                1 - 2.5*jnp.mean(jnp.square(action_mean - last_action_mean)) # zero'th order smoothness
                ])

            transition = PPOTransition(
                obs=obs,
                actions=action,
                action_noises=action_noise,
                action_log_std=action_log_std,
                # rewards=jnp.clip(state.reward - state.metrics["forward_reward"] + 3.0, min=0.0),
                rewards=rewards,
                td_lambda_returns=jnp.zeros((1,)),
                baselines=jnp.zeros((1,)),
                gaes=jnp.zeros((1,)),
                dones=next_state.done,
                # truncations=jnp.where(step_num < truncate_length, 0.0, 1.0),
                truncations=0.0,
                weights=jnp.zeros((1,)),
                )

            return (next_state, action_mean, key), transition

        final_carry, transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, last_action_means, keys),
            length=rollout_length,
        )
        
        final_states, final_action, _ = final_carry

        return final_states, final_action, transitions
    
    return explorative_rollout_fn






def calculate_td_lambda_returns(
    final_v_value: jnp.ndarray,
    v_values: jnp.ndarray, 
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    discount: float, 
    td_lambda_discount: float,
) -> jnp.ndarray:

    def scan_calculate_td_lambda(
        carry: jnp.ndarray, 
        data: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        (last_td_lambda_value, last_value, last_weight) = carry
        reward, v_value, mask = data
        current_td_lambda_value = reward + mask * discount * (
                (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
            )
        weight = discount * td_lambda_discount * (last_weight - 1) * mask + 1

        return (current_td_lambda_value, v_value, weight), (current_td_lambda_value, weight)
        
    _, (td_lambda_values, weights) = jax.lax.scan(
        jax.vmap(scan_calculate_td_lambda),
        (final_v_value, final_v_value, jnp.zeros_like(final_v_value)),
        (rewards, v_values, masks),
        reverse=True,
    ) # length x batch x d

    return td_lambda_values, weights
        



def shuffle_transitions(key: RNGKey, transitions: PPOTransition) -> PPOTransition:
    flattened_transitions = transitions.flatten()
    num_transitions = flattened_transitions.shape[0]
    index = jax.random.permutation(key, num_transitions)
    transitions = transitions.__class__.from_flatten(flattened_transitions[index], transitions)
    
    return transitions


def vx_transition_fn( 
    state: EnvState, 
    action_mean: jnp.ndarray, 
    action_std: jnp.ndarray, 
    key: RNGKey,
    env: Env,
    ) -> Tuple[EnvState, PPOTransition]:

    candidate_action_noise = action_std * jax.random.normal(key, action_mean.shape)
    action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
    action_noise = action - action_mean
    action_log_std = jnp.log(action_std + 1e-6)

    next_state = env.step(state, action)
    rewards = jnp.atleast_1d(next_state.reward)

    transition = PPOTransition(
        obs=state.obs,
        actions=action,
        action_noises=action_noise,
        action_log_std=action_log_std,
        rewards=rewards,
        preferences=jnp.zeros_like(rewards),
        td_lambda_returns=jnp.zeros((1,)),
        baselines=jnp.zeros((1,)),
        gaes=jnp.zeros((1,)),
        dones=jnp.atleast_1d(next_state.done),
        # truncations=jnp.atleast_1d(next_state.info['truncation']),
        truncations=jnp.zeros((1,)),
        weights=jnp.zeros((1,)),
        )

    return next_state, transition


def vy_transition_fn( 
    state: EnvState, 
    action_mean: jnp.ndarray, 
    action_std: jnp.ndarray, 
    key: RNGKey,
    env: Env,
    ) -> Tuple[EnvState, PPOTransition]:

    candidate_action_noise = action_std * jax.random.normal(key, action_mean.shape)
    action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
    action_noise = action - action_mean
    action_log_std = jnp.log(action_std + 1e-6)

    next_state = env.step(state, action)
    rewards = jnp.atleast_1d(
        next_state.reward - next_state.metrics["x_velocity"] + next_state.metrics["y_velocity"]
        )

    transition = PPOTransition(
        obs=state.obs,
        actions=action,
        action_noises=action_noise,
        action_log_std=action_log_std,
        rewards=rewards,
        preferences=jnp.zeros_like(rewards),
        td_lambda_returns=jnp.zeros((1,)),
        baselines=jnp.zeros((1,)),
        gaes=jnp.zeros((1,)),
        dones=jnp.atleast_1d(next_state.done),
        truncations=jnp.zeros((1,)),
        weights=jnp.zeros((1,)),
        )

    return next_state, transition



def xy_transition_fn( 
    state: EnvState, 
    action_mean: jnp.ndarray, 
    action_std: jnp.ndarray, 
    key: RNGKey,
    env: Env,
    ) -> Tuple[EnvState, PPOTransition]:

    candidate_action_noise = action_std * jax.random.normal(key, action_mean.shape)
    action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
    action_noise = action - action_mean
    action_log_std = jnp.log(action_std + 1e-6)

    next_state = env.step(state, action)
    rewards = jnp.atleast_1d(
        next_state.reward - 0.292893 * next_state.metrics["x_velocity"] + 0.707107 * next_state.metrics["y_velocity"]
        )

    transition = PPOTransition(
        obs=state.obs,
        actions=action,
        action_noises=action_noise,
        action_log_std=action_log_std,
        rewards=rewards,
        preferences=jnp.zeros_like(rewards),
        td_lambda_returns=jnp.zeros((1,)),
        baselines=jnp.zeros((1,)),
        gaes=jnp.zeros((1,)),
        dones=jnp.atleast_1d(next_state.done),
        truncations=jnp.zeros((1,)),
        weights=jnp.zeros((1,)),
        )

    return next_state, transition



def diag_transition_fn( 
    state: EnvState, 
    action: jnp.ndarray, 
    env: Env,
    ) -> Tuple[EnvState, PPOTransition]:

    next_state = env.step(state, action)
    rewards = jnp.atleast_1d(
        next_state.reward - 0.292893 * next_state.metrics["x_velocity"] + 0.707107 * next_state.metrics["y_velocity"]
        )

    transition = PPOTransition(
        obs=state.obs,
        actions=action,
        action_noises=jnp.zeros_like(action),
        action_log_std=jnp.zeros_like(action),
        rewards=rewards,
        preferences=jnp.zeros_like(rewards),
        td_lambda_returns=jnp.zeros((1,)),
        baselines=jnp.zeros((1,)),
        gaes=jnp.zeros((1,)),
        dones=jnp.atleast_1d(next_state.done),
        truncations=jnp.zeros((1,)),
        weights=jnp.zeros((1,)),
        )

    return next_state, transition


