import numpy as np
import numpy.linalg as lg
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple

from custom_types import RNGKey
from data_struct import MORelocationTransition
from data_struct.states import GeneralizedState



def calculate_smoothing_kernel(l, trunk_length):

    x = np.arange(trunk_length + 1).reshape(-1, 1)
    cov = np.exp(- 0.5 * np.square((x - x.T)/l))
    cov = 0.0625 * cov + np.ones_like(cov) * 100
    noisy_cov = cov + np.eye(trunk_length + 1) * 0.01
    smoothing_coefs = lg.solve(noisy_cov, cov[:, -1:])

    return smoothing_coefs # (trunk_length + 1, 1)


def calculate_coefs_for_trunk(
    total_length: int, 
    start_index: int,
    end_index: int,
    l: float,
    discount: float = 0.99,
    td_lambda_discount: float = 0.95,
    ) -> Tuple[jax.Array, jax.Array]:
    """
    Args:
        total_length: total number of transitions
        start_index: starting index of the state sequence
        end_index: final index of the state sequence, maximum value is total_length

    Returns:
        reward_coefs: jax.Array shape of (total_length, 1)
        value_coefs: jax.Array shape of (total_length + 1, 1)
    """

    v_coefs = np.zeros((total_length + 1, 1))
    log_discount = np.log(td_lambda_discount * discount)
    
    trunk_length = end_index - start_index
    smoothing_coefs = calculate_smoothing_kernel(l, trunk_length) # (trunk_length + 1, 1)
    final_weight = discount / (1 - discount * td_lambda_discount) * (
        1 - np.exp(log_discount * trunk_length)
        )
    usual_weights = discount / (1 - discount * td_lambda_discount) * (
        1 - np.exp(log_discount * np.arange(trunk_length + 1))
        ) * (1 - td_lambda_discount) - 1.0 # (trunk_length + 1,)
    usual_weights[-1] = 0
    trunk_coefs = smoothing_coefs * final_weight # (trunk_length + 1, 1)
    trunk_coefs = trunk_coefs + usual_weights[:, None]
    v_coefs[start_index: end_index + 1, :] = trunk_coefs

    reward_coefs = np.arange(total_length) - start_index # (total_length,)
    reward_coefs = (1 - np.exp(log_discount * reward_coefs + log_discount)) / (1 - discount * td_lambda_discount)
    
    condition = (start_index <= np.arange(total_length)) & (np.arange(total_length) < end_index)
    reward_coefs = np.where(
        condition,
        reward_coefs,
        0.0,
    ) # (total_length,)

    return jnp.array(reward_coefs[:, None], dtype=jnp.float32), jnp.array(v_coefs, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Teacher-distillation dataset (Stage 3 prep)
# ---------------------------------------------------------------------------

def with_policy_action_mean(transitions, policy_network, policy_params):
    """Return a copy of ``transitions`` with the ``actions`` field replaced by
    the teacher policy's deterministic action mean evaluated at each
    (obs, z).

    The input is left unchanged so it can still be passed to relocation.
    Used to build the teacher-distillation dataset.
    """
    lead_shape = transitions.obs.shape[:-1]
    obs = transitions.obs.reshape(-1, transitions.observation_dim)
    zs = transitions.zs.reshape(-1, transitions.z_dim)
    action_mean, _ = jax.vmap(policy_network.apply, in_axes=(None, 0, 0))(
        policy_params, obs, zs,
    )
    action_mean = action_mean.reshape(*lead_shape, action_mean.shape[-1])
    return transitions.replace(actions=action_mean)


# ---------------------------------------------------------------------------
# On-policy rollout (Stage 1)
# ---------------------------------------------------------------------------

def build_on_policy_rollout(env, policy_network, policy_params, rollout_length):
    """Build a rollout function that fills a MORelocationTransition.

    The teacher ``policy_params`` are captured here and fixed for the lifetime
    of the returned function.

    The returned function performs one rollout sweep of length ``rollout_length``
    across the parallel environments using the (Gaussian) teacher policy.
    ``play_step_fn`` is vmapped over envs and scanned over time.  Not jitted
    here because the caller jits the outer scan anyway.

    Returns:
        rollout_fn(starting_states, keys) ->
            (final_states, transitions)  where transitions fields have shape
            (rollout_length, vec_env, ...).
    """

    def rollout_fn(
        starting_states: GeneralizedState,
        keys: RNGKey,
    ) -> Tuple[GeneralizedState, MORelocationTransition]:

        action_std = jax.nn.sigmoid(policy_params["params"]["std_logits"])
        # Teacher (Gaussian PPO) log-prob helpers: log p_teacher(a) of the
        # sampled action, recorded in `log_likelihood` for the two-stage
        # distiller's importance-sampling stage. Drop 2*pi; include the
        # log-variance term (-sum log_std). std is shared, so this is the
        # single-Gaussian limit of the GMM log-likelihood formula.
        teacher_log_std = nn.log_sigmoid(policy_params["params"]["std_logits"])
        teacher_inv_var = jnp.exp(-2.0 * teacher_log_std)

        def play_step_fn(carry):
            state, key = carry
            obs, z = env.get_obs(state)
            action_mean, _ = policy_network.apply(policy_params, obs, z)

            key, subkey = jax.random.split(key)
            noise = action_std * jax.random.normal(subkey, action_mean.shape)
            action = jnp.clip(action_mean + noise, -1.0, 1.0)

            last_action = state.z_state.last_action

            # log p_teacher(action | obs, z) of the executed (clipped) action
            log_likelihood = (
                -0.5 * jnp.sum(
                    teacher_inv_var * jnp.square(action - action_mean),
                    axis=-1, keepdims=True,
                )
                - jnp.sum(teacher_log_std, axis=-1, keepdims=True)
            )  # (1,)

            state, transition_info = env.step(state, action)

            transition = MORelocationTransition(
                obs=obs,
                zs=z,
                actions=action,
                # actions=action_mean,
                last_actions=last_action,
                mo_rewards=transition_info.mo_reward,
                weights=jnp.zeros((1,)),
                dones=transition_info.done,
                truncations=transition_info.truncation,
                td_lambda_returns=jnp.zeros((1,)),
                log_likelihood=log_likelihood,
            )
            return (state, key), transition

        (final_states, _), transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, keys),
            length=rollout_length,
        )
        return final_states, transitions

    return rollout_fn


# ---------------------------------------------------------------------------
# Dropout rollout (Stage 2, optional)
# ---------------------------------------------------------------------------

def build_dropout_rollout(
    env, policy_network, policy_params, rollout_length,
):
    """Like ``build_on_policy_rollout`` but applies inverted dropout to the
    teacher policy's hidden activations at every step (MC-dropout style),
    yielding more diverse action means / trajectories.

    ``policy_network`` must be a dropout-enabled module (e.g.
    ``GC_PPO_Policy_Dropout``) sharing the teacher's parameter tree, and is
    applied with ``deterministic=False`` plus a per-step ``dropout`` RNG.
    Not jitted here because the caller jits the outer scan anyway.

    Returns:
        rollout_fn(starting_states, keys) ->
            (final_states, transitions)  with shape
            (rollout_length, vec_env, ...).
    """

    def rollout_fn(
        starting_states: GeneralizedState,
        keys: RNGKey,
    ) -> Tuple[GeneralizedState, MORelocationTransition]:

        action_std = jax.nn.sigmoid(policy_params["params"]["std_logits"])
        # Teacher log-prob helpers (same convention as build_on_policy_rollout):
        # for the dropout rollout the "policy making the demonstration" is the
        # dropout-perturbed teacher, so log p is evaluated at the
        # dropout-perturbed action_mean (computed per step below).
        teacher_log_std = nn.log_sigmoid(policy_params["params"]["std_logits"])
        teacher_inv_var = jnp.exp(-2.0 * teacher_log_std)

        # One dropout key per env, held constant for the whole rollout so the
        # same dropout-perturbed ("mutated") policy is used at every step.
        split_keys = jax.vmap(jax.random.split)(keys)  # (vec_env, 2, 2)
        step_keys = split_keys[:, 0]
        dropout_keys = split_keys[:, 1]

        def play_step_fn(carry):
            state, key, dropout_key = carry
            obs, z = env.get_obs(state)

            key, noise_key = jax.random.split(key)
            action_mean, _ = policy_network.apply(
                policy_params, obs, z,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )

            noise = action_std * jax.random.normal(noise_key, action_mean.shape)
            action = jnp.clip(action_mean + noise, -1.0, 1.0)

            last_action = state.z_state.last_action

            # log p of the executed action under the dropout-perturbed teacher
            log_likelihood = (
                -0.5 * jnp.sum(
                    teacher_inv_var * jnp.square(action - action_mean),
                    axis=-1, keepdims=True,
                )
                - jnp.sum(teacher_log_std, axis=-1, keepdims=True)
            )  # (1,)

            state, transition_info = env.step(state, action)

            transition = MORelocationTransition(
                obs=obs,
                zs=z,
                actions=action,
                # actions=action_mean,
                last_actions=last_action,
                mo_rewards=transition_info.mo_reward,
                weights=jnp.zeros((1,)),
                dones=transition_info.done,
                truncations=transition_info.truncation,
                td_lambda_returns=jnp.zeros((1,)),
                log_likelihood=log_likelihood,
            )
            # dropout_key is carried through unchanged
            return (state, key, dropout_key), transition

        (final_states, _, _), transitions = jax.lax.scan(
            lambda x, _: jax.vmap(play_step_fn)(x),
            (starting_states, step_keys, dropout_keys),
            length=rollout_length,
        )
        return final_states, transitions

    return rollout_fn






def calculate_coefs_for_trajectory(
    total_length: int, 
    start_index: int,
    end_index: int,
    l: float,
    discount: float = 0.99,
    td_lambda_discount: float = 0.95,
    ) -> Tuple[jax.Array, jax.Array]:
    """
    Args:
        total_length: total number of transitions
        start_index: starting index of the state sequence
        end_index: final index of the state sequence, maximum value is total_length

    Returns:
        reward_coefs: jax.Array shape of (total_length, 1)
        value_coefs: jax.Array shape of (total_length + 1, 1)
    """

    log_discount = jnp.log(td_lambda_discount * discount)
    trunk_length = end_index - start_index
    v_condition = (start_index <= jnp.arange(total_length + 1)) & (jnp.arange(total_length + 1) < end_index + 1)
    reward_condition = (start_index <= jnp.arange(total_length)) & (jnp.arange(total_length) < end_index)

    v_coefs = discount / (1 - discount * td_lambda_discount) * (
        1 - jnp.exp(log_discount * (jnp.arange(total_length + 1) - start_index))
        ) * (1 - td_lambda_discount) - 1.0 # (total_length + 1,)
    
    v_coefs = jnp.where(jnp.arange(total_length + 1) < end_index, v_coefs, 0.0)
    smoothing_coefs = jnp.exp(-(jnp.arange(total_length + 1) - end_index)**2/l**2)
    smoothing_coefs = jnp.where(v_condition, smoothing_coefs, 0.0)
    smoothing_coefs = smoothing_coefs / (jnp.sum(smoothing_coefs) + 1e-6)
    v_coefs = smoothing_coefs * discount / (1 - discount * td_lambda_discount) * (
        1 - jnp.exp(log_discount * trunk_length)) + v_coefs

    reward_coefs = jnp.arange(total_length) - start_index # (total_length,)
    reward_coefs = (1 - jnp.exp(log_discount * reward_coefs + log_discount)) / (1 - discount * td_lambda_discount)
    reward_coefs = jnp.where(
        reward_condition,
        reward_coefs,
        0.0,
    ) # (total_length,)

    return reward_coefs[:, None], v_coefs[:, None]




def calculate_coefs_by_trunk(
    total_length: int, 
    start_index: int,
    end_index: int,
    l: float,
    discount: float = 0.99,
    td_lambda_discount: float = 0.95,
    ) -> Tuple[jax.Array, jax.Array]:
    """
    Args:
        total_length: total number of transitions
        start_index: starting index of the state sequence
        end_index: final index of the state sequence, maximum value is total_length

    Returns:
        reward_coefs: jax.Array shape of (total_length, 1)
        value_coefs: jax.Array shape of (total_length + 1, 1)
    """

    log_discount = jnp.log(td_lambda_discount * discount)
    trunk_length = end_index - start_index
    v_condition = (start_index <= jnp.arange(total_length + 1)) & (jnp.arange(total_length + 1) < end_index + 1)
    reward_condition = (start_index <= jnp.arange(total_length)) & (jnp.arange(total_length) < end_index)

    v_coefs = discount / (1 - discount * td_lambda_discount) * (
        1 - jnp.exp(log_discount * (jnp.arange(total_length + 1) - start_index))
        ) * (1 - td_lambda_discount) - 1.0 # (total_length + 1,)
    
    v_coefs = jnp.where(jnp.arange(total_length + 1) < end_index, v_coefs, 0.0)
    smoothing_coefs = jnp.exp(-(jnp.arange(total_length + 1) - end_index)**2/l**2)
    smoothing_coefs = jnp.where(v_condition, smoothing_coefs, 0.0)
    smoothing_coefs = smoothing_coefs / (jnp.sum(smoothing_coefs) + 1e-6)
    v_coefs = smoothing_coefs * discount / (1 - discount * td_lambda_discount) * (
        1 - jnp.exp(log_discount * trunk_length)) + v_coefs

    reward_coefs = jnp.arange(total_length) - start_index # (total_length,)
    reward_coefs = (1 - jnp.exp(log_discount * reward_coefs + log_discount)) / (1 - discount * td_lambda_discount)
    reward_coefs = jnp.where(
        reward_condition,
        reward_coefs,
        0.0,
    ) # (total_length,)

    return reward_coefs[:, None], v_coefs[:, None]
