from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import optax
from flax.struct import PyTreeNode, dataclass
from jax import numpy as jnp

from custom_types import Params, RNGKey
from data_struct.states import GeneralizedState
from task_wrappers.base import BaseTaskWrapper


zero = jnp.float32(0.0)
one = jnp.float32(1.0)


@dataclass
class PPOConfigs:
    policy_learning_rate_per_std: float = 1e-3
    critic_learning_rate: float = 5e-4
    clip_ratio: float = 0.2
    approx_kl_threshold: float = 0.0125
    entropy_gain: float = 0.01
    discount: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 64
    vec_env: int = 256
    mini_batch_size: int = 4096
    critic_epochs: int = 4
    policy_epochs: int = 4


class PPORollout(PyTreeNode):
    """Rollout data with leading axes ``(time, environment)``."""

    obs: jax.Array
    zs: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    truncations: jax.Array


class TrainingData(PyTreeNode):
    """Policy and critic data with arbitrary leading batch dimensions."""

    obs: jax.Array
    zs: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    gaes: jax.Array
    target_values: jax.Array
    weights: jax.Array


class PPOTrainingState(PyTreeNode):
    policy_params: Params
    critic_params: Params
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    iteration_num: jax.Array
    moving_mean: jax.Array
    moving_squared_diff: jax.Array
    moving_mse: jax.Array
    lr_scale: jax.Array


class PPOMetrics(PyTreeNode):
    critic_rmse: jax.Array
    policy_approx_kl: jax.Array
    average_reward: jax.Array
    average_return: jax.Array
    done_count: jax.Array
    gae_mean: jax.Array
    gae_std: jax.Array


class PPO:
    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        ppo_configs: PPOConfigs,
    ):
        self._env = env
        self._policy_network = policy_network
        self._critic_network = critic_network
        self.configs = ppo_configs

        # Calculating configs
        self.samples_per_rollout = ppo_configs.vec_env * ppo_configs.rollout_length
        self.mini_batch_num = self.samples_per_rollout // ppo_configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2.0 / self.mini_batch_num)
        self._clip_log_ratio = jnp.log(1.0 + ppo_configs.clip_ratio)

        # defining optimizers
        self._policy_optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=ppo_configs.policy_learning_rate_per_std
        )
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate
        )


    @staticmethod
    def _log_probability(
        actions: jax.Array,
        action_mean: jax.Array,
        log_action_std: jax.Array,
        inv_action_var: jax.Array,
    ) -> jax.Array:
        """
        Log likelihood for Gaussian, log of 2*pi is ignored
        """
        return -jnp.sum(
            log_action_std + 0.5 * jnp.square(actions - action_mean) * inv_action_var,
            axis=-1,
            keepdims=True,
        )

    def _update_learning_rate(
        self,
        training_state: PPOTrainingState,
    ) -> optax.OptState:
        std_logits = training_state.policy_params["params"]["std_logits"]
        action_std = nn.sigmoid(std_logits)
        rms_std = jnp.sqrt(jnp.mean(jnp.square(action_std)))
        learning_rate = jnp.minimum(
            rms_std * self.configs.policy_learning_rate_per_std,
            3e-4,
        ) * training_state.lr_scale
        policy_opt_state = training_state.policy_opt_state
        policy_opt_state = policy_opt_state._replace(
            hyperparams={
                **policy_opt_state.hyperparams,
                "learning_rate": learning_rate,
            }
        )
        return policy_opt_state


    def init(self, key: RNGKey) -> PPOTrainingState:
        fake_obs = jnp.zeros((self._env.observation_size,))
        fake_zs = jnp.zeros((self._env.z_size,))

        key, policy_key, critic_key = jax.random.split(key, 3)
        policy_params = self._policy_network.init(
            policy_key, obs=fake_obs, z=fake_zs
        )
        critic_params = self._critic_network.init(
            critic_key, obs=fake_obs, z=fake_zs
        )
        policy_opt_state = self._policy_optimizer.init(policy_params)
        training_state = PPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=self._critic_optimizer.init(critic_params),
            iteration_num=jnp.asarray(0, dtype=jnp.int32),
            moving_mean=zero,
            moving_squared_diff=one,
            moving_mse=zero,
            lr_scale=one,
        )
        policy_opt_state = self._update_learning_rate(training_state)

        return training_state.replace(policy_opt_state=policy_opt_state)


    @partial(jax.jit, static_argnames=("self",))
    def rollout(
        self,
        policy_params: Params,
        states: GeneralizedState,
        keys: RNGKey,
    ) -> Tuple[GeneralizedState, GeneralizedState, PPORollout]:
        """Roll out the policy and reservoir-sample one state per environment."""

        std_logits = policy_params["params"]["std_logits"]
        action_std = nn.sigmoid(std_logits)
        log_action_std = nn.log_sigmoid(std_logits)
        inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))

        def play_env_step(carry):
            state, sampled_state, episode_length, key = carry
            obs, z = self._env.get_obs(state)
            action_mean, _ = self._policy_network.apply(policy_params, obs, z)

            key, noise_key = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(
                noise_key, action_mean.shape
            )
            log_probability = self._log_probability(
                action,
                action_mean,
                log_action_std,
                inv_action_var,
            )
            next_state, transition_info = self._env.step(
                state, jnp.clip(action, -1.0, 1.0)
            )
            episode_length = episode_length + one \
                - next_state.env_state.done * episode_length
            key, sample_key = jax.random.split(key)
            replace_sample = jax.random.uniform(sample_key) < one / episode_length
            sampled_state = jax.tree.map(
                lambda new, old: jax.lax.select(replace_sample, new, old),
                next_state,
                sampled_state,
            )

            transition = PPORollout(
                obs=obs,
                zs=z,
                actions=action,
                old_log_probs=log_probability,
                rewards=transition_info.reward,
                dones=transition_info.done,
                truncations=transition_info.truncation,
            )
            return (next_state, sampled_state, episode_length, key), transition

        final_carry, rollout_data = jax.lax.scan(
            lambda x, _: jax.vmap(play_env_step)(x),
            (states, states, jnp.zeros((self.configs.vec_env,)), keys),
            length=self.configs.rollout_length,
        )
        final_states, sampled_states = final_carry[:2]
        return final_states, sampled_states, rollout_data


    def calculate_v(
        self,
        training_state: PPOTrainingState,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        moving_std = jnp.sqrt(training_state.moving_squared_diff)
        v_values = self._critic_network.apply(training_state.critic_params, obs, zs)
        return v_values * moving_std + training_state.moving_mean


    def calculate_td_lambda_returns(
        self,
        all_v_values: jax.Array,
        rollout_data: PPORollout,
    ) -> Tuple[jax.Array, jax.Array]:
        
        actual_discount = jnp.where(rollout_data.dones > 0.5, zero, self.configs.discount) # (rollout, vec_env, 1)
        ks = self.configs.gae_lambda * actual_discount # (rollout, vec_env, 1)
        bs = actual_discount * ((1 - self.configs.gae_lambda) * all_v_values[1:, ...]) # (rollout, vec_env, 1)
        init_target = all_v_values[-1] # (vec_env, 1)

        def scan_calculate_td_lambda(carry, data):
            last_target, last_bootstrap_portion = carry
            reward, truncation, k, b, v_value = data
            target = jnp.where(truncation > 0.5, v_value, reward + k * last_target + b)
            bootstrap_portion = jnp.where(truncation > 0.5, one, last_bootstrap_portion * k)
            new_carry = (target, bootstrap_portion)
            return new_carry, new_carry

        _, (critic_targets, bootstrap_portions) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (init_target, jnp.ones_like(init_target)),
            (rollout_data.rewards, rollout_data.truncations, ks, bs, all_v_values[:-1]),
            reverse=True,
        )
        critic_target_weights = one / (one + jnp.square(bootstrap_portions))

        return critic_targets, critic_target_weights


    def _process_gaes(
        self,
        gaes: jax.Array,
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        clipped_gaes = jnp.clip(gaes, gae_mean - 3*gae_std, gae_mean + 3*gae_std)
        clipped_mean = jnp.mean(clipped_gaes)
        processed_gaes = clipped_gaes - jnp.minimum(clipped_mean, zero)
        processed_gaes = processed_gaes / (1e-6 + jnp.sqrt(jnp.mean(processed_gaes**2)))
        return processed_gaes, (clipped_mean, jnp.std(clipped_gaes))


    def _make_shuffle_indices(self, key: RNGKey) -> jax.Array:
        return jax.random.permutation(
            key, self.samples_per_rollout
        ).reshape(self.mini_batch_num, self.configs.mini_batch_size)

    def _shuffle_data(
        self,
        data: Any,
        shuffled_indices: jax.Array,
    ) -> Any:
        def shuffle_leaf(value):
            flattened = value.reshape(self.samples_per_rollout, *value.shape[2:])
            return flattened[shuffled_indices]
        
        return jax.tree.map(shuffle_leaf, data)


    def _update_policy(
        self,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        training_data: TrainingData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        
        def loss_fn(params: Params, mini_batch: TrainingData):
            action_mean, std_logits = self._policy_network.apply(
                params, mini_batch.obs, mini_batch.zs
            )
            log_action_std = nn.log_sigmoid(std_logits)
            inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
            new_log_probability = self._log_probability(
                mini_batch.actions,
                action_mean,
                log_action_std,
                inv_action_var,
            )
            log_ratio = new_log_probability - mini_batch.old_log_probs
            ratio = jnp.exp(log_ratio)
            loss_condition = jax.lax.stop_gradient(
                log_ratio * jnp.sign(mini_batch.gaes)
                <= self._clip_log_ratio
            )
            approx_kl = jnp.mean((ratio - one) - log_ratio)
            entropy = jnp.sum(log_action_std, axis=-1, keepdims=True)
            loss = jnp.mean(
                jnp.where(
                    loss_condition,
                    -mini_batch.gaes * ratio,
                    zero,
                    ) - self.configs.entropy_gain * entropy
                )
            return loss, approx_kl

        def update_minibatch(carry, mini_batch):
            params, opt_state, running_approx_kl = carry
            gradients, approx_kl = jax.grad(loss_fn, has_aux=True)(params, mini_batch)
            running_approx_kl = (
                (one - self.ema_alpha) * approx_kl
                + self.ema_alpha * running_approx_kl
            )
            updates, opt_state = self._policy_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state, running_approx_kl), None

        def update_or_skip(carry, mini_batch):
            return jax.lax.cond(
                carry[-1] > self.configs.approx_kl_threshold,
                lambda _: (carry, None),
                lambda _: update_minibatch(carry, mini_batch),
                operand=None,
            )
        
        (
            policy_params,
            policy_opt_state,
            approx_kl,
        ), _ = jax.lax.scan(
            lambda x, _: jax.lax.scan(update_or_skip, x, training_data),
            (policy_params, policy_opt_state, zero),
            length=self.configs.policy_epochs,
        )
        return policy_params, policy_opt_state, approx_kl
    

    def _update_critic(
        self,
        critic_params: Params,
        critic_opt_state: optax.OptState,
        training_data: TrainingData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        def loss_fn(params: Params, mini_batch: TrainingData):
            predictions = self._critic_network.apply(
                params,
                mini_batch.obs,
                mini_batch.zs,
            )
            squared_errors = jnp.square(predictions - mini_batch.target_values)
            loss = jnp.average(squared_errors, weights=mini_batch.weights)
            return loss, jnp.sqrt(loss)

        def update_minibatch(carry, mini_batch):
            params, opt_state, running_rmse = carry
            gradients, rmse = jax.grad(loss_fn, has_aux=True)(params, mini_batch)
            running_rmse = (one - self.ema_alpha) * rmse + self.ema_alpha * running_rmse
            updates, opt_state = self._critic_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state, running_rmse), None

        (critic_params, critic_opt_state, critic_rmse), _ = jax.lax.scan(
            lambda x, _: jax.lax.scan(update_minibatch, x, training_data),
            (critic_params, critic_opt_state, zero),
            length=self.configs.critic_epochs,
        )
        return critic_params, critic_opt_state, critic_rmse
    

    @partial(jax.jit, static_argnames=("self",))
    def train(
        self,
        starting_states: GeneralizedState,
        training_state: PPOTrainingState,
        key: RNGKey,
    ) -> Tuple[
        Tuple[GeneralizedState, GeneralizedState, PPOTrainingState, RNGKey], PPOMetrics,
    ]:
        key, rollout_key, shuffle_key = jax.random.split(key, 3)
        rollout_keys = jax.random.split(
            rollout_key, self.configs.vec_env
        )
        policy_opt_state = self._update_learning_rate(training_state)
        final_states, sampled_states, rollout_data = self.rollout(
            training_state.policy_params,
            starting_states,
            rollout_keys,
        )
        final_obs, final_zs = self._env.get_obs(final_states)
        all_obs = jnp.concatenate([rollout_data.obs, final_obs[None, ...]], axis=0)
        all_zs = jnp.concatenate([rollout_data.zs, final_zs[None, ...]], axis=0)
        all_v_values = self.calculate_v(training_state, all_obs, all_zs)
        v_values = all_v_values[:-1]
        critic_targets, critic_target_weights = self.calculate_td_lambda_returns(
            all_v_values,
            rollout_data,
        )
        raw_gaes = critic_targets - v_values

        iteration_num = training_state.iteration_num + 1
        average_return = jnp.mean(critic_targets)
        statistics_learning_rate = one / iteration_num
        mse_learning_rate = jnp.maximum(statistics_learning_rate, 0.05)
        moving_mean = (
            (one - statistics_learning_rate) * training_state.moving_mean
            + statistics_learning_rate * average_return
        )
        moving_squared_diff = jnp.maximum(
            (one - statistics_learning_rate) * training_state.moving_squared_diff
            + statistics_learning_rate * jnp.mean(jnp.square(critic_targets - moving_mean)),
            one,
        )
        moving_std = jnp.sqrt(moving_squared_diff)
        moving_mse = (
            (one - mse_learning_rate) * training_state.moving_mse
            + mse_learning_rate * jnp.mean(jnp.square(raw_gaes))
        )

        gaes, (gae_mean, gae_std) = self._process_gaes(raw_gaes)
        clipped_critic_targets = jnp.clip(
            critic_targets,
            v_values - 3.0 * jnp.sqrt(moving_mse),
            v_values + 3.0 * jnp.sqrt(moving_mse),
        )
        normalized_critic_targets = (
            clipped_critic_targets - moving_mean
        ) / moving_std


        shuffled_indices = self._make_shuffle_indices(shuffle_key)
        training_data = self._shuffle_data(
            TrainingData(
                obs=rollout_data.obs,
                zs=rollout_data.zs,
                actions=rollout_data.actions,
                old_log_probs=rollout_data.old_log_probs,
                gaes=gaes,
                target_values=normalized_critic_targets,
                weights=critic_target_weights,
            ),
            shuffled_indices,
        )

        critic_params, critic_opt_state, critic_rmse, = self._update_critic(
            training_state.critic_params,
            training_state.critic_opt_state,
            training_data,
        )
        policy_params, policy_opt_state, policy_approx_kl = self._update_policy(
            training_state.policy_params,
            policy_opt_state,
            training_data,
        )
        lr_scale = jnp.where(
            policy_approx_kl > self.configs.approx_kl_threshold,
            0.9 * training_state.lr_scale,
            training_state.lr_scale,
        )
        lr_scale = jnp.where(
            policy_approx_kl < 0.8 * self.configs.approx_kl_threshold,
            0.9 * lr_scale + 0.1,
            lr_scale,
        )

        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            iteration_num=iteration_num,
            moving_mean=moving_mean,
            moving_squared_diff=moving_squared_diff,
            moving_mse=moving_mse,
            lr_scale=lr_scale,
        )
        metrics = PPOMetrics(
            critic_rmse=critic_rmse * moving_std,
            policy_approx_kl=policy_approx_kl,
            average_reward=jnp.mean(rollout_data.rewards),
            average_return=average_return,
            done_count=jnp.sum(rollout_data.dones),
            gae_mean=gae_mean,
            gae_std=gae_std,
        )
        return (final_states, sampled_states, training_state, key), metrics
