"""Scaffold for PPO with a trainable option and a frozen backup option.

Only the trainable policy is rolled out and optimized.  The frozen policy has
absorbing persistence, and its pretrained critic is retained as a fixed
terminal value for selector decisions and trainable-option targets.
"""

from functools import partial
from typing import Any, Callable, Tuple

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
class FrozenBackupOptionCriticPPOConfigs:
    policy_learning_rate_per_std: float = 1e-3
    critic_learning_rate: float = 5e-4
    selector_learning_rate: float = 3e-4
    policy_clip_ratio: float = 0.2
    approx_kl_threshold: float = 0.0125
    selector_deviation_limit: float = 0.05
    entropy_gain: float = 0.001
    discount: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 32
    vec_env: int = 2048
    mini_batch_size: int = 4096
    policy_epochs: int = 4
    critic_epochs: int = 4
    selector_epochs: int = 4
    moving_mse_ema_learning_rate: float = 0.05


class OptionRollout(PyTreeNode):
    """Trainable-option rollout with leading axes ``(time, environment)``."""

    obs: jax.Array
    zs: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    truncations: jax.Array


class PolicyTrainingData(PyTreeNode):
    """Policy data with arbitrary leading batch dimensions."""

    obs: jax.Array
    zs: jax.Array
    actions: jax.Array
    log_likelihood: jax.Array
    gaes: jax.Array


class WeightedRegressionData(PyTreeNode):
    """Shared data layout for selector and trainable-critic regression."""

    obs: jax.Array
    zs: jax.Array
    target_values: jax.Array
    weights: jax.Array


class FrozenBackupOptionCriticPPOTrainingState(PyTreeNode):
    policy_params: Params
    frozen_policy_params: Params
    critic_params: Params
    frozen_critic_params: Params
    selector_params: Params
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    selector_opt_state: optax.OptState
    critic_mean: jax.Array  # scalar, shared normalization
    critic_std: jax.Array  # scalar, shared normalization
    moving_mse: jax.Array  # scalar, trainable critic only
    iteration_num: jax.Array
    lr_scale: float


class FrozenBackupOptionCriticPPOMetrics(PyTreeNode):
    policy_approx_kl: jax.Array  # scalar
    policy_return: jax.Array  # scalar
    critic_rmse: jax.Array  # scalar
    selector_preference: jax.Array  # scalar, probability of trainable option
    selector_value: jax.Array  # scalar
    selector_advantage: jax.Array  # scalar
    selector_SNR: jax.Array  # scalar
    selector_certainty: jax.Array # scalar
    total_dones: jax.Array # scalar


class FrozenBackupOptionCriticPPO:
    """PPO with one trainable option and one absorbing frozen backup."""

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        selector_network: nn.Module,
        configs: FrozenBackupOptionCriticPPOConfigs,
        persistence_anneal_fn: Callable = lambda x: jnp.maximum(
            0.8, 0.95 - x * 0.001
        ),
    ):
        if configs.policy_learning_rate_per_std <= 0:
            raise ValueError("policy_learning_rate_per_std must be positive")
        if configs.critic_learning_rate <= 0:
            raise ValueError("critic_learning_rate must be positive")
        if configs.selector_learning_rate <= 0:
            raise ValueError("selector_learning_rate must be positive")
        if configs.policy_clip_ratio <= 0:
            raise ValueError("policy_clip_ratio must be positive")
        if configs.approx_kl_threshold <= 0:
            raise ValueError("approx_kl_threshold must be positive")
        if not 0 <= configs.selector_deviation_limit <= 1:
            raise ValueError("selector_deviation_limit must be in [0, 1]")
        if configs.selector_deviation_limit > 0.0625:
            print("selector_deviation_limit exceeds 0.0625; clipping it to 0.0625")
            configs = configs.replace(selector_deviation_limit=0.0625)

        if not 0 < configs.moving_mse_ema_learning_rate <= 1:
            raise ValueError(
                "moving_mse_ema_learning_rate must be in (0, 1]"
            )

        samples_per_option = configs.rollout_length * configs.vec_env
        if configs.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive")
        if samples_per_option % configs.mini_batch_size != 0:
            raise ValueError(
                "rollout_length * vec_env must be divisible by mini_batch_size"
            )

        self._env = env
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._selector_network = selector_network
        self._persistence_anneal_fn = persistence_anneal_fn
        self.configs = configs
        self.samples_per_option = samples_per_option
        self.mini_batch_num = samples_per_option // configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2.0 / self.mini_batch_num)


        self._clip_log_ratio = jnp.log(1.0 + configs.policy_clip_ratio)
        self._policy_optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=configs.policy_learning_rate_per_std
        )
        self._critic_optimizer = optax.adam(configs.critic_learning_rate)
        self._selector_optimizer = optax.adam(configs.selector_learning_rate)

    def init(
        self,
        key: RNGKey,
        policy_params: Params,
        frozen_policy_params: Params,
        frozen_critic_params: Params,
        critic_mean: jax.Array,  # scalar, shared normalization
        critic_std: jax.Array,  # scalar, shared normalization
        moving_mse: jax.Array,  # scalar, trainable critic only
    ) -> FrozenBackupOptionCriticPPOTrainingState:
        fake_obs = jnp.zeros((self._env.observation_size,))
        fake_z = jnp.zeros((self._env.z_size,))

        critic_mean = jnp.asarray(critic_mean)
        critic_std = jnp.asarray(critic_std)
        moving_mse = jnp.asarray(moving_mse)
        for name, value in (
            ("critic_mean", critic_mean),
            ("critic_std", critic_std),
            ("moving_mse", moving_mse),
        ):
            if value.shape != ():
                raise ValueError(f"{name} must be a scalar")

        key, selector_key = jax.random.split(key)

        selector_params = self._selector_network.init(
            selector_key, obs=fake_obs, z=fake_z
        )

        policy_opt_state = self._policy_optimizer.init(policy_params)
        std_logits = policy_params["params"]["std_logits"]  # (action_dim,)
        rms_std = jnp.sqrt(jnp.mean(nn.sigmoid(std_logits) ** 2))  # scalar
        adjusted_learning_rate = jnp.clip(
            rms_std * self.configs.policy_learning_rate_per_std,
            max=3e-4,
        )
        policy_opt_state = policy_opt_state._replace(
            hyperparams={
                **policy_opt_state.hyperparams,
                "learning_rate": adjusted_learning_rate,
            }
        )

        return FrozenBackupOptionCriticPPOTrainingState(
            policy_params=policy_params,
            frozen_policy_params=frozen_policy_params,
            critic_params=frozen_critic_params,
            frozen_critic_params=frozen_critic_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=self._critic_optimizer.init(frozen_critic_params),
            selector_opt_state=self._selector_optimizer.init(selector_params),
            critic_mean=critic_mean,
            critic_std=critic_std,
            moving_mse=moving_mse,
            iteration_num=jnp.asarray(0, dtype=jnp.int32),
            lr_scale=one,
        )


    def calculate_option_GAEs(
        self,
        rollout_data: OptionRollout, # (rollout_length, vec_env, ...)
        all_option_values: jax.Array, # (rollout_length + 1, vec_env, 2)
        all_selector_probs: jax.Array, # (rollout_length + 1, vec_env, 1)
        persistence: jax.Array, # scalar
    ) -> jax.Array:
        
        p = persistence + (1.0 - persistence) * all_selector_probs[1:, ...] # (rollout, vec_env, 1)
        actual_discount = jnp.where(rollout_data.dones > 0.5, zero, self.configs.discount) # (rollout, vec_env, 1)

        ks = self.configs.gae_lambda * actual_discount * p # (rollout, vec_env, 1)
        bs = actual_discount * (
            (1 - self.configs.gae_lambda) * p * all_option_values[1:, ..., :1] + (1 - p) * all_option_values[1:, ..., 1:]
        ) # (rollout, vec_env, 1)

        def scan_calculate_conditioned_gaes(carry, data) -> Tuple[jax.Array, jax.Array]:
            reward, truncation, k, b, v = data
            new_carry = jnp.where(truncation > 0.5, v, reward + k * carry + b)
            gae = new_carry - v
            return new_carry, gae

        _, gaes = jax.lax.scan(
            scan_calculate_conditioned_gaes,
            all_option_values[-1, ..., :1], # (vec_env, 1)
            (rollout_data.rewards, rollout_data.truncations, ks, bs, all_option_values[:-1, ..., :1]),
            reverse=True,
            )

        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        clipped_gaes = jnp.clip(gaes, gae_mean - 3*gae_std, gae_mean + 3*gae_std) # (rollout, vec_env, 1)
        gaes = clipped_gaes - jnp.minimum(jnp.mean(clipped_gaes), zero) # (rollout, vec_env, 1)
        
        return gaes / (1e-6 + jnp.sqrt(jnp.mean(gaes**2))) # (rollout, vec_env, 1)

        

    def calculate_option_critic_targets(
        self,
        rollout_data: OptionRollout, # (rollout, vec_env, ...)
        all_option_values: jax.Array, # (rollout + 1, vec_env, 2)
        all_selector_probs: jax.Array, # (rollout + 1, vec_env, 1)
        training_state: FrozenBackupOptionCriticPPOTrainingState,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        persistence = self._persistence_anneal_fn(training_state.iteration_num + 1)
        p = persistence + (1.0 - persistence) * all_selector_probs[1:, ...] # (rollout, vec_env, 1)
        actual_discount = jnp.where(rollout_data.dones > 0.5, zero, self.configs.discount) # (rollout, vec_env, 1)
        ks = self.configs.gae_lambda * actual_discount * p # (rollout, vec_env, 1)
        bs = actual_discount * (
            (1 - self.configs.gae_lambda) * p * all_option_values[1:, ..., :1] + (1 - p) * all_option_values[1:, ..., 1:]
        ) # (rollout, vec_env, 1)

        def scan_calculate_critic_targets(carry, data) -> Tuple[jax.Array, jax.Array]:
            last_target, last_bootstrap_portion = carry
            reward, truncation, k, b, v = data
            target = jnp.where(truncation > 0.5, v, reward + k * last_target + b)
            bootstrap_portion = jnp.where(truncation > 0.5, one, last_bootstrap_portion * k)
            new_carry = (target, bootstrap_portion)
            return new_carry, new_carry

        init_target = all_option_values[-1, ..., :1] # (vec_env, 1)
        current_values = all_option_values[:-1, ..., :1] # (rollout, vec_env, 1)
        _, (
            critic_targets, # (rollout, vec_env, 1)
            bootstrap_portions, # (rollout, vec_env, 1)
            ) = jax.lax.scan(
            scan_calculate_critic_targets,
            (init_target, jnp.ones_like(init_target)),
            (rollout_data.rewards, rollout_data.truncations, ks, bs, current_values),
            reverse=True,
            )
        critic_target_weights = 1 / (jnp.square(bootstrap_portions) + 1)
        average_return = jnp.mean(critic_targets) # scalar

        ema_learning_rate = self.configs.moving_mse_ema_learning_rate
        latest_mse = jnp.mean(jnp.square(critic_targets - current_values))
        moving_mse = (
            ema_learning_rate * latest_mse
            + (1.0 - ema_learning_rate) * training_state.moving_mse
        ) # scalar

        # Clip raw target deviations using the newly updated RMSE.
        rms_errors = jnp.sqrt(jnp.maximum(moving_mse, zero))
        clipped_targets = jnp.clip(
            critic_targets,
            current_values - 3.0 * rms_errors,
            current_values + 3.0 * rms_errors,
        )
        normalized_targets = (
            clipped_targets - training_state.critic_mean
        ) / (training_state.critic_std + 1e-8)

        return (
            normalized_targets, # (rollout, vec_env, 1)
            critic_target_weights, # (rollout, vec_env, 1)
            average_return, # scalar
            moving_mse, # scalar
            )


    def calculate_selector_target(
        self,
        option_values: jax.Array, # (rollout_length, vec_env, 2)
        anchor_probs: jax.Array, # (rollout_length, vec_env, 1)
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        value_diffs = jnp.expand_dims(option_values[..., 0] - option_values[..., 1], axis=-1) # (rollout_length, vec_env, 1)
        selector_targets = jnp.clip(
            jnp.where(
                value_diffs > 0,
                anchor_probs + self.configs.selector_deviation_limit,
                anchor_probs - self.configs.selector_deviation_limit,
                ) * 0.88888888 + 0.055555555,
            min=0.001,
            max=0.999,
            ) # (rollout_length, vec_env, 1)
        selector_target_weights = jnp.abs(value_diffs) # (rollout_length, vec_env, 1)
        selector_target_weights = selector_target_weights / (jnp.mean(selector_target_weights) + 1e-6) 

        # for monitoring
        selector_advantages = value_diffs * (anchor_probs * 2 - 1)
        selector_advantage = jnp.mean(selector_advantages)
        selector_value = jnp.mean(option_values) + jnp.mean(value_diffs * (anchor_probs - 0.5))
        selector_SNR = selector_advantage / (jnp.mean(jnp.abs(selector_advantages)) + 1e-6)

        return selector_targets, selector_target_weights, selector_value, selector_advantage, selector_SNR
    

    def calculate_first_round_selector_target(
        self,
        option_values: jax.Array,
        anchor_probs: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        selector_target_weights = jnp.ones_like(anchor_probs) # (rollout_length, 2, vec_env, 1)
        selector_targets = jnp.ones_like(anchor_probs) * 0.02 # (rollout_length, 2, vec_env, 1)
        selector_value = jnp.mean(option_values[..., 1])

        return selector_targets, selector_target_weights, selector_value, zero, zero
    

    @partial(jax.jit, static_argnames=("self",))
    def rollout(
        self,
        starting_states: GeneralizedState,
        policy_params: Params,
        keys: RNGKey,
    ) -> Tuple[GeneralizedState, OptionRollout]:
        """Roll out the trainable policy through vectorized environments."""

        action_std_logits = policy_params["params"]["std_logits"] # (action_dim,)
        action_std = nn.sigmoid(action_std_logits)  # (action_dim,)
        log_action_std = nn.log_sigmoid(action_std_logits) # (action_dim,)
        inv_action_var = jnp.square(1 + jnp.exp(-action_std_logits)) # (action_dim,)

        def play_env_step(state, key):
            obs, z = self._env.get_obs(state)
            action_mean, _ = self._policy_network.apply(policy_params, obs, z)
            key, noise_key = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(noise_key, action_mean.shape)
            log_prob = -jnp.sum(
                log_action_std + 0.5 * jnp.square(action - action_mean) * inv_action_var,
                axis=-1,
                keepdims=True,
            )
            next_state, transition_info = self._env.step(state, jnp.clip(action, -1.0, 1.0))
            transition = OptionRollout(
                obs=obs,
                zs=z,
                actions=action,
                old_log_probs=log_prob,
                rewards=transition_info.reward,
                dones=transition_info.done,
                truncations=transition_info.truncation,
            )
            return (next_state, key), transition

        def scan_step(carry, _):
            states, step_keys = carry
            (next_states, next_keys), transitions = jax.vmap(
                play_env_step
            )(states, step_keys)
            return (next_states, next_keys), transitions

        (final_states, _), transitions = jax.lax.scan(
            scan_step,
            (starting_states, keys),
            length=self.configs.rollout_length,
        )
        return final_states, transitions

    def _eval_critics(
        self,
        training_state: FrozenBackupOptionCriticPPOTrainingState,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        """Evaluate and denormalize the trainable and frozen critics.

        The final axis is ordered as ``(trainable, frozen)``.  For rollout
        observations the result is ``(rollout_length, vec_env, 2)``.
        """

        trainable_values = self._critic_network.apply(
            training_state.critic_params, obs, zs
        )  # (..., 1)
        frozen_values = self._critic_network.apply(
            training_state.frozen_critic_params, obs, zs
        )  # (..., 1)
        stacked_values = jnp.concatenate(
            [trainable_values, frozen_values], axis=-1
        ) * training_state.critic_std + training_state.critic_mean # (..., 2)
        return stacked_values

    def _selector_probs(
        self,
        selector_params: Params,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        """Return the probability of continuing with the trainable option."""

        logits = self._selector_network.apply(selector_params, obs, zs)
        return jnp.clip(nn.sigmoid(logits) * 1.125 - 0.0625, 0.0, 1.0)  # (..., 1)

    def _make_shuffle_indices(self, key: RNGKey) -> jax.Array:
        """Return a permutation shaped ``(mini_batch_num, mini_batch_size)``."""

        return jax.random.permutation(
            key, self.samples_per_option
        ).reshape(
            self.mini_batch_num,
            self.configs.mini_batch_size,
        )

    def _shuffle_data(
        self,
        data: Any,
        shuffled_indices: jax.Array,
    ) -> Any:
        """Shuffle ``(time, environment, ...)`` data into minibatches."""

        def shuffle_leaf(value):
            flattened = value.reshape(
                self.samples_per_option,
                *value.shape[2:],
            )
            return flattened[shuffled_indices]

        return jax.tree.map(shuffle_leaf, data)

    def _update_policy(
        self,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        policy_training_data: PolicyTrainingData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update the trainable policy, stopping when EMA KL is too large."""

        def loss_fn(params, training_data):
            action_mean, std_logits = self._policy_network.apply(
                params, training_data.obs, training_data.zs
            )
            log_action_std = nn.log_sigmoid(std_logits)
            inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
            entropy = jnp.sum(
                log_action_std, axis=-1, keepdims=True
            )
            new_log_likelihood = -jnp.sum(
                log_action_std + 0.5 * jnp.square(
                    action_mean - training_data.actions
                    ) * inv_action_var,
                axis=-1,
                keepdims=True,
            )
            log_ratio = new_log_likelihood - training_data.log_likelihood
            ratio = jnp.exp(log_ratio)
            loss_cond = jax.lax.stop_gradient(
                log_ratio * jnp.sign(training_data.gaes)
                <= self._clip_log_ratio
            )
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            loss = jnp.mean(
                jnp.where(
                    loss_cond,
                    -training_data.gaes * ratio,
                    0.0,
                ) - self.configs.entropy_gain * entropy
            )
            return loss, approx_kl

        def update_minibatch(carry, training_data):
            params, opt_state, running_approx_kl = carry
            gradients, approx_kl = jax.grad(
                loss_fn, has_aux=True
            )(params, training_data)
            running_approx_kl = (
                (1.0 - self.ema_alpha) * approx_kl
                + self.ema_alpha * running_approx_kl
            )
            updates, opt_state = self._policy_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state, running_approx_kl), None

        def update_or_skip(carry, training_data):
            return jax.lax.cond(
                carry[-1] > self.configs.approx_kl_threshold,
                lambda _: (carry, None),
                lambda _: update_minibatch(carry, training_data),
                operand=None,
            )

        def update_epoch(carry, _):
            return jax.lax.scan(
                update_or_skip,
                carry,
                policy_training_data,
            )

        (
            policy_params,
            policy_opt_state,
            approx_kl,
        ), _ = jax.lax.scan(
            update_epoch,
            (policy_params, policy_opt_state, zero),
            length=self.configs.policy_epochs,
        )
        return policy_params, policy_opt_state, approx_kl

    def _update_selector(
        self,
        selector_params: Params,
        selector_opt_state: optax.OptState,
        selector_training_data: WeightedRegressionData,
    ) -> Tuple[Params, optax.OptState]:
        """Update the selector with weighted BCE minibatches.

        ``target_values`` are already transformed into raw sigmoid space.
        """

        def loss_fn(params, training_data):
            logits = self._selector_network.apply(
                params, training_data.obs, training_data.zs
            )  # (mini_batch_size, 1)
            losses = optax.sigmoid_binary_cross_entropy(
                logits,
                training_data.target_values,
            )  # (mini_batch_size, 1)
            return jnp.mean(training_data.weights * losses)

        def update_minibatch(carry, training_data):
            params, opt_state = carry
            gradients = jax.grad(loss_fn)(params, training_data)
            updates, opt_state = self._selector_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        def update_epoch(carry, _):
            return jax.lax.scan(
                update_minibatch,
                carry,
                selector_training_data,
            )

        (selector_params, selector_opt_state), _ = jax.lax.scan(
            update_epoch,
            (selector_params, selector_opt_state),
            length=self.configs.selector_epochs,
        )
        return selector_params, selector_opt_state

    def _update_critic(
        self,
        critic_params: Params,
        critic_opt_state: optax.OptState,
        critic_training_data: WeightedRegressionData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update the trainable critic with weighted MSE minibatches."""

        def loss_fn(params, training_data):
            prediction = self._critic_network.apply(
                params, training_data.obs, training_data.zs
            )  # (mini_batch_size, 1)
            squared_error = jnp.square(
                prediction - training_data.target_values
            )  # (mini_batch_size, 1)
            loss = jnp.average(
                squared_error, weights=training_data.weights
            )
            return loss, jnp.sqrt(loss)

        def update_minibatch(carry, training_data):
            params, opt_state = carry
            gradients, rmse = jax.grad(
                loss_fn, has_aux=True
            )(params, training_data)
            updates, opt_state = self._critic_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), rmse

        def update_epoch(carry, _):
            return jax.lax.scan(
                update_minibatch,
                carry,
                critic_training_data,
            )

        (critic_params, critic_opt_state), critic_rmses = jax.lax.scan(
            update_epoch,
            (critic_params, critic_opt_state),
            length=self.configs.critic_epochs,
        )
        return (
            critic_params,
            critic_opt_state,
            jnp.mean(critic_rmses),
        )

    @partial(jax.jit, static_argnames=("self",))
    def train(
        self,
        starting_states: GeneralizedState,
        training_state: FrozenBackupOptionCriticPPOTrainingState,
        key: RNGKey,
    ) -> Tuple[
        Tuple[
            GeneralizedState,
            FrozenBackupOptionCriticPPOTrainingState,
            RNGKey,
        ],
        FrozenBackupOptionCriticPPOMetrics,
    ]:
        """Run one frozen-backup option-critic PPO iteration."""

        key, rollout_key, shuffle_key = jax.random.split(key, 3)
        rollout_keys = jax.random.split(rollout_key, self.configs.vec_env)

        std_logits = training_state.policy_params['params']['std_logits']
        rms_std = jnp.sqrt(jnp.mean(nn.sigmoid(std_logits) ** 2))
        lr_scale = training_state.lr_scale
        adjusted_learning_rate = jnp.minimum(
            rms_std * self.configs.policy_learning_rate_per_std,
            3e-4,
        ) * lr_scale
        policy_opt_state = training_state.policy_opt_state
        policy_opt_state = policy_opt_state._replace(
            hyperparams={
                **policy_opt_state.hyperparams,
                "learning_rate": adjusted_learning_rate,
            }
        )

        final_states, rollout_data = self.rollout(
            starting_states,
            training_state.policy_params,
            rollout_keys,
        )
        final_obs, final_zs = self._env.get_obs(final_states)
        all_obs = jnp.concatenate([rollout_data.obs, final_obs[None, ...]], axis=0)
        all_zs = jnp.concatenate([rollout_data.zs, final_zs[None, ...]], axis=0)

        all_option_values = self._eval_critics(
            training_state, all_obs, all_zs
        )  # (rollout_length + 1, vec_env, 2)
        all_selector_probs = self._selector_probs(
            training_state.selector_params, all_obs, all_zs
        )  # (rollout_length + 1, vec_env, 1)

        persistence = self._persistence_anneal_fn(
            training_state.iteration_num
        )
        gaes = self.calculate_option_GAEs(
            rollout_data,
            all_option_values,
            all_selector_probs,
            persistence,
        )  # (rollout_length, vec_env, 1)

        (
            selector_targets,
            selector_target_weights,
            selector_value,
            selector_advantage,
            selector_SNR,
        ) = jax.lax.cond(
            training_state.iteration_num > 0,
            self.calculate_selector_target,
            self.calculate_first_round_selector_target,
            all_option_values[:-1],
            all_selector_probs[:-1],
        )
        selector_preference = jnp.mean(all_selector_probs)  # scalar
        selector_certainty = jnp.mean(jnp.abs(all_selector_probs - 0.5) * 2)

        shuffled_indices = self._make_shuffle_indices(
            shuffle_key
        )  # (mini_batch_num, mini_batch_size)

        policy_training_data = PolicyTrainingData(
            obs=rollout_data.obs,
            zs=rollout_data.zs,
            actions=rollout_data.actions,
            log_likelihood=rollout_data.old_log_probs,
            gaes=gaes,
        )
        policy_training_data = self._shuffle_data(
            policy_training_data,
            shuffled_indices,
        )

        (
            shuffled_selector_targets,
            shuffled_selector_target_weights,
        ) = self._shuffle_data(
            (selector_targets, selector_target_weights),
            shuffled_indices,
        )
        selector_training_data = WeightedRegressionData(
            obs=policy_training_data.obs,
            zs=policy_training_data.zs,
            target_values=shuffled_selector_targets,
            weights=shuffled_selector_target_weights,
        )
        selector_params, selector_opt_state = self._update_selector(
            training_state.selector_params,
            training_state.selector_opt_state,
            selector_training_data,
        )

        (
            policy_params,
            policy_opt_state,
            policy_approx_kl,
        ) = self._update_policy(
            training_state.policy_params,
            policy_opt_state,
            policy_training_data,
        )

        # adaptively scale the lr
        lr_scale = jnp.where(
            policy_approx_kl > self.configs.approx_kl_threshold,
            0.9 * lr_scale,
            lr_scale,
            )
        lr_scale = jnp.where(
            policy_approx_kl < self.configs.approx_kl_threshold * 0.8,
            0.9 * lr_scale + 0.1,
            lr_scale,
            )

        all_selector_probs = self._selector_probs(
            selector_params, all_obs, all_zs
        )  # (rollout_length + 1, vec_env, 1)
        (
            normalized_critic_targets,
            critic_target_weights,
            policy_return,
            moving_mse,
        ) = self.calculate_option_critic_targets(
            rollout_data,
            all_option_values,
            all_selector_probs,
            training_state,
        )
        (
            shuffled_critic_targets,
            shuffled_critic_target_weights,
        ) = self._shuffle_data(
            (normalized_critic_targets, critic_target_weights),
            shuffled_indices,
        )
        critic_training_data = WeightedRegressionData(
            obs=policy_training_data.obs,
            zs=policy_training_data.zs,
            target_values=shuffled_critic_targets,
            weights=shuffled_critic_target_weights,
        )
        critic_params, critic_opt_state, critic_rmse = (
            self._update_critic(
                training_state.critic_params,
                training_state.critic_opt_state,
                critic_training_data,
            )
        )

        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            selector_opt_state=selector_opt_state,
            moving_mse=moving_mse,
            iteration_num=training_state.iteration_num + 1,
            lr_scale=lr_scale,
        )
        metrics = FrozenBackupOptionCriticPPOMetrics(
            policy_approx_kl=policy_approx_kl,
            policy_return=policy_return,
            critic_rmse=critic_rmse * training_state.critic_std,
            selector_preference=selector_preference,
            selector_value=selector_value,
            selector_advantage=selector_advantage,
            selector_SNR=selector_SNR,
            selector_certainty=selector_certainty,
            total_dones=jnp.sum(rollout_data.dones),
        )
        return (final_states, training_state, key), metrics
