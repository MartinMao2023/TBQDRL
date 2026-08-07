"""Initial scaffold for the two-option option-critic PPO pipeline.

The two policies and critics have identical architectures but independent
parameters.  Their parameter trees carry a leading option axis of size two.
Rollouts consequently have shape ``(rollout_length, 2, vec_env, ...)``.

The option-conditioned target and GAE equations are intentionally left as a
placeholder method and must be implemented before using this algorithm for
experiments.
"""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import flax.linen as nn
import jax
import optax
from flax.struct import PyTreeNode, dataclass
from jax import numpy as jnp

from custom_types import Params, RNGKey
from data_struct.states import GeneralizedState
from task_wrappers.base import BaseTaskWrapper


NUM_OPTIONS = 2
zero = jnp.float32(0.0)
one = jnp.float32(1.0)


class _EmptyOptState(NamedTuple):
    """Placeholder Optax state for a stateless transform."""


def _scale_updates_by_option_learning_rate(
    learning_rate: jax.Array,
) -> optax.GradientTransformation:
    """Multiply Adam updates by a leading-option learning-rate vector."""

    def init_fn(_params):
        return _EmptyOptState()

    def update_fn(updates, state, params=None):
        del params
        learning_rates = jnp.asarray(learning_rate)

        def scale(update):
            shape = (NUM_OPTIONS,) + (1,) * (update.ndim - 1)
            return -learning_rates.reshape(shape) * update

        return jax.tree.map(scale, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def _make_policy_optimizer(learning_rate: jax.Array) -> optax.GradientTransformation:
    """Adam whose learning rate is a ``(2,)`` vector over stacked policies."""
    return optax.chain(
        optax.scale_by_adam(),
        _scale_updates_by_option_learning_rate(learning_rate),
    )


@dataclass
class BinaryOptionCriticPPOConfigs:
    policy_learning_rate_per_std: float = 1e-3
    critic_learning_rate: float = 5e-4
    selector_learning_rate: float = 3e-4
    policy_clip_ratio: float = 0.2
    selector_deviation_limit: float = 0.1
    entropy_gain: float = 0.001
    discount: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 32
    vec_env: int = 2048
    mini_batch_size: int = 1024
    policy_epochs: int = 4
    critic_epochs: int = 4
    selector_epochs: int = 2
    moving_mse_ema_learning_rate: float = 0.05
    state_swap_fraction: float = 0.25


class OptionRollout(PyTreeNode):
    """A rollout whose leading axes are ``(time, option, environment)``."""

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
    """Shared data layout for selector and critic regression."""

    obs: jax.Array
    zs: jax.Array
    target_values: jax.Array
    weights: jax.Array


class BinaryOptionCriticPPOTrainingState(PyTreeNode):
    policy_params: Params
    critic_params: Params
    selector_params: Params
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    selector_opt_state: optax.OptState
    critic_means: jax.Array
    critic_stds: jax.Array
    moving_mses: jax.Array
    iteration_num: jax.Array


class BinaryOptionCriticPPOMetrics(PyTreeNode):
    policy_approx_kls: jax.Array  # (2,)
    option_returns: jax.Array  # (2,)
    critic_rmses: jax.Array  # (2,)
    selector_preference: jax.Array  # scalar
    selector_value: jax.Array  # scalar
    selector_advantage: jax.Array  # scalar
    selector_SNR: jax.Array  # scalar


class BinaryOptionCriticPPO:
    """Two-policy PPO pipeline with option critics and a binary selector."""

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        selector_network: nn.Module,
        configs: BinaryOptionCriticPPOConfigs,
        persistence_anneal_fn: Callable = lambda x: jnp.maximum(0.8, 0.95 - x * 0.001),
    ):
        if configs.policy_clip_ratio <= 0:
            raise ValueError("policy_clip_ratio must be positive")
        if not 0 <= configs.selector_deviation_limit <= 1:
            raise ValueError("selector_deviation_limit must be in [0, 1]")
        if not 0 <= configs.state_swap_fraction <= 1:
            raise ValueError("state_swap_fraction must be in [0, 1]")
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

        policy_learning_rates = jnp.asarray(
            configs.policy_learning_rate_per_std, dtype=jnp.float32
        )

        self._env = env
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._selector_network = selector_network
        self._persistence_anneal_fn = persistence_anneal_fn
        self.configs = configs
        self.samples_per_option = samples_per_option
        self.mini_batch_num = samples_per_option // configs.mini_batch_size
        self._clip_log_ratio = jnp.log(1.0 + configs.policy_clip_ratio)

        # Learning rates remain injectable so each policy can be adapted later.
        self._policy_optimizer = optax.inject_hyperparams(
            _make_policy_optimizer
        )(learning_rate=policy_learning_rates)
        self._critic_optimizer = optax.adam(configs.critic_learning_rate)
        self._selector_optimizer = optax.adam(configs.selector_learning_rate)



    def calculate_option_GAEs(
        self, 
        rollout_data: OptionRollout, # (rollout_length, 2, vec_env, ...)
        all_option_values: jax.Array, # (rollout_length + 1, 2, vec_env, 2)
        all_selector_probs: jax.Array, # (rollout_length + 1, 2, vec_env, 1)
        persistence: float,
        ) -> jax.Array:

        corresponding_probs = jnp.concatenate([
          all_selector_probs[1:, :1, ...],
          1 - all_selector_probs[1:, 1:, ...],
        ], axis=1) # (rollout, 2, vec_env, 1)
        rearranged_option_values = jnp.concatenate([
            all_option_values[:, :1, ...],
            jnp.flip(all_option_values[:, 1:, ...], axis=-1),
        ], axis=1) # (rollout + 1, 2, vec_env, 2)

        p = persistence + (1 - persistence) * corresponding_probs # (rollout, 2, vec_env, 1)
        actual_discount = jnp.where(rollout_data.dones > 0.5, zero, self.configs.discount) # if done, discount = 0

        ks = self.configs.gae_lambda * actual_discount * p # (rollout, 2, vec_env, 1)
        bs = actual_discount * (
            (1 - self.configs.gae_lambda) * p * rearranged_option_values[1:, ..., :1] + (1 - p) * rearranged_option_values[1:, ..., 1:]
        ) # (rollout, 2, vec_env, 1)


        def scan_calculate_conditioned_gaes(carry, data) -> Tuple[jax.Array, jax.Array]:
            reward, truncation, k, b, v = data
            new_carry = jnp.where(truncation > 0.5, v, reward + k * carry + b)
            gae = new_carry - v
            return new_carry, gae

        _, gaes = jax.lax.scan(
            scan_calculate_conditioned_gaes,
            rearranged_option_values[-1, ..., :1],
            (rollout_data.rewards, rollout_data.truncations, ks, bs, rearranged_option_values[:-1, ..., :1]),
            reverse=True,
            )

        gae_mean = jnp.mean(gaes, axis=(0, 2, 3), keepdims=True) # (1, 2, 1, 1)
        gae_std = jnp.std(gaes, axis=(0, 2, 3), keepdims=True) # (1, 2, 1, 1)
        clipped_gaes = jnp.clip(gaes, gae_mean - 3*gae_std, gae_mean + 3*gae_std) # (rollout, 2, vec_env, 1)
        gaes = clipped_gaes - jnp.minimum(jnp.mean(clipped_gaes, axis=(0, 2, 3), keepdims=True), zero) # (rollout, 2, vec_env, 1)

        return gaes / (jnp.sqrt(jnp.mean(gaes**2, axis=(0, 2, 3), keepdims=True)) + 1e-6) # (rollout, 2, vec_env, 1)


    def calculate_option_critic_targets(
        self, 
        rollout_data: OptionRollout, # (rollout_length, 2, vec_env, ...)
        all_option_values: jax.Array, # (rollout_length + 1, 2, vec_env, 2)
        all_selector_probs: jax.Array, # (rollout_length + 1, 2, vec_env, 1)
        training_state: BinaryOptionCriticPPOTrainingState,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        corresponding_probs = jnp.concatenate([
            all_selector_probs[1:, :1, ...],
            1 - all_selector_probs[1:, 1:, ...],
        ], axis=1) # (rollout, 2, vec_env, 1)
        rearranged_option_values = jnp.concatenate([
            all_option_values[:, :1, ...],
            jnp.flip(all_option_values[:, 1:, ...], axis=-1),
        ], axis=1) # (rollout + 1, 2, vec_env, 2)

        persistence = self._persistence_anneal_fn(training_state.iteration_num + 1)
        p = persistence + (1 - persistence) * corresponding_probs # (rollout, 2, vec_env, 1)
        actual_discount = jnp.where(rollout_data.dones > 0.5, zero, self.configs.discount) # if done, discount = 0

        ks = self.configs.gae_lambda * actual_discount * p # (rollout, 2, vec_env, 1)
        bs = actual_discount * (
            (1 - self.configs.gae_lambda) * p * rearranged_option_values[1:, ..., :1] + (1 - p) * rearranged_option_values[1:, ..., 1:]
        ) # (rollout, 2, vec_env, 1)

        def scan_calculate_critic_targets(carry, data) -> Tuple[jax.Array, jax.Array]:
            last_target, last_bootstrap_portion = carry
            reward, truncation, k, b, v = data
            target = jnp.where(truncation > 0.5, v, reward + k * last_target + b)
            bootstrap_portion = jnp.where(truncation > 0.5, one, last_bootstrap_portion * k)
            new_carry = (target, bootstrap_portion)
            return new_carry, new_carry

        init_target = rearranged_option_values[-1, ..., :1] # (2, vec_env, 1)
        current_values = rearranged_option_values[:-1, ..., :1]
        _, (
            critic_targets, # (rollout, 2, vec_env, 1)
            bootstrap_portions, # (rollout, 2, vec_env, 1)
            ) = jax.lax.scan(
            scan_calculate_critic_targets,
            (init_target, jnp.ones_like(init_target)),
            (rollout_data.rewards, rollout_data.truncations, ks, bs, current_values),
            reverse=True,
            )
        critic_target_weights = 1 / (jnp.square(bootstrap_portions) + 1)
        average_returns = jnp.mean(critic_targets, axis=(0, 2, 3)) # (2,)

        ema_learning_rate = self.configs.moving_mse_ema_learning_rate
        latest_mses = jnp.mean(jnp.square(critic_targets - current_values), axis=(0, 2, 3)) # (2,)
        moving_mses = (
            ema_learning_rate * latest_mses
            + (1.0 - ema_learning_rate) * training_state.moving_mses
        )

        # Clip raw target deviations using the newly updated per-option RMSE.
        rms_errors = jnp.sqrt(jnp.maximum(moving_mses, zero))[None, :, None, None]
        clipped_targets = jnp.clip(
            critic_targets,
            current_values - 3.0 * rms_errors,
            current_values + 3.0 * rms_errors,
        )
        critic_means = training_state.critic_means[None, :, None, None]
        critic_stds = training_state.critic_stds[None, :, None, None]
        normalized_targets = (
            clipped_targets - critic_means
        ) / (critic_stds + 1e-6)

        return (
            normalized_targets, # (rollout, 2, vec_env, 1)
            critic_target_weights, # (rollout, 2, vec_env, 1)
            average_returns, # (2,)
            moving_mses, # (2,)
            )


    def calculate_selector_target(
        self, 
        option_values: jax.Array, # (rollout_length, 2, vec_env, 2)
        anchor_probs: jax.Array, # (rollout_length, 2, vec_env, 1)
        ) -> Tuple[jax.Array, jax.Array, float, float, float]:

        value_diffs = jnp.expand_dims(option_values[..., 0] - option_values[..., 1], axis=-1) # (rollout_length, 2, vec_env, 1)
        selector_targets = jnp.clip(
            jnp.where(
                value_diffs > 0,
                anchor_probs + self.configs.selector_deviation_limit,
                anchor_probs - self.configs.selector_deviation_limit,
                ),
            min=1e-6,
            max=1.0 - 1e-6,
            ) # (rollout_length, 2, vec_env, 1)
        selector_target_weights = jnp.abs(value_diffs)
        selector_target_weights = selector_target_weights / (jnp.mean(selector_target_weights) + 1e-6)

        # for monitoring
        selector_advantages = value_diffs * (anchor_probs - 0.5)
        selector_advantage = jnp.mean(selector_advantages)
        selector_value = jnp.mean(option_values) + selector_advantage
        selector_SNR = selector_advantage / (jnp.mean(jnp.abs(selector_advantages)) + 1e-6)

        return selector_targets, selector_target_weights, selector_value, selector_advantage, selector_SNR


    def calculate_first_round_selector_target(
        self, 
        option_values: jax.Array, # (rollout_length, 2, vec_env, 2)
        anchor_probs: jax.Array, # (rollout_length, 2, vec_env, 1)
        ) -> Tuple[jax.Array, jax.Array, float, float, float]:

        selector_target_weights = jnp.ones_like(anchor_probs) # (rollout_length, 2, vec_env, 1)
        selector_targets = jnp.ones_like(anchor_probs) * 0.5 # (rollout_length, 2, vec_env, 1)
        selector_value = jnp.mean(option_values)

        return selector_targets, selector_target_weights, selector_value, zero, zero


    def init(
        self,
        key: RNGKey,
        policy_params_0: Params,
        policy_params_1: Params,
        critic_params_0: Params,
        critic_params_1: Params,
        critic_means: jax.Array, # (2,)
        critic_stds: jax.Array, # (2,)
        moving_mses: jax.Array, # (2,)
    ) -> BinaryOptionCriticPPOTrainingState:
        fake_obs = jnp.zeros((self._env.observation_size,))
        fake_z = jnp.zeros((self._env.z_size,))

        policy_params = jax.tree.map(
            lambda param_0, param_1: jnp.stack((param_0, param_1), axis=0),
            policy_params_0,
            policy_params_1,
        )
        critic_params = jax.tree.map(
            lambda param_0, param_1: jnp.stack((param_0, param_1), axis=0),
            critic_params_0,
            critic_params_1,
        )

        critic_means = jnp.asarray(critic_means)
        critic_stds = jnp.asarray(critic_stds)
        moving_mses = jnp.asarray(moving_mses)
        for name, value in (
            ("critic_means", critic_means),
            ("critic_stds", critic_stds),
            ("moving_mses", moving_mses),
        ):
            if value.shape != (NUM_OPTIONS,):
                raise ValueError(f"{name} must have shape ({NUM_OPTIONS},)")

        selector_params = self._selector_network.init(
            key, obs=fake_obs, z=fake_z
        )

        policy_opt_state = self._policy_optimizer.init(policy_params)
        std_logits = policy_params["params"]["std_logits"] # (2, action_dim)
        rms_stds = jnp.sqrt(jnp.mean(nn.sigmoid(std_logits)**2, axis=-1)) # (2,)
        adjusted_learning_rates = jnp.clip(
            rms_stds * self.configs.policy_learning_rate_per_std,
            max=3e-4,
        ) # (2,)
        policy_opt_state = policy_opt_state._replace(
            hyperparams={
                **policy_opt_state.hyperparams,
                "learning_rate": adjusted_learning_rates,
            }
        )

        iteration_num = jnp.asarray(0, dtype=jnp.int32)
        return BinaryOptionCriticPPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=self._critic_optimizer.init(critic_params),
            selector_opt_state=self._selector_optimizer.init(selector_params),
            critic_means=critic_means,
            critic_stds=critic_stds,
            moving_mses=moving_mses,
            iteration_num=iteration_num,
        )

    @partial(jax.jit, static_argnames=("self",))
    def rollout(
        self,
        starting_states: GeneralizedState,
        policy_params: Params,
        keys: RNGKey,
    ) -> Tuple[GeneralizedState, OptionRollout]:
        """Roll out both policies through independent vectorized environments."""

        def play_env_step(policy_params, state, key, action_std):
            obs, z = self._env.get_obs(state)
            action_mean, _ = self._policy_network.apply(
                policy_params, obs, z
            )

            key, noise_key = jax.random.split(key)
            action = jnp.clip(
                action_mean
                + action_std * jax.random.normal(noise_key, action_mean.shape),
                -1.0,
                1.0,
            )
            log_prob = -jnp.sum(
                jnp.log(action_std + 1e-8)
                + 0.5
                * jnp.square(action - action_mean)
                / (jnp.square(action_std) + 1e-8),
                axis=-1,
                keepdims=True,
            )
            next_state, transition_info = self._env.step(state, action)
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

        def play_option_step(params, states, option_keys):
            action_std = nn.sigmoid(params["params"]["std_logits"])
            return jax.vmap(
                play_env_step,
                in_axes=(None, 0, 0, None),
            )(params, states, option_keys, action_std)

        def scan_step(carry, _):
            states, step_keys = carry
            (next_states, next_keys), transitions = jax.vmap(
                play_option_step,
                in_axes=(0, 0, 0),
            )(policy_params, states, step_keys)
            return (next_states, next_keys), transitions

        (final_states, _), transitions = jax.lax.scan(
            scan_step,
            (starting_states, keys),
            length=self.configs.rollout_length,
        )
        return final_states, transitions

    def _eval_critics(
        self,
        training_state: BinaryOptionCriticPPOTrainingState,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        """Evaluate and denormalize both critics at every state.

        The candidate-critic axis is moved next to the scalar value axis.  For
        rollout observations the result is ``(T, 2, E, 2)``.
        """

        values = jax.vmap(
            lambda params: self._critic_network.apply(params, obs, zs)
        )(training_state.critic_params)

        values = jnp.moveaxis(values, 0, -2).squeeze(-1) # (rollout, 2, vec_env, 2)

        return values * training_state.critic_stds + training_state.critic_means

    def _selector_probs(
        self,
        selector_params: Params,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        logits = self._selector_network.apply(selector_params, obs, zs)
        return nn.sigmoid(logits) # value corresponds to policy 0

    def _make_split_shuffle_indices(self, key: RNGKey) -> jax.Array:
        """Return two independent permutations shaped ``(2, M, B)``."""
        option_keys = jax.random.split(key, NUM_OPTIONS)
        return jax.vmap(
            lambda option_key: jax.random.permutation(
                option_key, self.samples_per_option
            ).reshape(
                self.mini_batch_num,
                self.configs.mini_batch_size,
            )
        )(option_keys)

    def _split_shuffle_option_data(
        self,
        data: PyTreeNode,
        shuffled_indices: jax.Array,
    ) -> PyTreeNode:
        """Shuffle each option independently into ``(M, 2, B, ...)``."""

        def shuffle_leaf(value):
            # (T, 2, E, ...) -> (2, T * E, ...)
            option_major = jnp.swapaxes(value, 1, 0) # (2, T, E, ...)
            flattened = option_major.reshape(
                NUM_OPTIONS,
                self.samples_per_option,
                -1,
            ) # (2, T * E, ...)

            # Each option uses its own permutation in shuffled_indices.
            shuffled = jax.vmap(
                lambda option_data, option_indices: option_data[option_indices]
            )(flattened, shuffled_indices)  # (2, M, B, ...)
            return jnp.swapaxes(shuffled, 0, 1)  # (M, 2, B, ...)

        return jax.tree.map(shuffle_leaf, data)
    

    def _update_critics(
        self,
        critic_params: Params,
        critic_opt_state: optax.OptState,
        critic_training_data: WeightedRegressionData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update both option critics with weighted MSE minibatches.

        ``critic_training_data`` has leading shape
        ``(mini_batch_num, 2, mini_batch_size, ...)``.
        """

        def loss_fn(params, training_data):
            # training_data fields: (2, mini_batch_size, feature_dim)

            def option_loss(option_params, obs, zs, target, weight):
                prediction = self._critic_network.apply(
                    option_params, obs, zs
                ) # (mini_batch_size, 1)
                squared_error = jnp.square(prediction - target) # (mini_batch_size, 1)
                return jnp.average(squared_error, weights=weight) # scalar

            losses = jax.vmap(option_loss)(
                params,
                training_data.obs,
                training_data.zs,
                training_data.target_values,
                training_data.weights,
            ) # (2,)
            return jnp.sum(losses), jnp.sqrt(losses) # scalar, (2,)

        def update_minibatch(carry, training_data):
            params, opt_state = carry
            gradients, losses = jax.grad(loss_fn, has_aux=True)(
                params, training_data
            ) # losses: (2,)
            updates, opt_state = self._critic_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), losses # (2,)

        def update_epoch(carry, _):
            return jax.lax.scan(
                update_minibatch,
                carry,
                critic_training_data,
            ) # losses: (mini_batch_num, 2)

        (critic_params, critic_opt_state), critic_losses = jax.lax.scan(
            update_epoch,
            (critic_params, critic_opt_state),
            length=self.configs.critic_epochs,
        ) # critic_losses: (critic_epochs, mini_batch_num, 2)
        return (
            critic_params,
            critic_opt_state,
            jnp.mean(critic_losses, axis=(0, 1)), # (2,)
        )

    def _update_policies(
        self,
        policy_params: Params,
        policy_opt_state: optax.OptState,
        policy_training_data: PolicyTrainingData,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update both policies with learnable-std PPO minibatches.

        ``policy_training_data`` has leading shape
        ``(mini_batch_num, 2, mini_batch_size, ...)``.
        """

        def loss_fn(params, training_data):
            # training_data fields: (2, mini_batch_size, feature_dim)

            def option_loss(option_params, obs, zs, actions, old_log_likelihood, gaes):
                action_mean, std_logits = self._policy_network.apply(
                    option_params, obs, zs
                )
                entropy = jnp.sum(
                    nn.log_sigmoid(std_logits), axis=-1, keepdims=True
                ) # (1,)
                new_log_likelihood = (
                    -0.5 * jnp.sum(
                        jnp.square(jnp.exp(-std_logits) + 1) * (jnp.square(action_mean - actions) + 1e-6),
                        axis=-1,
                        keepdims=True,
                    ) - entropy
                ) # (mini_batch_size, 1)
                log_ratio = new_log_likelihood - old_log_likelihood
                ratio = jnp.exp(log_ratio)
                loss_cond = jax.lax.stop_gradient(
                    log_ratio * jnp.sign(gaes)
                    <= self._clip_log_ratio
                )
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                loss = jnp.mean(
                    jnp.where(loss_cond, -gaes * ratio, 0.0)
                    - self.configs.entropy_gain * entropy
                )
                return loss, approx_kl

            losses, approx_kls = jax.vmap(option_loss)(
                params,
                training_data.obs,
                training_data.zs,
                training_data.actions,
                training_data.log_likelihood,
                training_data.gaes,
            ) # (2,), (2,)
            return jnp.sum(losses), approx_kls

        def update_minibatch(carry, training_data):
            params, opt_state = carry
            gradients, approx_kls = jax.grad(loss_fn, has_aux=True)(
                params, training_data
            ) # approx_kls: (2,)
            updates, opt_state = self._policy_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), approx_kls # (2,)

        def update_epoch(carry, _):
            return jax.lax.scan(
                update_minibatch,
                carry,
                policy_training_data,
            ) # approx_kls: (mini_batch_num, 2)

        (policy_params, policy_opt_state), approx_kls = jax.lax.scan(
            update_epoch,
            (policy_params, policy_opt_state),
            length=self.configs.policy_epochs,
        ) # approx_kls: (policy_epochs, mini_batch_num, 2)
        return (
            policy_params,
            policy_opt_state,
            jnp.mean(approx_kls, axis=(0, 1)), # (2,)
        )

    def _update_selector(
        self,
        selector_params: Params,
        selector_opt_state: optax.OptState,
        selector_training_data: WeightedRegressionData,
    ) -> Tuple[Params, optax.OptState]:
        """Update the scalar sigmoid selector with weighted BCE minibatches.

        ``selector_training_data`` has leading shape
        ``(mini_batch_num, 2, mini_batch_size, ...)``.
        """

        def loss_fn(params, training_data):
            logits = self._selector_network.apply(
                params, training_data.obs, training_data.zs
            ) # (2, mini_batch_size, 1)
            losses = optax.sigmoid_binary_cross_entropy(
                logits,
                training_data.target_values, # (2, mini_batch_size, 1)
            ) # (2, mini_batch_size, 1)
            weighted_losses = (
                training_data.weights * losses
            ) # (2, mini_batch_size, 1)
            return jnp.mean(weighted_losses) # scalar

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

    def _swap_option_states(
        self,
        states: GeneralizedState,
        key: RNGKey,
    ) -> GeneralizedState:
        swap_mask = (
            jax.random.uniform(key, (self.configs.vec_env,))
            < self.configs.state_swap_fraction
        )

        def swap_leaf(value):
            if value.ndim < 2:
                return value
            mask_shape = (1, self.configs.vec_env) + (1,) * (value.ndim - 2)
            return jnp.where(
                jnp.reshape(swap_mask, mask_shape),
                jnp.flip(value, axis=0),
                value,
            )

        return jax.tree.map(swap_leaf, states)

    @partial(jax.jit, static_argnames=("self",))
    def train(
        self,
        starting_states: GeneralizedState,
        training_state: BinaryOptionCriticPPOTrainingState,
        key: RNGKey,
    ) -> Tuple[
        Tuple[GeneralizedState, BinaryOptionCriticPPOTrainingState, RNGKey],
        BinaryOptionCriticPPOMetrics,
    ]:
        """Run one structurally complete, mathematically incomplete iteration."""

        key, rollout_key, shuffle_key, swap_key = jax.random.split(key, 4)
        rollout_keys = jax.random.split(
            rollout_key, NUM_OPTIONS * self.configs.vec_env
        ).reshape((NUM_OPTIONS, self.configs.vec_env, -1))

        final_states, rollout_data = self.rollout(
            starting_states,
            training_state.policy_params,
            rollout_keys,
        )
        final_obs, final_zs = self._env.get_obs(final_states)

        all_obs = jnp.concatenate((rollout_data.obs, final_obs[None, ...]), axis=0)
        all_zs = jnp.concatenate((rollout_data.zs, final_zs[None, ...]), axis=0)
        all_option_values = self._eval_critics(
            training_state, all_obs, all_zs
        ) # (rollout + 1, 2, vec_env, 2)
        all_selector_probs = self._selector_probs(
            training_state.selector_params, all_obs, all_zs
        ) # (rollout + 1, 2, vec_env, 1)

        persistence = self._persistence_anneal_fn(training_state.iteration_num)
        gaes = self.calculate_option_GAEs(rollout_data, all_option_values, all_selector_probs, persistence) # (rollout, 2, vec_env, 1)

        (
            selector_targets, # (rollout, 2, vec_env, 1)
            selector_target_weights, # (rollout, 2, vec_env, 1)
            selector_value, # scalar
            selector_advantage, # scalar
            selector_SNR, # scalar
            ) = jax.lax.cond(
            training_state.iteration_num > 0,
            self.calculate_selector_target,
            self.calculate_first_round_selector_target,
            all_option_values[:-1], 
            all_selector_probs[:-1],
        )
        average_selector_preference = jnp.mean(all_selector_probs) # scalar

        # Independently shuffle data generated by each rollout policy.
        shuffled_indices = self._make_split_shuffle_indices(shuffle_key) # (2, mini_batch_num, mini_batch_size)

        policy_training_data = PolicyTrainingData(
            obs=rollout_data.obs,
            zs=rollout_data.zs,
            actions=rollout_data.actions,
            log_likelihood=rollout_data.old_log_probs,
            gaes=gaes,
        )
        policy_training_data = self._split_shuffle_option_data(
            policy_training_data,
            shuffled_indices,
        )  # (mini_batch_num, 2, mini_batch_size, ...)

        (
            shuffled_selector_targets,
            shuffled_selector_target_weights,
            ) = self._split_shuffle_option_data(
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
            policy_kls,
        ) = self._update_policies(
            training_state.policy_params,
            training_state.policy_opt_state,
            policy_training_data,
        )

        # Critic targets use the updated selector, then reuse the same split indices.
        all_selector_probs = self._selector_probs(
            selector_params, all_obs, all_zs
        ) # (rollout + 1, 2, vec_env, 1)
        (
            normalized_critic_targets, # (rollout, 2, vec_env, 1)
            critic_target_weights, # (rollout, 2, vec_env, 1)
            average_returns, # (2,)
            new_moving_mses, # (2,)
        ) = self.calculate_option_critic_targets(
            rollout_data,
            all_option_values,
            all_selector_probs,
            training_state,
        )
        (
            shuffled_critic_targets,
            shuffled_critic_target_weights,
        ) = self._split_shuffle_option_data(
            (normalized_critic_targets, critic_target_weights),
            shuffled_indices,
        )
        critic_training_data = WeightedRegressionData(
            obs=policy_training_data.obs,
            zs=policy_training_data.zs,
            target_values=shuffled_critic_targets,
            weights=shuffled_critic_target_weights,
        )
        (
            critic_params,
            critic_opt_state,
            critic_losses,
        ) = self._update_critics(
            training_state.critic_params,
            training_state.critic_opt_state,
            critic_training_data,
        )

        # learning rate annealing
        std_logits = policy_params["params"]["std_logits"] # (2, action_dim)
        rms_stds = jnp.sqrt(jnp.mean(nn.sigmoid(std_logits)**2, axis=-1)) # (2,)
        adjusted_learning_rates = jnp.clip(
            rms_stds * self.configs.policy_learning_rate_per_std, 
            max=3e-4,
            ) # (2,)
        policy_opt_state = policy_opt_state._replace(
            hyperparams={
                **policy_opt_state.hyperparams,
                "learning_rate": adjusted_learning_rates,  # shape (2,)
            }
        )

        training_state = training_state.replace(
            policy_params=policy_params,
            critic_params=critic_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            selector_opt_state=selector_opt_state,
            iteration_num=training_state.iteration_num + 1,
            moving_mses=new_moving_mses,
        )
        final_states = self._swap_option_states(final_states, swap_key)

        metrics = BinaryOptionCriticPPOMetrics(
            policy_approx_kls=policy_kls, # (2,)
            option_returns=average_returns, # (2,)
            critic_rmses=critic_losses * training_state.critic_stds, # (2,)
            selector_preference=average_selector_preference, # scalar
            selector_value=selector_value, # scalar
            selector_advantage=selector_advantage, # scalar
            selector_SNR=selector_SNR, # scalar
        )
        return (final_states, training_state, key), metrics
