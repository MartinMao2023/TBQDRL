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
    policy_learning_rates: Tuple[float, float] = (3e-4, 3e-4)
    critic_learning_rate: float = 5e-4
    selector_learning_rate: float = 1e-4
    policy_clip_ratio: float = 0.2
    selector_deviation_limit: float = 0.1
    entropy_gain: float = 0.01
    discount: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 32
    vec_env: int = 2048
    policy_epochs: int = 4
    critic_epochs: int = 4
    selector_epochs: int = 4
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
    reward_per_option: jax.Array
    policy_loss_per_option: jax.Array
    critic_loss_per_option: jax.Array
    policy_kl_per_option: jax.Array
    selector_loss: jax.Array
    selector_max_deviation: jax.Array
    selector_boundary_fraction: jax.Array
    persistence: jax.Array


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

        if any(lr <= 0 for lr in configs.policy_learning_rates):
            raise ValueError("policy_learning_rates must be positive")

        policy_learning_rates = jnp.asarray(
            configs.policy_learning_rates, dtype=jnp.float32
        )
        if policy_learning_rates.shape != (NUM_OPTIONS,):
            raise ValueError(
                f"policy_learning_rates must have shape ({NUM_OPTIONS},)"
            )

        self._env = env
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._selector_network = selector_network
        self._persistence_anneal_fn = persistence_anneal_fn
        self.configs = configs

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
            new_carry = jnp.where(truncation > 0.5, v, reward + k * last_target + b)
            bootstrap_portion = jnp.where(truncation > 0.5, one, last_bootstrap_portion * k)
            return new_carry, (new_carry, bootstrap_portion)

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


        iteration_num = jnp.asarray(0, dtype=jnp.int32)
        return BinaryOptionCriticPPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            selector_params=selector_params,
            policy_opt_state=self._policy_optimizer.init(policy_params),
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
    

    def _update_critics(
        self,
        training_state: BinaryOptionCriticPPOTrainingState,
        rollout: OptionRollout,
        targets: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        option_obs = jnp.swapaxes(rollout.obs, 0, 1)
        option_zs = jnp.swapaxes(rollout.zs, 0, 1)
        option_targets = jnp.swapaxes(targets, 0, 1)

        def loss_fn(critic_params):
            def option_loss(params, obs, zs, target):
                prediction = self._critic_network.apply(params, obs, zs)
                return jnp.mean(jnp.square(prediction - target))

            losses = jax.vmap(option_loss)(
                critic_params, option_obs, option_zs, option_targets
            )
            return jnp.sum(losses), losses

        def update_step(carry, _):
            params, opt_state = carry
            (_, losses), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, opt_state = self._critic_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), losses

        (critic_params, critic_opt_state), losses = jax.lax.scan(
            update_step,
            (training_state.critic_params, training_state.critic_opt_state),
            length=self.configs.critic_epochs,
        )
        return critic_params, critic_opt_state, losses[-1]

    def _update_policies(
        self,
        training_state: BinaryOptionCriticPPOTrainingState,
        rollout: OptionRollout,
        advantages: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array, jax.Array]:
        option_rollout = jax.tree.map(
            lambda value: jnp.swapaxes(value, 0, 1), rollout
        )
        option_advantages = jnp.swapaxes(advantages, 0, 1)

        def loss_fn(policy_params):
            def option_loss(params, data, advantage):
                action_mean, std_logits = self._policy_network.apply(
                    params, data.obs, data.zs
                )
                action_std = nn.sigmoid(std_logits)
                new_log_prob = -jnp.sum(
                    jnp.log(action_std + 1e-8)
                    + 0.5
                    * jnp.square(data.actions - action_mean)
                    / (jnp.square(action_std) + 1e-8),
                    axis=-1,
                    keepdims=True,
                )
                log_ratio = new_log_prob - data.old_log_probs
                ratio = jnp.exp(log_ratio)
                clipped_ratio = jnp.clip(
                    ratio,
                    1.0 - self.configs.policy_clip_ratio,
                    1.0 + self.configs.policy_clip_ratio,
                )
                surrogate = jnp.minimum(
                    ratio * advantage, clipped_ratio * advantage
                )
                entropy = jnp.sum(
                    jnp.log(action_std + 1e-8), axis=-1
                )
                loss = -jnp.mean(surrogate) - (
                    self.configs.entropy_gain * jnp.mean(entropy)
                )
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                return loss, approx_kl

            losses, approximate_kls = jax.vmap(
                option_loss, in_axes=(0, 0, 0)
            )(
                policy_params,
                option_rollout,
                option_advantages,
            )
            return jnp.sum(losses), (losses, approximate_kls)

        def update_step(carry, _):
            params, opt_state = carry
            (_, metrics), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, opt_state = self._policy_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), metrics

        (policy_params, policy_opt_state), (losses, approximate_kls) = (
            jax.lax.scan(
                update_step,
                (training_state.policy_params, training_state.policy_opt_state),
                length=self.configs.policy_epochs,
            )
        )
        return (
            policy_params,
            policy_opt_state,
            losses[-1],
            approximate_kls[-1],
        )

    def _update_selector( # <--- TO DO
        self,
        selector_params: Params,
        selector_opt_state: optax.OptState,
        rollout: OptionRollout,
        updated_option_values: jax.Array,
        anchor_probs: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array, jax.Array, jax.Array]:
        preferred_option_one = jax.lax.stop_gradient(
            (
                updated_option_values[..., 1]
                > updated_option_values[..., 0]
            ).astype(jnp.float32)
        )
        anchor = jax.lax.stop_gradient(anchor_probs[..., 1])
        lower = jnp.clip(
            anchor - self.configs.selector_deviation_limit, 1e-6, 1.0 - 1e-6
        )
        upper = jnp.clip(
            anchor + self.configs.selector_deviation_limit, 1e-6, 1.0 - 1e-6
        )

        def loss_fn(params):
            logits = self._selector_network.apply(
                params, rollout.obs, rollout.zs
            )
            option_one_logit = logits[..., 1] - logits[..., 0]
            probability = nn.sigmoid(option_one_logit)
            below = probability < lower
            above = probability > upper
            outside = below | above
            bounded_target = jnp.where(below, lower, upper)
            target = jnp.where(
                outside, bounded_target, preferred_option_one
            )
            loss = jnp.mean(
                optax.sigmoid_binary_cross_entropy(option_one_logit, target)
            )
            return loss, (
                jnp.max(jnp.abs(probability - anchor)),
                jnp.mean(outside),
            )

        def update_step(carry, _):
            params, opt_state = carry
            (loss, metrics), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, opt_state = self._selector_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, *metrics)

        (selector_params, selector_opt_state), selector_metrics = jax.lax.scan(
            update_step,
            (selector_params, selector_opt_state),
            length=self.configs.selector_epochs,
        )
        return (
            selector_params,
            selector_opt_state,
            selector_metrics[0][-1],
            selector_metrics[1][-1],
            selector_metrics[2][-1],
        )

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

        key, rollout_key, swap_key = jax.random.split(key, 3)
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
            selector_value, # float
            selector_advantage, # float
            selector_SNR, # float
            ) = jax.lax.cond(
            training_state.iteration_num > 0,
            self.calculate_selector_target,
            self.calculate_first_round_selector_target,
            all_option_values[:-1], 
            all_selector_probs[:-1],
        )

        # ==============================================
        #                TO DO
        # ==============================================
        # shuffle the data and update the policies and selectors



        # ==============================================
        #                TO DO
        # ==============================================
        # compute the critic target with the new selector
        # shuffle the data and update the critics
        (
            normalized_critic_targets, # (rollout, 2, vec_env, 1)
            critic_target_weights, # (rollout, 2, vec_env, 1)
            average_returns, # (2,)
            new_moving_mses, # (2,)
        ) = self.calculate_option_critic_targets(
            rollout_data,
            all_selector_probs, # <--- This needs to be computed with the updated selector
            all_option_values,
            training_state,
        )




        # ==============================================
        #                TO DO
        # ==============================================
        # Anneal the learning rates based on the policy stds
        # policy_opt_state = policy_opt_state._replace(
        #     hyperparams={
        #         **policy_opt_state.hyperparams,
        #         "learning_rate": new_lrs,  # shape (2,)
        #     }
        # )

        new_training_state = training_state.replace(
            policy_params=policy_params, # TO DO
            critic_params=critic_params, # TO DO
            selector_params=selector_params, # TO DO
            policy_opt_state=policy_opt_state, # TO DO
            critic_opt_state=critic_opt_state, # TO DO
            selector_opt_state=selector_opt_state, # TO DO
            iteration_num=training_state.iteration_num + 1,
            moving_mses=new_moving_mses,
        )
        final_states = self._swap_option_states(final_states, swap_key)


        # ==============================================
        #                TO DO
        # ==============================================
        # Things to monitor
        # 1) policy approx kl, shape of (2,) 
        # 2) option returns (from critic targets), shape of (2,) 
        # 3) critic rmse errors during training, shape of (2,) 
        # 4) average selector preference, float 
        # 5) selector_value, float
        # 6) selector advantage, float
        # 7) selector SNR, float

        metrics = BinaryOptionCriticPPOMetrics(
            reward_per_option=jnp.mean(rollout_data.rewards, axis=reward_axes),
            policy_loss_per_option=policy_losses,
            critic_loss_per_option=critic_losses,
            policy_kl_per_option=policy_kls,
            selector_loss=selector_loss,
            selector_max_deviation=selector_max_deviation,
            selector_boundary_fraction=selector_boundary_fraction,
            persistence=persistence,
        )
        return (final_states, new_training_state, key), metrics
