"""Initial scaffold for the two-option option-critic PPO pipeline.

The two policies and critics have identical architectures but independent
parameters.  Their parameter trees carry a leading option axis of size two.
Rollouts consequently have shape ``(rollout_length, 2, vec_env, ...)``.

The option-conditioned target and GAE equations are intentionally left as a
placeholder method and must be implemented before using this algorithm for
experiments.
"""

from functools import partial
from typing import Callable, Tuple

import flax.linen as nn
import jax
import optax
from flax.struct import PyTreeNode, dataclass
from jax import numpy as jnp

from custom_types import Params, RNGKey
from data_struct.states import GeneralizedState
from task_wrappers.base import BaseTaskWrapper


NUM_OPTIONS = 2


@dataclass
class BinaryOptionCriticPPOConfigs:
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    selector_learning_rate: float = 1e-4
    policy_clip_ratio: float = 0.2
    selector_probability_limit: float = 0.1
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
        if not 0 <= configs.selector_probability_limit <= 1:
            raise ValueError("selector_probability_limit must be in [0, 1]")
        if not 0 <= configs.state_swap_fraction <= 1:
            raise ValueError("state_swap_fraction must be in [0, 1]")
        if not 0 < configs.moving_mse_ema_learning_rate <= 1:
            raise ValueError(
                "moving_mse_ema_learning_rate must be in (0, 1]"
            )

        self._env = env
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._selector_network = selector_network
        self._persistence_anneal_fn = persistence_anneal_fn
        self.configs = configs

        self._policy_optimizer = optax.adam(configs.policy_learning_rate)
        self._critic_optimizer = optax.adam(configs.critic_learning_rate)
        self._selector_optimizer = optax.adam(configs.selector_learning_rate)



    def calculate_option_targets_and_gaes(
        self,
        rollout: OptionRollout,
        all_option_values: jax.Array,
        all_selector_probs: jax.Array,
        training_state: BinaryOptionCriticPPOTrainingState,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Placeholder for the option-conditioned target and GAE equations.

        ``all_option_values`` has shape ``(T + 1, 2, E, 2)`` and
        ``all_selector_probs`` has shape ``(T + 1, 2, E, 2)``.  The last
        axis indexes the candidate critic/option.  Critic values are already
        denormalized to the raw return scale.

        TODO: continuation must combine persistence and selector probabilities.
        TODO: termination and time-limit truncation need separate handling.
        TODO: GAE must remain option-conditioned and on the raw return scale.
        """
        del all_selector_probs  # TODO: used by the real continuation target

        # Select each rollout policy's matching critic from the candidate axis.
        option_identity = jnp.eye(NUM_OPTIONS)[None, :, None, :]
        current_values = jnp.sum(
            all_option_values[:-1] * option_identity, axis=-1, keepdims=True
        )  # (T, 2, E, 1), raw return scale

        # DUMMY PLACEHOLDER: replace rewards with option-conditioned targets.
        raw_targets = rollout.rewards
        advantages = raw_targets - current_values  # TODO: replace with GAE

        # Update one MSE per option from the unclipped raw target errors.
        mse_axes = (0,) + tuple(range(2, advantages.ndim))
        latest_mses = jnp.mean(jnp.square(advantages), axis=mse_axes)
        ema_learning_rate = self.configs.moving_mse_ema_learning_rate
        moving_mses = (
            ema_learning_rate * latest_mses
            + (1.0 - ema_learning_rate) * training_state.moving_mses
        )

        # Clip raw target deviations using the newly updated per-option RMSE.
        stat_shape = (1, NUM_OPTIONS) + (1,) * (raw_targets.ndim - 2)
        rms_errors = jnp.sqrt(jnp.maximum(moving_mses, 0.0)).reshape(stat_shape)
        clipped_targets = jnp.clip(
            raw_targets,
            current_values - 3.0 * rms_errors,
            current_values + 3.0 * rms_errors,
        )

        # Normalize only critic targets; policy advantages remain in raw scale.
        critic_means = training_state.critic_means.reshape(stat_shape)
        critic_stds = training_state.critic_stds.reshape(stat_shape)
        normalized_targets = (
            clipped_targets - critic_means
        ) / (critic_stds + 1e-6)

        return normalized_targets, advantages, moving_mses


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
        return nn.sigmoid(logits)

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
            anchor - self.configs.selector_probability_limit, 1e-6, 1.0 - 1e-6
        )
        upper = jnp.clip(
            anchor + self.configs.selector_probability_limit, 1e-6, 1.0 - 1e-6
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

        final_states, rollout = self.rollout(
            starting_states,
            training_state.policy_params,
            rollout_keys,
        )
        final_obs, final_zs = self._env.get_obs(final_states)

        all_obs = jnp.concatenate((rollout.obs, final_obs[None, ...]), axis=0)
        all_zs = jnp.concatenate((rollout.zs, final_zs[None, ...]), axis=0)
        all_option_values = self._eval_critics(
            training_state, all_obs, all_zs
        )
        all_selector_probs = self._selector_probs(
            training_state.selector_params, all_obs, all_zs
        )
        anchor_probs = all_selector_probs[:-1]

        (
            normalized_targets,
            advantages,
            new_moving_mses,
        ) = self.calculate_option_targets_and_gaes(
            rollout,
            all_option_values,
            all_selector_probs,
            training_state,
        )

        critic_params, critic_opt_state, critic_losses = (
            self._update_critics(training_state, rollout, normalized_targets)
        )
        (
            policy_params,
            policy_opt_state,
            policy_losses,
            policy_kls,
        ) = self._update_policies(training_state, rollout, advantages)

        updated_critic_state = training_state.replace(
            critic_params=critic_params
        )
        updated_option_values = self._eval_critics(
            updated_critic_state, rollout.obs, rollout.zs
        )
        (
            selector_params,
            selector_opt_state,
            selector_loss,
            selector_max_deviation,
            selector_boundary_fraction,
        ) = self._update_selector(
            training_state.selector_params,
            training_state.selector_opt_state,
            rollout,
            updated_option_values,
            anchor_probs,
        )

        new_training_state = training_state.replace(
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

        reward_axes = (0,) + tuple(range(2, rollout.rewards.ndim))
        metrics = BinaryOptionCriticPPOMetrics(
            reward_per_option=jnp.mean(rollout.rewards, axis=reward_axes),
            policy_loss_per_option=policy_losses,
            critic_loss_per_option=critic_losses,
            policy_kl_per_option=policy_kls,
            selector_loss=selector_loss,
            selector_max_deviation=selector_max_deviation,
            selector_boundary_fraction=selector_boundary_fraction,
            persistence=self._persistence_anneal_fn(training_state.iteration_num),
        )
        return (final_states, new_training_state, key), metrics
