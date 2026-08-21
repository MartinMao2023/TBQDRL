"""Advantage-weighted regression using stored TD(lambda) returns.

Critic fitting and policy extraction are separate so the same fitted critic
and prepared dataset can be reused across Gaussian policy architectures.
"""

from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import optax
from flax.struct import PyTreeNode, dataclass
from jax import numpy as jnp

from custom_types import Params, RNGKey
from data_struct import PPOTransition
from networks import Multi_Action_PPO_Policy, PPO_Policy, Selector
from task_wrappers.base import BaseTaskWrapper


@dataclass
class AWRConfigs:
    critic_learning_rate: float = 5e-4
    critic_epochs: int = 4
    mini_batch_size: int = 4096


class AWRTrainingState(PyTreeNode):
    critic_params: Params
    critic_opt_state: optax.OptState
    moving_mean: jax.Array
    moving_std: jax.Array
    step_num: jax.Array


class AWRMetrics(PyTreeNode):
    critic_rmse: jax.Array
    gae_mean: jax.Array
    gae_std: jax.Array


@dataclass
class AWRPolicyConfigs:
    policy_learning_rate: float = 3e-4
    policy_epochs: int = 1
    stage2_epochs: int = 0
    mini_batch_size: int = 4096
    temperature: float = 1.0
    max_weight: float = 20.0
    loss_ema_alpha: float = 0.05


class AWRPolicyTrainingState(PyTreeNode):
    policy_params: Params
    policy_opt_state: optax.OptState
    step_num: jax.Array


class AWRPolicyMetrics(PyTreeNode):
    stage1_loss: jax.Array
    stage2_loss: jax.Array
    awr_weight_mean: jax.Array


@dataclass
class AWRGMMPolicyConfigs:
    policy_learning_rate: float = 3e-4
    selector_learning_rate: float = 3e-4
    policy_epochs: int = 1
    stage2_epochs: int = 0
    mini_batch_size: int = 4096
    temperature: float = 1.0
    max_weight: float = 20.0
    loss_ema_alpha: float = 0.05


class AWRGMMPolicyTrainingState(PyTreeNode):
    policy_params: Params
    selector_params: Params
    policy_opt_state: optax.OptState
    selector_opt_state: optax.OptState
    step_num: jax.Array


def _validate_transition_shapes(transitions: PPOTransition) -> int:
    if transitions.obs.ndim < 2:
        raise ValueError("transitions.obs must have at least one sample axis")

    sample_shape = transitions.obs.shape[:-1]
    for name in transitions.__dataclass_fields__:
        value = getattr(transitions, name)
        if value.ndim < 1 or value.shape[:-1] != sample_shape:
            raise ValueError(
                f"transitions.{name} has sample shape {value.shape[:-1]}; "
                f"expected {sample_shape}"
            )

    sample_num = 1
    for size in sample_shape:
        sample_num *= size
    if sample_num == 0:
        raise ValueError("the AWR dataset must not be empty")
    return sample_num


class AWR:
    """Fit a critic and attach final-critic advantages to its input data.

    ``train`` shuffles the complete ``PPOTransition`` exactly once, reshapes
    it to ``(mini_batch_num, mini_batch_size, ...)``, and uses that same order
    for every critic epoch. The returned transition remains in this layout.
    """

    def __init__(
        self,
        critic_network: nn.Module,
        configs: AWRConfigs,
    ):
        if configs.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive")
        if configs.critic_epochs < 1:
            raise ValueError("critic_epochs must be at least 1")
        if configs.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be at least 1")

        self._critic_network = critic_network
        self.configs = configs
        self._critic_optimizer = optax.adam(configs.critic_learning_rate)

    def init(
        self,
        critic_params: Params,
        moving_mean: jax.Array,
        moving_std: jax.Array,
    ) -> AWRTrainingState:
        """Warm-start from critic parameters and their output scaling."""
        return AWRTrainingState(
            critic_params=critic_params,
            critic_opt_state=self._critic_optimizer.init(critic_params),
            moving_mean=jnp.asarray(moving_mean, dtype=jnp.float32),
            moving_std=jnp.asarray(moving_std, dtype=jnp.float32),
            step_num=jnp.asarray(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnames=("self",))
    def _shuffle_data(
        self,
        transitions: PPOTransition,
        key: RNGKey,
    ) -> PPOTransition:
        sample_num = transitions.obs.size // transitions.obs.shape[-1]
        mini_batch_size = self.configs.mini_batch_size
        mini_batch_num = sample_num // mini_batch_size
        permutation = jax.random.permutation(key, sample_num)

        def shuffle_leaf(value):
            flattened = value.reshape(sample_num, value.shape[-1])
            return flattened[permutation].reshape(
                mini_batch_num,
                mini_batch_size,
                value.shape[-1],
            )

        return jax.tree.map(shuffle_leaf, transitions)

    def _critic_loss(
        self,
        critic_params: Params,
        mini_batch: PPOTransition,
    ) -> jax.Array:
        predictions = self._critic_network.apply(
            critic_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        squared_errors = jnp.square(
            predictions - mini_batch.gaes
        )
        return jnp.average(squared_errors, weights=mini_batch.weights)

    @partial(jax.jit, static_argnames=("self",))
    def _train_shuffled(
        self,
        training_state: AWRTrainingState,
        transitions: PPOTransition,
    ) -> Tuple[Tuple[AWRTrainingState, PPOTransition], AWRMetrics]:
        def update_minibatch(carry, mini_batch):
            critic_params, critic_opt_state = carry
            loss, gradients = jax.value_and_grad(self._critic_loss)(
                critic_params,
                mini_batch,
            )
            updates, critic_opt_state = self._critic_optimizer.update(
                gradients,
                critic_opt_state,
                critic_params,
            )
            critic_params = optax.apply_updates(critic_params, updates)
            return (critic_params, critic_opt_state), loss

        transitions = transitions.replace(
            gaes=(
                transitions.td_lambda_returns - training_state.moving_mean
                ) / (1e-8 + training_state.moving_std),
            )

        def train_epoch(carry, _):
            carry, _ = jax.lax.scan(
                update_minibatch,
                carry,
                transitions,
            )
            return carry, None

        (critic_params, critic_opt_state), _ = jax.lax.scan(
            train_epoch,
            (
                training_state.critic_params,
                training_state.critic_opt_state,
            ),
            length=self.configs.critic_epochs,
        )

        def calculate_advantages(_, mini_batch):
            values = self._critic_network.apply(
                critic_params,
                mini_batch.obs,
                mini_batch.zs,
            ) * training_state.moving_std + training_state.moving_mean
            gaes = mini_batch.td_lambda_returns - values
            return None, gaes

        _, gaes = jax.lax.scan(
            calculate_advantages,
            None,
            transitions,
        )
        transitions = transitions.replace(gaes=gaes)

        critic_rmse = jnp.sqrt(
            jnp.average(jnp.square(gaes), weights=transitions.weights)
        )
        mini_batch_num = transitions.obs.shape[0]
        update_num = self.configs.critic_epochs * mini_batch_num
        new_state = AWRTrainingState(
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            moving_mean=training_state.moving_mean,
            moving_std=training_state.moving_std,
            step_num=training_state.step_num + update_num,
        )
        metrics = AWRMetrics(
            critic_rmse=critic_rmse,
            gae_mean=jnp.mean(gaes),
            gae_std=jnp.std(gaes),
        )
        return (new_state, transitions), metrics

    def train(
        self,
        training_state: AWRTrainingState,
        transitions: PPOTransition,
        key: RNGKey,
    ) -> Tuple[Tuple[AWRTrainingState, PPOTransition], AWRMetrics]:
        """Shuffle once, fit the critic, and fill ``transitions.gaes``."""
        sample_num = _validate_transition_shapes(transitions)
        mini_batch_size = self.configs.mini_batch_size
        if sample_num % mini_batch_size != 0:
            raise ValueError(
                f"sample count {sample_num} must be divisible by "
                f"mini_batch_size {mini_batch_size}; AWR does not drop data"
            )

        transitions = self._shuffle_data(transitions, key)
        return self._train_shuffled(training_state, transitions)


class AWRGaussianPolicyExtractor:
    """Fit a ``PPO_Policy`` from shuffled, AWR-prepared transitions."""

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: PPO_Policy,
        configs: AWRPolicyConfigs,
    ):
        if configs.policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be positive")
        if configs.policy_epochs < 1:
            raise ValueError("policy_epochs must be at least 1")
        if configs.stage2_epochs < 0:
            raise ValueError("stage2_epochs must be non-negative")
        if configs.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be at least 1")
        if configs.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if configs.max_weight <= 0.0:
            raise ValueError("max_weight must be positive")

        self._env = env
        self._policy_network = policy_network
        self.configs = configs
        self._policy_optimizer = optax.adam(configs.policy_learning_rate)

    def init(
        self,
        key: RNGKey,
        policy_params: Optional[Params] = None,
    ) -> AWRPolicyTrainingState:
        """Initialize a Gaussian policy, or warm-start supplied parameters."""
        if policy_params is None:
            fake_obs = jnp.zeros((self._env.observation_size,))
            fake_zs = jnp.zeros((self._env.z_size,))
            policy_params = self._policy_network.init(
                key,
                obs=fake_obs,
                z=fake_zs,
            )

        return AWRPolicyTrainingState(
            policy_params=policy_params,
            policy_opt_state=self._policy_optimizer.init(policy_params),
            step_num=jnp.asarray(0, dtype=jnp.int32),
        )

    @staticmethod
    def _log_likelihood(
        actions: jax.Array,
        action_mean: jax.Array,
        log_action_std: jax.Array,
        inv_action_var: jax.Array,
    ) -> jax.Array:
        """Gaussian log likelihood, excluding the constant log(2*pi)."""
        return -jnp.sum(
            log_action_std
            + 0.5 * jnp.square(actions - action_mean) * inv_action_var,
            axis=-1,
            keepdims=True,
        )

    def _policy_loss(
        self,
        policy_params: Params,
        mini_batch: PPOTransition,
    ) -> jax.Array:
        action_mean, std_logits = self._policy_network.apply(
            policy_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        log_action_std = nn.log_sigmoid(std_logits)
        inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
        log_likelihood = self._log_likelihood(
            mini_batch.actions,
            action_mean,
            log_action_std,
            inv_action_var,
        )
        return -jnp.average(log_likelihood, weights=mini_batch.weights)

    def _stage2_policy_loss(
        self,
        policy_params: Params,
        mini_batch: PPOTransition,
        std_logits: jax.Array,
    ) -> jax.Array:
        action_mean, _ = self._policy_network.apply(
            policy_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        log_action_std = nn.log_sigmoid(std_logits)
        inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
        log_likelihood = self._log_likelihood(
            mini_batch.actions,
            action_mean,
            log_action_std,
            inv_action_var,
        )
        return -jnp.average(log_likelihood, weights=mini_batch.weights)

    @staticmethod
    def _replace_std_logits(
        policy_params: Params,
        std_logits: jax.Array,
    ) -> Params:
        return {
            **policy_params,
            "params": {
                **policy_params["params"],
                "std_logits": std_logits,
            },
        }

    @staticmethod
    def _annealed_std_logits(
        stage1_std: jax.Array,
        target_std: jax.Array,
        alpha: jax.Array,
    ) -> jax.Array:
        std = (1.0 - alpha) * stage1_std + alpha * target_std
        return jax.lax.stop_gradient(
            -jnp.log(1.0 / std - 1.0 + 1e-8)
        )


    @partial(jax.jit, static_argnames=("self",))
    def _train_stage1(
        self,
        training_state: AWRPolicyTrainingState,
        transitions: PPOTransition,
    ) -> Tuple[AWRPolicyTrainingState, AWRPolicyMetrics]:

        loss_alpha = self.configs.loss_ema_alpha

        def update_minibatch(carry, mini_batch):
            policy_params, policy_opt_state, last_loss = carry
            loss, gradients = jax.value_and_grad(self._policy_loss)(
                policy_params,
                mini_batch,
            )
            updates, policy_opt_state = self._policy_optimizer.update(
                gradients,
                policy_opt_state,
                policy_params,
            )
            loss = (1.0 - loss_alpha) * last_loss + loss_alpha * loss 
            policy_params = optax.apply_updates(policy_params, updates)
            return (policy_params, policy_opt_state, loss), None

        def train_epoch(carry, _):
            carry, _ = jax.lax.scan(
                update_minibatch,
                carry,
                transitions,
            )
            return carry, None

        (policy_params, policy_opt_state, final_loss), _ = jax.lax.scan(
            train_epoch,
            (
                training_state.policy_params,
                training_state.policy_opt_state,
                0.0,
            ),
            length=self.configs.policy_epochs,
        )

        mini_batch_num = transitions.obs.shape[0]
        update_num = self.configs.policy_epochs * mini_batch_num
        new_state = AWRPolicyTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            step_num=training_state.step_num + update_num,
        )
        metrics = AWRPolicyMetrics(
            stage1_loss=final_loss,
            stage2_loss=0.0,
            awr_weight_mean=jnp.mean(transitions.weights),
        )
        return new_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def _train_stage2(
        self,
        training_state: AWRPolicyTrainingState,
        transitions: PPOTransition,
        stage1_std: jax.Array,
        target_std_logits: jax.Array,
    ) -> Tuple[AWRPolicyTrainingState, jax.Array]:
        loss_alpha = self.configs.loss_ema_alpha
        mini_batch_num = transitions.obs.shape[0]
        total_updates = self.configs.stage2_epochs * mini_batch_num
        target_std = nn.sigmoid(target_std_logits)

        def update_minibatch(carry, mini_batch):
            policy_params, policy_opt_state, last_loss, update_num = carry
            alpha = jnp.where(
                total_updates > 1,
                update_num / (total_updates - 1),
                1.0,
            )
            std_logits = self._annealed_std_logits(
                stage1_std,
                target_std,
                alpha,
            )
            loss, gradients = jax.value_and_grad(self._stage2_policy_loss)(
                policy_params,
                mini_batch,
                std_logits,
            )
            updates, policy_opt_state = self._policy_optimizer.update(
                gradients,
                policy_opt_state,
                policy_params,
            )
            policy_params = optax.apply_updates(policy_params, updates)
            policy_params = self._replace_std_logits(
                policy_params,
                std_logits,
            )
            loss = (1.0 - loss_alpha) * last_loss + loss_alpha * loss
            return (
                policy_params,
                policy_opt_state,
                loss,
                update_num + 1,
            ), None

        def train_epoch(carry, _):
            carry, _ = jax.lax.scan(
                update_minibatch,
                carry,
                transitions,
            )
            return carry, None

        (policy_params, policy_opt_state, final_loss, _), _ = jax.lax.scan(
            train_epoch,
            (
                training_state.policy_params,
                training_state.policy_opt_state,
                0.0,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            length=self.configs.stage2_epochs,
        )
        policy_params = self._replace_std_logits(
            policy_params,
            target_std_logits,
        )
        new_state = AWRPolicyTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            step_num=training_state.step_num + total_updates,
        )
        return new_state, final_loss

    def train(
        self,
        training_state: AWRPolicyTrainingState,
        transitions: PPOTransition,
        target_std_logits: Optional[jax.Array] = None,
    ) -> Tuple[AWRPolicyTrainingState, AWRPolicyMetrics]:
        """Run learnable-std stage 1 and optional frozen-std stage 2."""
        if transitions.obs.ndim != 3:
            raise ValueError(
                "transitions must have shape "
                "(mini_batch_num, mini_batch_size, feature_dim)"
            )
        if transitions.obs.shape[1] != self.configs.mini_batch_size:
            raise ValueError(
                f"transition mini-batches contain {transitions.obs.shape[1]} "
                f"samples; expected {self.configs.mini_batch_size}"
            )

        current_std_logits = training_state.policy_params["params"]["std_logits"]
        if self.configs.stage2_epochs > 0:
            if target_std_logits is None:
                raise ValueError(
                    "target_std_logits is required when stage2_epochs is positive"
                )
            target_std_logits = jnp.asarray(target_std_logits)
            if target_std_logits.shape != current_std_logits.shape:
                raise ValueError(
                    f"target_std_logits has shape {target_std_logits.shape}; "
                    f"expected {current_std_logits.shape}"
                )

        _validate_transition_shapes(transitions)
        awr_weights = jnp.clip(
            jnp.exp(jnp.minimum(transitions.gaes / self.configs.temperature, 10.0)), # avoid Inf or NaNs
            max=self.configs.max_weight,
        )
        transitions = transitions.replace(weights=awr_weights)
        training_state, stage1_metrics = self._train_stage1(
            training_state,
            transitions,
        )

        stage2_loss = jnp.asarray(0.0)
        if self.configs.stage2_epochs > 0:
            current_std_logits = training_state.policy_params["params"]["std_logits"]
            stage1_std = nn.sigmoid(current_std_logits)
            training_state, stage2_loss = self._train_stage2(
                training_state,
                transitions,
                stage1_std,
                target_std_logits,
            )

        metrics = stage1_metrics.replace(
            stage2_loss=stage2_loss,
        )
        return training_state, metrics


class AWRGMMPolicyExtractor:
    """Fit a decoupled Gaussian-mixture policy from AWR transitions."""

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: Multi_Action_PPO_Policy,
        selector_network: Selector,
        configs: AWRGMMPolicyConfigs,
    ):
        if configs.policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be positive")
        if configs.selector_learning_rate <= 0.0:
            raise ValueError("selector_learning_rate must be positive")
        if configs.policy_epochs < 1:
            raise ValueError("policy_epochs must be at least 1")
        if configs.stage2_epochs < 0:
            raise ValueError("stage2_epochs must be non-negative")
        if configs.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be at least 1")
        if configs.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if configs.max_weight <= 0.0:
            raise ValueError("max_weight must be positive")
        if policy_network.component_num != selector_network.component_num:
            raise ValueError(
                "policy and selector component counts must match; got "
                f"{policy_network.component_num} and "
                f"{selector_network.component_num}"
            )

        self._env = env
        self._policy_network = policy_network
        self._selector_network = selector_network
        self.configs = configs
        self._policy_optimizer = optax.adam(configs.policy_learning_rate)
        self._selector_optimizer = optax.adam(configs.selector_learning_rate)

    def init(
        self,
        key: RNGKey,
        policy_params: Optional[Params] = None,
        selector_params: Optional[Params] = None,
    ) -> AWRGMMPolicyTrainingState:
        """Initialize missing GMM parameters and both optimizer states."""
        policy_key, selector_key = jax.random.split(key)
        fake_obs = jnp.zeros((self._env.observation_size,))
        fake_zs = jnp.zeros((self._env.z_size,))
        if policy_params is None:
            policy_params = self._policy_network.init(
                policy_key,
                obs=fake_obs,
                z=fake_zs,
            )
        if selector_params is None:
            selector_params = self._selector_network.init(
                selector_key,
                obs=fake_obs,
                z=fake_zs,
            )

        return AWRGMMPolicyTrainingState(
            policy_params=policy_params,
            selector_params=selector_params,
            policy_opt_state=self._policy_optimizer.init(policy_params),
            selector_opt_state=self._selector_optimizer.init(selector_params),
            step_num=jnp.asarray(0, dtype=jnp.int32),
        )

    @staticmethod
    def _log_likelihood(
        actions: jax.Array,
        action_means: jax.Array,
        weight_logits: jax.Array,
        log_action_std: jax.Array,
        inv_action_var: jax.Array,
    ) -> jax.Array:
        """GMM log likelihood with the constant log(2*pi) omitted."""
        component_log_likelihoods = weight_logits - 0.5 * jnp.sum(
            jnp.square(actions[..., None, :] - action_means) * inv_action_var,
            axis=-1,
        )
        return (
            nn.logsumexp(
                component_log_likelihoods,
                axis=-1,
                keepdims=True,
            )
            - nn.logsumexp(weight_logits, axis=-1, keepdims=True)
            - jnp.sum(log_action_std)
        )

    def _policy_loss(
        self,
        policy_params: Params,
        selector_params: Params,
        mini_batch: PPOTransition,
    ) -> jax.Array:
        action_means, std_logits = self._policy_network.apply(
            policy_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        weight_logits = self._selector_network.apply(
            selector_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        log_action_std = nn.log_sigmoid(std_logits)
        inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
        log_likelihood = self._log_likelihood(
            mini_batch.actions,
            action_means,
            weight_logits,
            log_action_std,
            inv_action_var,
        )
        return -jnp.average(log_likelihood, weights=mini_batch.weights)

    def _stage2_policy_loss(
        self,
        policy_params: Params,
        selector_params: Params,
        mini_batch: PPOTransition,
        std_logits: jax.Array,
    ) -> jax.Array:
        action_means, _ = self._policy_network.apply(
            policy_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        weight_logits = self._selector_network.apply(
            selector_params,
            mini_batch.obs,
            mini_batch.zs,
        )
        log_action_std = nn.log_sigmoid(std_logits)
        inv_action_var = jnp.square(1.0 + jnp.exp(-std_logits))
        log_likelihood = self._log_likelihood(
            mini_batch.actions,
            action_means,
            weight_logits,
            log_action_std,
            inv_action_var,
        )
        return -jnp.average(log_likelihood, weights=mini_batch.weights)

    @staticmethod
    def _replace_std_logits(
        policy_params: Params,
        std_logits: jax.Array,
    ) -> Params:
        return {
            **policy_params,
            "params": {
                **policy_params["params"],
                "std_logits": std_logits,
            },
        }

    @staticmethod
    def _annealed_std_logits(
        stage1_std: jax.Array,
        target_std: jax.Array,
        alpha: jax.Array,
    ) -> jax.Array:
        std = (1.0 - alpha) * stage1_std + alpha * target_std
        return jax.lax.stop_gradient(
            -jnp.log(1.0 / std - 1.0 + 1e-8)
        )

    @partial(jax.jit, static_argnames=("self",))
    def _train_stage1(
        self,
        training_state: AWRGMMPolicyTrainingState,
        transitions: PPOTransition,
    ) -> Tuple[AWRGMMPolicyTrainingState, AWRPolicyMetrics]:
        loss_alpha = self.configs.loss_ema_alpha

        def update_minibatch(carry, mini_batch):
            (
                policy_params,
                selector_params,
                policy_opt_state,
                selector_opt_state,
                last_loss,
            ) = carry
            loss, (policy_gradients, selector_gradients) = jax.value_and_grad(
                self._policy_loss,
                argnums=(0, 1),
                has_aux=False,
            )(
                policy_params,
                selector_params,
                mini_batch,
            )
            policy_updates, policy_opt_state = self._policy_optimizer.update(
                policy_gradients,
                policy_opt_state,
                policy_params,
            )
            selector_updates, selector_opt_state = (
                self._selector_optimizer.update(
                    selector_gradients,
                    selector_opt_state,
                    selector_params,
                )
            )
            policy_params = optax.apply_updates(policy_params, policy_updates)
            selector_params = optax.apply_updates(
                selector_params,
                selector_updates,
            )
            loss = (1.0 - loss_alpha) * last_loss + loss_alpha * loss
            return (
                policy_params,
                selector_params,
                policy_opt_state,
                selector_opt_state,
                loss,
            ), None

        def train_epoch(carry, _):
            carry, _ = jax.lax.scan(
                update_minibatch,
                carry,
                transitions,
            )
            return carry, None

        (
            policy_params,
            selector_params,
            policy_opt_state,
            selector_opt_state,
            final_loss,
        ), _ = jax.lax.scan(
            train_epoch,
            (
                training_state.policy_params,
                training_state.selector_params,
                training_state.policy_opt_state,
                training_state.selector_opt_state,
                0.0,
            ),
            length=self.configs.policy_epochs,
        )

        mini_batch_num = transitions.obs.shape[0]
        update_num = self.configs.policy_epochs * mini_batch_num
        new_state = AWRGMMPolicyTrainingState(
            policy_params=policy_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            selector_opt_state=selector_opt_state,
            step_num=training_state.step_num + update_num,
        )
        metrics = AWRPolicyMetrics(
            stage1_loss=final_loss,
            stage2_loss=0.0,
            awr_weight_mean=jnp.mean(transitions.weights),
        )
        return new_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def _train_stage2(
        self,
        training_state: AWRGMMPolicyTrainingState,
        transitions: PPOTransition,
        stage1_std: jax.Array,
        target_std_logits: jax.Array,
    ) -> Tuple[AWRGMMPolicyTrainingState, jax.Array]:
        loss_alpha = self.configs.loss_ema_alpha
        mini_batch_num = transitions.obs.shape[0]
        total_updates = self.configs.stage2_epochs * mini_batch_num
        target_std = nn.sigmoid(target_std_logits)

        def update_minibatch(carry, mini_batch):
            (
                policy_params,
                selector_params,
                policy_opt_state,
                selector_opt_state,
                last_loss,
                update_num,
            ) = carry
            alpha = jnp.where(
                total_updates > 1,
                update_num / (total_updates - 1),
                1.0,
            )
            std_logits = self._annealed_std_logits(
                stage1_std,
                target_std,
                alpha,
            )
            loss, (policy_gradients, selector_gradients) = jax.value_and_grad(
                self._stage2_policy_loss,
                argnums=(0, 1),
                has_aux=False,
            )(
                policy_params,
                selector_params,
                mini_batch,
                std_logits,
            )
            policy_updates, policy_opt_state = self._policy_optimizer.update(
                policy_gradients,
                policy_opt_state,
                policy_params,
            )
            selector_updates, selector_opt_state = (
                self._selector_optimizer.update(
                    selector_gradients,
                    selector_opt_state,
                    selector_params,
                )
            )
            policy_params = optax.apply_updates(policy_params, policy_updates)
            selector_params = optax.apply_updates(
                selector_params,
                selector_updates,
            )
            policy_params = self._replace_std_logits(
                policy_params,
                std_logits,
            )
            loss = (1.0 - loss_alpha) * last_loss + loss_alpha * loss
            return (
                policy_params,
                selector_params,
                policy_opt_state,
                selector_opt_state,
                loss,
                update_num + 1,
            ), None

        def train_epoch(carry, _):
            carry, _ = jax.lax.scan(
                update_minibatch,
                carry,
                transitions,
            )
            return carry, None

        (
            policy_params,
            selector_params,
            policy_opt_state,
            selector_opt_state,
            final_loss,
            _,
        ), _ = jax.lax.scan(
            train_epoch,
            (
                training_state.policy_params,
                training_state.selector_params,
                training_state.policy_opt_state,
                training_state.selector_opt_state,
                0.0,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            length=self.configs.stage2_epochs,
        )
        policy_params = self._replace_std_logits(
            policy_params,
            target_std_logits,
        )
        new_state = AWRGMMPolicyTrainingState(
            policy_params=policy_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            selector_opt_state=selector_opt_state,
            step_num=training_state.step_num + total_updates,
        )
        return new_state, final_loss

    def train(
        self,
        training_state: AWRGMMPolicyTrainingState,
        transitions: PPOTransition,
        target_std_logits: Optional[jax.Array] = None,
    ) -> Tuple[AWRGMMPolicyTrainingState, AWRPolicyMetrics]:
        """Run GMM extraction with optional frozen-std stage 2."""
        if transitions.obs.ndim != 3:
            raise ValueError(
                "transitions must have shape "
                "(mini_batch_num, mini_batch_size, feature_dim)"
            )
        if transitions.obs.shape[1] != self.configs.mini_batch_size:
            raise ValueError(
                f"transition mini-batches contain {transitions.obs.shape[1]} "
                f"samples; expected {self.configs.mini_batch_size}"
            )

        current_std_logits = training_state.policy_params["params"][
            "std_logits"
        ]
        if self.configs.stage2_epochs > 0:
            if target_std_logits is None:
                raise ValueError(
                    "target_std_logits is required when stage2_epochs is positive"
                )
            target_std_logits = jnp.asarray(target_std_logits)
            if target_std_logits.shape != current_std_logits.shape:
                raise ValueError(
                    f"target_std_logits has shape {target_std_logits.shape}; "
                    f"expected {current_std_logits.shape}"
                )

        _validate_transition_shapes(transitions)
        awr_weights = jnp.clip(
            jnp.exp(
                jnp.minimum(
                    transitions.gaes / self.configs.temperature,
                    10.0,
                )
            ),
            max=self.configs.max_weight,
        )
        transitions = transitions.replace(weights=awr_weights)
        training_state, stage1_metrics = self._train_stage1(
            training_state,
            transitions,
        )

        stage2_loss = jnp.asarray(0.0)
        if self.configs.stage2_epochs > 0:
            current_std_logits = training_state.policy_params["params"][
                "std_logits"
            ]
            stage1_std = nn.sigmoid(current_std_logits)
            training_state, stage2_loss = self._train_stage2(
                training_state,
                transitions,
                stage1_std,
                target_std_logits,
            )

        metrics = stage1_metrics.replace(stage2_loss=stage2_loss)
        return training_state, metrics
