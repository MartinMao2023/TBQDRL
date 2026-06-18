from flax.struct import dataclass
from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from data_struct.distillation_transitions import GMMDistillationTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper


@dataclass
class GMMBCConfigs:
    learning_rate: float = 1e-3
    bc_epochs: int = 1
    mini_batch_size: int = 2048
    # Number of mini-batches per epoch; used to set the EMA decay for loss
    # tracking.  Set this to match (buffer_size // mini_batch_size) at the
    # call site so the EMA behaves like a per-epoch running average.
    num_mini_batches: int = 16


class GMMBCTrainingState(PyTreeNode):
    """Training state for the GMM-to-GMM distillation learner.

    Both teacher and student are GMMs; the loss is a divergence between them.
    The student std parameters are learned (not fixed), so no teacher_std is
    stored here.
    """

    policy_params: Params
    policy_opt_state: optax.OptState
    ema_loss: jax.Array
    step_num: int


class GMMBCTrainingMetrics(PyTreeNode):
    bc_loss: float


class GMMDistillationBC:
    """Behavior-cloning trainer that distills a GMM teacher into a GMM student.

    Unlike the standard ``BC`` class — which distills into a Gaussian student
    and supervises the mean — this class trains a student policy that is itself
    a Gaussian Mixture Model.  The distillation loss is a divergence between
    the teacher GMM and the student GMM (e.g. KL or Jeffrey divergence),
    implemented in ``gmm_distillation_loss_fn``.

    The student policy network is expected to have call signature::

        (obs, z) -> (component_weights, action_means, action_stds)

    where:
        component_weights  (k,)              — mixture weights (after softmax)
        action_means       (k, action_dim)   — per-component means
        action_stds        (k, action_dim)   — per-component standard deviations

    Args:
        env:            Task wrapper, used only during ``init`` to obtain
                        ``observation_size`` and ``z_size`` for parameter
                        initialisation.
        policy_network: Student GMM policy network.
        bc_configs:     Hyper-parameters for the BC training loop.
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        teacher_std_logits: jax.Array,
        bc_configs: GMMBCConfigs,
    ):
        self._env = env
        self.configs = bc_configs
        self._policy_network = policy_network
        self._teacher_std_logits = teacher_std_logits
        teacher_std = nn.sigmoid(teacher_std_logits) # d
        teacher_inv_var = jnp.square(jnp.exp(-teacher_std_logits) + 1) # d
        # teacher_std_entropy = -jnp.sum(nn.log_sigmoid(teacher_std_logits), axis=-1) # float

        self.ema_alpha = jnp.exp(
            jnp.array(-2.0 / bc_configs.num_mini_batches)
        )

        def make_optimizer(learning_rate):
            return optax.adam(learning_rate=learning_rate)

        self._policy_optimizer = optax.inject_hyperparams(make_optimizer)(
            learning_rate=bc_configs.learning_rate
        )

        def gmm_distillation_loss_fn(
            policy_params: Params,
            transitions: GMMDistillationTransition,
            key: RNGKey,
        ) -> Tuple[float, float]:
            """Divergence loss between the teacher GMM and the student GMM.

            Args:
                policy_params:  Student policy parameters.
                transitions:    Mini-batch of ``GMMDistillationTransition``,
                                carrying the teacher's component means and
                                weight_logits for each state.
                key:            Random key

            Returns:
                (loss, aux): scalar loss to differentiate through, and an
                auxiliary scalar for metric tracking (e.g. the same loss or
                a secondary diagnostic).
            """

            key1, key2, key3 = jax.random.split(key, num=3)

            action_means, weight_logits, _ = policy_network.apply(
                policy_params, transitions.obs, transitions.zs
            )
            current_means, current_weight_logits = jax.lax.stop_gradient((action_means, weight_logits))
            component_weights = nn.softmax(weight_logits) # B, k
            sampled_actions = jax.random.normal(
                key1, action_means.shape
                ) * teacher_std + action_means # B, k, d

            current_distances = jnp.square(sampled_actions[:, :, None, :] - current_means[:, None, :, :]) # B, k, k, d
            target_distances = jnp.square(sampled_actions[:, :, None, :] - transitions.action_means[:, None, :, :])

            log_q_components = current_weight_logits[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * current_distances, axis=-1)   # B, k, k

            log_p_components = transitions.component_logits[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * target_distances, axis=-1)    # B, k, k

            log_ratio = (
                nn.logsumexp(log_q_components, axis=-1) - nn.logsumexp(current_weight_logits, axis=-1, keepdims=True)
            ) - (
                nn.logsumexp(log_p_components, axis=-1) - nn.logsumexp(transitions.component_logits, axis=-1, keepdims=True)
            )  # B, k


            # Try to add density loss
            # p = nn.softmax(transitions.component_logits) # B, k
            # sampled_target_action_mean = jax.random.choice(key2, a=transitions.action_means, p=p, axis=1) # B, d
            # sampled_target_action = sampled_target_action_mean + jax.random.normal(
            #     key3, sampled_target_action_mean
            #     ) * teacher_std # B, d
            # sample_distances = jnp.square(sampled_target_action[:, None, :] - action_means) # B, k, d
            # target_log_likelihood = weight_logits - 0.5 * jnp.sum(teacher_inv_var * sample_distances, axis=-1) # B, k
            # target_log_likelihood = nn.logsumexp(target_log_likelihood, axis=-1) - nn.logsumexp(weight_logits, axis=-1) # B

            kl = jnp.mean(
                jnp.sum(log_ratio * component_weights, axis=-1)
            )
            # loss = kl - jnp.mean(target_log_likelihood)

            # return loss, kl
            return kl, kl
        

        self._gmm_distillation_loss_fn = gmm_distillation_loss_fn

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init(self, key: RNGKey) -> GMMBCTrainingState:
        """Initialise student policy parameters and optimiser state."""
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        policy_params = {
            **policy_params,
            "params": {
                **policy_params["params"],
                "std_logits": self._teacher_std_logits,
            },
        }

        return GMMBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=1.0, # For KL
            step_num=0,
        )

    # -----------------------------------------------------------------------
    # GMM-to-GMM distillation update  (GMMDistillationTransition)
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: GMMBCTrainingState,
        transitions: GMMDistillationTransition,
        key: RNGKey,
    ) -> Tuple[GMMBCTrainingState, GMMBCTrainingMetrics]:
        """Run ``bc_epochs`` epochs of GMM-to-GMM distillation gradient updates.

        ``transitions`` must already be batched into mini-batches, i.e. have
        leading shape ``(num_mini_batches, mini_batch_size, ...)``.
        """

        (policy_params, policy_opt_state, final_loss, key), _ = jax.lax.scan(
            lambda carry, _: self._train_gmm_epoch(carry, transitions),
            (
                training_state.policy_params,
                training_state.policy_opt_state,
                training_state.ema_loss,
                key,
            ),
            length=self.configs.bc_epochs,
        )

        corrected_loss = jnp.clip(final_loss, min=0.01, max=1)
        current_learning_rate = jnp.sqrt(corrected_loss) * 1e-3
        policy_opt_state = policy_opt_state._replace(
            hyperparams={**policy_opt_state.hyperparams, 'learning_rate': current_learning_rate}
        )

        new_training_state = GMMBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
        )
        return new_training_state, GMMBCTrainingMetrics(bc_loss=final_loss)

    @partial(jax.jit, static_argnames=("self",))
    def _train_gmm_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float, RNGKey],
        transitions: GMMDistillationTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:
        """One epoch of mini-batch gradient descent for GMM-to-GMM distillation."""

        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss, key = carry
            key, subkey = jax.random.split(key)

            grad, loss = jax.grad(self._gmm_distillation_loss_fn, has_aux=True)(
                policy_params, mini_batch, subkey
            )
            new_loss = loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss

            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)

            return (new_params, new_opt_state, new_loss, key), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None
