from flax.struct import dataclass
from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
from numpy import average
import optax
from jax import numpy as jnp
from optax.losses import sigmoid_binary_cross_entropy
from data_struct.distillation_transitions import GMMDistillationTransition
from data_struct import PPOTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper
from networks import GC_GMM_PPO_Policy



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
    # kl: float
    # nll: float
    # average_diff: float
    bc_loss: jax.Array


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
        policy_network: GC_GMM_PPO_Policy,
        teacher_std_logits: jax.Array,
        bc_configs: GMMBCConfigs,
        k: int = None,
    ):
        self._env = env
        self.configs = bc_configs
        self._policy_network = policy_network
        self._teacher_std_logits = teacher_std_logits
        teacher_std = nn.sigmoid(teacher_std_logits) # d
        teacher_inv_var = jnp.square(jnp.exp(-teacher_std_logits) + 1) # d

        component_num = policy_network.component_num
        if k is None:
            k = component_num // 2  # default: half of the components
        if not (1 <= k <= component_num - 1):
            raise Exception(
                f"k must be in [1, component_num - 1]; "
                f"got k={k}, component_num={component_num}"
            )

        self.k = k


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
            transitions: Tuple[GMMDistillationTransition, PPOTransition],
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
            gmm_transitions, demonstrate_transitions = transitions

            action_means1, weight_logits1, _ = policy_network.apply(
                policy_params, gmm_transitions.obs, gmm_transitions.zs
            ) # B, 2k, d;   B, 2k;
            action_means2, weight_logits2, _ = policy_network.apply(
                policy_params, demonstrate_transitions.obs, demonstrate_transitions.zs
            ) # B, 2k, d;   B, 2k;

            current_means, current_weight_logits = jax.lax.stop_gradient(
                (action_means1[:, :self.k, :], weight_logits1[:, :self.k])
                )
            component_weights = nn.softmax(weight_logits1[:, :self.k]) # B, k
            sampled_actions = jax.random.normal(
                key, current_means.shape
                ) * teacher_std + action_means1[:, :self.k, :] # B, k, d
            current_distances = jnp.square(sampled_actions[:, :, None, :] - current_means[:, None, :, :]) # B, k, k, d
            target_distances = jnp.square(sampled_actions[:, :, None, :] - gmm_transitions.action_means[:, None, :, :]) # B, k, 2k, d

            log_q_components = current_weight_logits[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * current_distances, axis=-1)   # B, k, k

            log_p_components = gmm_transitions.component_logits[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * target_distances, axis=-1)    # B, k, 2k

            log_ratio = (
                nn.logsumexp(log_q_components, axis=-1) - nn.logsumexp(current_weight_logits, axis=-1, keepdims=True)
            ) - (
                nn.logsumexp(log_p_components, axis=-1) - nn.logsumexp(gmm_transitions.component_logits, axis=-1, keepdims=True)
            )  # B, k

            kl = jnp.mean(
                jnp.sum(log_ratio * component_weights, axis=-1)
            )

            demonstrate_distances = jnp.square(action_means2[:, self.k:, :] - demonstrate_transitions.actions[:, None, :]) # B, k, d
            log_likelihoods = weight_logits2[:, self.k:] - 0.5 * jnp.sum(teacher_inv_var * demonstrate_distances, axis=-1) # B, k
            log_likelihoods = nn.logsumexp(log_likelihoods, axis=-1, keepdims=True) \
                - nn.logsumexp(weight_logits2[:, self.k:], axis=-1, keepdims=True) # B, 1
            # advantage-weighted NLL: weights carry the per-sample advantage
            # (filled with ones for now; relocate will provide real advantages)
            nll = -jnp.mean(demonstrate_transitions.weights * log_likelihoods)

            log_weight_ratio1 = nn.logsumexp(weight_logits1[:, :self.k], axis=-1) - nn.logsumexp(weight_logits1[:, self.k:], axis=-1) # B
            log_weight_ratio2 = nn.logsumexp(weight_logits2[:, :self.k], axis=-1) - nn.logsumexp(weight_logits2[:, self.k:], axis=-1) # B

            weight_regularize_loss = 0.5 * jnp.mean(
                sigmoid_binary_cross_entropy(log_weight_ratio1, 0.5) + sigmoid_binary_cross_entropy(log_weight_ratio2, 0.5)
            )
            # weight_regularize_loss = 0.5 * jnp.mean(
            #     sigmoid_binary_cross_entropy(log_weight_ratio1, 0.0) + sigmoid_binary_cross_entropy(log_weight_ratio2, 0.0)
            # )
            average_diff = 0.5 * (jnp.mean(
                jnp.abs(nn.sigmoid(log_weight_ratio1) - 0.5) + jnp.abs(nn.sigmoid(log_weight_ratio2) - 0.5)
                ))

            # return kl + nll + weight_regularize_loss, jnp.array([kl, nll, average_diff])
            # return nll + weight_regularize_loss, jnp.array([kl, nll, average_diff])
            return kl + weight_regularize_loss, jnp.array([kl, nll, average_diff])
        

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
            ema_loss=jnp.array([1.0, 5.0, 0.0]), # For KL, nll, average_diff
            step_num=0,
        )

    # -----------------------------------------------------------------------
    # GMM-to-GMM distillation update  (GMMDistillationTransition)
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: GMMBCTrainingState,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
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

        # corrected_loss = jnp.clip(final_loss, min=0.01, max=1)
        # current_learning_rate = jnp.sqrt(corrected_loss) * 1e-3
        # policy_opt_state = policy_opt_state._replace(
        #     hyperparams={**policy_opt_state.hyperparams, 'learning_rate': current_learning_rate}
        # )

        new_training_state = GMMBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
        )
        # kl, nll, average_diff = final_loss
        # return new_training_state, GMMBCTrainingMetrics(kl=kl, nll=nll, average_diff=average_diff)
        return new_training_state, GMMBCTrainingMetrics(bc_loss=final_loss)


    @partial(jax.jit, static_argnames=("self",))
    def _train_gmm_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float, RNGKey],
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:
        """One epoch of mini-batch gradient descent for GMM-to-GMM distillation."""

        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss, key = carry
            key, subkey = jax.random.split(key)

            grad, loss = jax.grad(self._gmm_distillation_loss_fn, has_aux=True)(
                policy_params, mini_batch, subkey
            )
            new_loss = loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss # (3,)

            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)

            return (new_params, new_opt_state, new_loss, key), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None
