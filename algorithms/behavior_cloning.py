from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from data_struct.distillation_transitions import BCTransition, GMMDistillationTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper


@dataclass
class BCConfigs:
    learning_rate: float = 5e-4
    bc_epochs: int = 1
    mini_batch_size: int = 2048
    # Number of mini-batches per epoch; used to set the EMA decay for loss
    # tracking.  Set this to match (buffer_size // mini_batch_size) at the
    # call site so the EMA behaves like a per-epoch running average.
    num_mini_batches: int = 16


class BCTrainingState(PyTreeNode):
    """Training state for the behavior-cloning / distillation learner.

    Intentionally minimal: no critic, no std schedule.  The policy std is
    supplied externally (from the teacher's parameter tree) and kept fixed
    throughout training.
    """

    policy_params: Params
    policy_opt_state: optax.OptState
    ema_loss: jax.Array
    step_num: int


class BCTrainingMetrics(PyTreeNode):
    bc_loss: float


class BC:
    """Behavior-cloning trainer that distills a GMM teacher into a Gaussian
    student policy.

    Two update paths are provided depending on which buffer type is used:

    * ``state_update``    — GMM distillation via ``GMMDistillationTransition``.
      The loss supervises the student mean toward the teacher's dominant
      (highest-weight) component mean, using the teacher's shared std.

    * ``state_update_bc`` — Standard action cloning via ``BCTransition``.
      The loss is the negative log-likelihood of the recorded action under
      the student Gaussian with the teacher's std.

    In both cases the std is held fixed (``teacher_std = sigmoid(std_logits)``
    extracted from the teacher's parameter tree), consistent with how
    ``GC_GMM_PPO_Policy`` shares a single global ``std_logits`` parameter
    across all states.

    Args:
        env:          Task wrapper, used only during ``init`` to obtain
                      ``observation_size`` and ``z_size`` for parameter
                      initialization.
        policy_network: Student policy network (e.g. ``GC_PPO_Policy``).
                        Expected call signature:
                        ``(obs, z) -> (action_mean, std_logits)``.
        teacher_std:  Fixed standard deviation array of shape ``(action_dim,)``,
                      computed as ``jax.nn.sigmoid(teacher_params['params']['std_logits'])``.
        bc_configs:   Hyper-parameters for the BC training loop.
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        teacher_std: jax.Array,
        bc_configs: BCConfigs,
    ):
        self._env = env
        self.configs = bc_configs
        self._policy_network = policy_network
        self._teacher_std = teacher_std

        self.ema_alpha = jnp.exp(
            jnp.array(-2.0 / bc_configs.num_mini_batches)
        )

        def make_ppo_optimizer(learning_rate):
            return optax.adam(learning_rate=learning_rate)
        
        self._policy_optimizer = optax.inject_hyperparams(make_ppo_optimizer)(
            learning_rate=bc_configs.learning_rate)
        # self._policy_optimizer = optax.adam(learning_rate=bc_configs.learning_rate)

        def gmm_distillation_loss_fn(
            policy_params: Params,
            transitions: GMMDistillationTransition,
        ) -> Tuple[float, float]:
            action_mean, _ = policy_network.apply(
                policy_params, transitions.obs, transitions.zs
            )

            # ---------------------------------------
            #  Used to use the largest peak, test if mean works better
            # ---------------------------------------
            # # Select the component with the largest weight per sample.
            # # One-hot gather avoids dynamic integer indexing.
            # best_idx = jnp.argmax(transitions.component_weights, axis=-1)  # (B,)
            # k = transitions.action_means.shape[-2]
            # one_hot = jax.nn.one_hot(best_idx, num_classes=k)              # (B, k)
            # # (B, action_dim)
            # target_mean = jnp.einsum("bk,bkd->bd", one_hot, transitions.action_means)

            # test if mean works better
            # target_mean = jnp.einsum("bk,bkd->bd", transitions.component_weights, transitions.action_means)
            target_mean = jnp.sum(
                transitions.component_weights[..., None] * transitions.action_means,
                axis=-2,
            )

            std = teacher_std  # (action_dim,)  — fixed, from teacher params
            nll = jnp.mean(
                jnp.sum(
                    jnp.log(std) + 0.5 * jnp.square((action_mean - target_mean) / std),
                    axis=-1,
                )
            )

            rmse = jnp.sqrt(jnp.mean(jnp.square(action_mean - target_mean)))


            return nll, rmse  # (loss, aux) — aux echoes loss for metric tracking

        self._gmm_distillation_loss_fn = gmm_distillation_loss_fn

        # ------------------------------------------------------------------
        # Simple behavior-cloning loss
        # ------------------------------------------------------------------
        # NLL of the recorded action under the student Gaussian.
        def bc_loss_fn(
            policy_params: Params,
            transitions: BCTransition,
        ) -> Tuple[float, float]:
            action_mean, _ = policy_network.apply(
                policy_params, transitions.obs, transitions.zs
            )

            std = teacher_std  # (action_dim,)
            nll = jnp.mean(
                jnp.sum(
                    jnp.log(std)
                    + 0.5 * jnp.square((transitions.actions - action_mean) / std),
                    axis=-1,
                )
            )
            return nll, nll

        self._bc_loss_fn = bc_loss_fn

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init(self, key: RNGKey) -> BCTrainingState:
        """Initialise student policy parameters and optimiser state."""
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        return BCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=0.1,
            step_num=0,
        )

    # -----------------------------------------------------------------------
    # GMM distillation update  (GMMDistillationTransition)
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: BCTrainingState,
        transitions: GMMDistillationTransition,
    ) -> Tuple[BCTrainingState, BCTrainingMetrics]:
        """Run ``bc_epochs`` epochs of GMM-distillation gradient updates.

        ``transitions`` must already be batched into mini-batches, i.e. have
        leading shape ``(num_mini_batches, mini_batch_size, ...)``, consistent
        with how PPO reshapes its buffer before calling ``state_update``.
        """

        (policy_params, policy_opt_state, final_loss), _ = jax.lax.scan(
            lambda carry, _: self._train_gmm_epoch(carry, transitions),
            (training_state.policy_params, training_state.policy_opt_state, training_state.ema_loss),
            length=self.configs.bc_epochs,
        )

        current_learning_rate = jnp.clip(final_loss * 0.01, 1e-4, 1e-3)
        policy_opt_state = policy_opt_state._replace(
            hyperparams={**policy_opt_state.hyperparams, 'learning_rate': current_learning_rate}
        )

        new_training_state = BCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
        )
        return new_training_state, BCTrainingMetrics(bc_loss=final_loss)


    @partial(jax.jit, static_argnames=("self",))
    def _train_gmm_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float],
        transitions: GMMDistillationTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:
        """One epoch of mini-batch gradient descent for GMM distillation."""

        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss = carry

            grad, loss = jax.grad(self._gmm_distillation_loss_fn, has_aux=True)(
                policy_params, mini_batch
            )
            new_loss = (
                loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss
            )

            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)

            return (new_params, new_opt_state, new_loss), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None

    # -----------------------------------------------------------------------
    # Simple behavior-cloning update  (BCTransition)
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("self",))
    def state_update_bc(
        self,
        training_state: BCTrainingState,
        transitions: BCTransition,
    ) -> Tuple[BCTrainingState, BCTrainingMetrics]:
        """Run ``bc_epochs`` epochs of standard action-cloning gradient updates.

        ``transitions`` must already be batched into mini-batches, i.e. have
        leading shape ``(num_mini_batches, mini_batch_size, ...)``.
        """

        (policy_params, policy_opt_state, final_loss), _ = jax.lax.scan(
            lambda carry, _: self._train_bc_epoch(carry, transitions),
            (training_state.policy_params, training_state.policy_opt_state, training_state.ema_loss),
            length=self.configs.bc_epochs,
        )

        new_training_state = BCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
        )
        return new_training_state, BCTrainingMetrics(bc_loss=final_loss)

    @partial(jax.jit, static_argnames=("self",))
    def _train_bc_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float],
        transitions: BCTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:
        """One epoch of mini-batch gradient descent for plain action cloning."""

        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss = carry

            grad, loss = jax.grad(self._bc_loss_fn, has_aux=True)(
                policy_params, mini_batch
            )
            new_loss = (
                loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss
            )

            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)

            return (new_params, new_opt_state, new_loss), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None

    # -----------------------------------------------------------------------
    # Staged GMM distillation update with per-bc-iteration loss output
    # -----------------------------------------------------------------------
