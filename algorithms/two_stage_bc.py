from flax.struct import dataclass
from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from data_struct import PPOTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper
from networks import GC_GMM_PPO_Policy


@dataclass
class TwoStageBCConfigs:
    learning_rate: float = 1e-3
    bc_epochs: int = 1
    mini_batch_size: int = 2048
    # Number of mini-batches per epoch; used to set the EMA decay for loss
    # tracking.  Set this to match (buffer_size // mini_batch_size) at the
    # call site so the EMA behaves like a per-epoch running average.
    num_mini_batches: int = 16
    # Stage 2 (importance-sampling PPO) upper clip: skip the update when
    # (new_log_likelihood - old_log_likelihood) exceeds this margin.
    clip_log_ratio: float = 0.2


class TwoStageBCTrainingState(PyTreeNode):
    """Training state for the two-stage distillation learner."""

    policy_params: Params
    policy_opt_state: optax.OptState
    ema_loss: jax.Array
    step_num: int
    # Stage 2 only: the (clipped) std at the end of stage 1, used as the
    # start point of the linear anneal toward PPO_std. Ignored in stage 1.
    stage1_std: jax.Array


class TwoStageBCTrainingMetrics(PyTreeNode):
    bc_loss: jax.Array


class TwoStageBC:
    """Two-stage demonstration distillation into a GMM student.

    Stage 1: advantage-weighted NLL. Both the component means and the
    shared std are learned; the effective std is floored at the teacher
    (PPO) std via ``max(PPO_std, learnable_std)`` so the NLL cannot NaN from
    std collapse on scarce data.

    Stage 2: importance-sampling PPO on the network from stage 1. The
    shared std is frozen (``stop_gradient``) and linearly interpolated
    from the stage-1 (clipped) std toward ``PPO_std`` by an ``alpha`` in
    ``[0, 1]`` that the caller passes to :meth:`stage2_update` (so the
    anneal schedule is fully controlled by the caller). The objective
    maximises ``weight * exp(new_log_likelihood - old_log_likelihood)``
    where ``old_log_likelihood`` is the teacher log-prob of the demo action
    (recorded in the buffer) and the upper clip stops the update once the new
    log-likelihood exceeds the old by ``clip_log_ratio``.

    The student policy network is expected to have call signature::

        (obs, z) -> (action_means, weight_logits, std_logits)

    where:
        action_means   (2k, action_dim)  — per-component means
        weight_logits   (2k,)            — per-component mixture logits
        std_logits      (action_dim,)     — shared std logits (sigmoid -> std)
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: GC_GMM_PPO_Policy,
        teacher_std_logits: jax.Array,
        bc_configs: TwoStageBCConfigs,
        k: int = None,
    ):
        self._env = env
        self.configs = bc_configs
        self._policy_network = policy_network
        self._teacher_std_logits = teacher_std_logits

        # PPO std (frozen teacher std) and its log, used as the floor in
        # stage 1 and as the anneal target in stage 2.
        self._ppo_std = nn.sigmoid(teacher_std_logits)               # (d,)
        self._ppo_std_logits = teacher_std_logits
        # self._ppo_log_std = nn.log_sigmoid(teacher_std_logits)        # (d,)

        component_num = policy_network.component_num
        if k is None:
            k = component_num // 2
        if not (1 <= k <= component_num):
            raise Exception(
                f"k must be in [1, component_num]; "
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

        max_ratio = jnp.exp(bc_configs.clip_log_ratio)

        # -----------------------------------------------------------------
        # Shared log-likelihood of an action under the GMM student.
        # Drop 2*pi; include the log-variance term (std is learnable in
        # stage 1, scheduled in stage 2). The std is shared across all
        # components, so -log(std**2) is pulled outside the logsumexp.
        #
        #   log p(a) = logsumexp_i [ w_i - 0.5 * sum_d(inv_var*(a-mean_i)^2 )
        #                         - sum_d log_std ]
        #            - logsumexp_i [ w_i ]
        #
        # where log_std = max(ppo_log_std, learnable_log_std) in stage 1
        # (effective std = max(PPO_std, sigmoid(std_logits))).
        # -----------------------------------------------------------------
        def gmm_log_likelihood(action_means, weight_logits, std_logits, actions):
            """log p(a) under the GMM; `log_std` is per-dim shared (already
            pulled-outside-ready). Returns (..., 1)."""
            inv_var = jnp.square(1 + jnp.exp(-std_logits))                  # (..., d)
            # (..., 2k) per-component log N (without the shared -sum log_std)
            per_comp = weight_logits - 0.5 * jnp.sum(
                inv_var * jnp.square(action_means - actions[..., None, :]),
                axis=-1,
            ) # B, k
            log_norm = jnp.sum(nn.log_sigmoid(std_logits), axis=-1, keepdims=True)   # (..., 1)
            return nn.logsumexp(per_comp, axis=-1, keepdims=True) \
                - nn.logsumexp(weight_logits, axis=-1, keepdims=True) \
                - log_norm

        # -----------------------------------------------------------------
        # Stage 1 loss: advantage-weighted NLL over ALL 2k components.
        # Means and the shared std are learnable; the effective std is
        # max(PPO_std, sigmoid(std_logits)).
        # -----------------------------------------------------------------
        def stage1_loss_fn(
            policy_params: Params,
            transitions: PPOTransition,
        ) -> Tuple[float, jax.Array]:
            action_means, weight_logits, std_logits = policy_network.apply(
                policy_params, transitions.obs, transitions.zs,
            )  # B, 2k, d ; B, 2k ; d
            std_logits = jnp.maximum(
                self._ppo_std_logits, std_logits
            )
            log_likelihoods = gmm_log_likelihood(
                action_means, weight_logits, std_logits, transitions.actions,
            )  # B, 1
            nll = -jnp.mean(transitions.weights * log_likelihoods)
            return nll, jnp.array([nll, 0.0, 0.0])

        # -----------------------------------------------------------------
        # Stage 2 loss: importance-sampling PPO with an upper clip.
        # The scheduled (frozen) per-dim `log_std` is passed in as an
        # argument so it can change per outer iteration without recompiling.
        # Only the means and the mixture weight_logits are trained here.
        # -----------------------------------------------------------------
        def stage2_loss_fn(
            policy_params: Params,
            transitions: PPOTransition,
            std_logits: jax.Array,
        ) -> Tuple[float, jax.Array]:
            action_means, weight_logits, _ = policy_network.apply(
                policy_params, transitions.obs, transitions.zs,
            )  # B, 2k, d ; B, 2k
            new_log_likelihood = gmm_log_likelihood(
                action_means, weight_logits, std_logits, transitions.actions,
            )  # B, 1
            log_ratio = new_log_likelihood - transitions.log_likelihood  # B, 1
            ratio = jnp.exp(log_ratio)
            # upper-only clip: stop updating when new exceeds old by the margin
            ratio = jnp.minimum(max_ratio, ratio)
            surrogate = transitions.weights * ratio
            advantage = jnp.mean(surrogate)
            return -advantage, jnp.array([0.0, advantage, jnp.mean(ratio)])

        self._stage1_loss_fn = stage1_loss_fn
        self._stage2_loss_fn = stage2_loss_fn

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
    def init(self, key: RNGKey) -> TwoStageBCTrainingState:
        """Initialise student policy parameters and optimiser state."""
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        # warm-start the shared std at the teacher (PPO) std
        policy_params = {
            **policy_params,
            "params": {
                **policy_params["params"],
                "std_logits": self._teacher_std_logits,
            },
        }

        return TwoStageBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=jnp.array([1.0, 5.0, 0.0]),
            step_num=0,
            stage1_std=self._ppo_std,
        )

    # -----------------------------------------------------------------------
    # Stage switching
    # -----------------------------------------------------------------------
    def annealed_std_logits(self, stage1_std: jax.Array, alpha: jax.Array) -> jax.Array:
        """Per-dim log std for stage 2, frozen (stop_gradient).

        Linearly interpolates the stage-1 effective std
        (``max(PPO_std, learnable_std)``) toward ``PPO_std`` by ``alpha`` in
        ``[0, 1]`` (0 -> stage-1 std, 1 -> PPO_std) and returns its log.
        ``alpha`` is a traced scalar so this stays jit-able without
        recompilation, and the caller controls the anneal schedule.
        """
        std = (1.0 - alpha) * stage1_std + alpha * self._ppo_std
        return jax.lax.stop_gradient(
            -jnp.log(1 / std - 1 + 1e-8)
            )

    def stage1_final_std(self, training_state: TwoStageBCTrainingState) -> jax.Array:
        """Effective std at the end of stage 1: ``max(PPO_std, sigmoid(std_logits))``.

        The caller should ``training_state.replace(stage1_std=...)`` with this
        value before switching to stage 2, so the anneal starts from the
        actual stage-1 std.
        """
        std_logits = training_state.policy_params["params"]["std_logits"]
        learnable_std = nn.sigmoid(std_logits)
        return jnp.maximum(self._ppo_std, learnable_std)

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def stage1_update(
        self,
        training_state: TwoStageBCTrainingState,
        transitions: PPOTransition,
        key: RNGKey,
    ) -> Tuple[TwoStageBCTrainingState, TwoStageBCTrainingMetrics]:
        """Run ``bc_epochs`` epochs of stage-1 advantage-weighted NLL.

        Both the component means and the shared std are learned; the
        effective std is floored at ``PPO_std``. Returns the updated
        training state, which is meant to be fed into :meth:`stage2_update`
        (after the caller sets ``stage1_std`` via :meth:`stage1_final_std`).

        ``transitions`` must already be batched into mini-batches, i.e. have
        leading shape ``(num_mini_batches, mini_batch_size, ...)``.
        """
        (policy_params, policy_opt_state, final_loss, key), _ = jax.lax.scan(
            lambda carry, _: self._train_stage1_epoch(carry, transitions),
            (
                training_state.policy_params,
                training_state.policy_opt_state,
                training_state.ema_loss,
                key,
            ),
            length=self.configs.bc_epochs,
        )

        new_training_state = TwoStageBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
            stage1_std=training_state.stage1_std,
        )
        return new_training_state, TwoStageBCTrainingMetrics(bc_loss=final_loss)

    @partial(jax.jit, static_argnames=("self",))
    def stage2_update(
        self,
        training_state: TwoStageBCTrainingState,
        transitions: PPOTransition,
        key: RNGKey,
        alpha: jax.Array,
    ) -> Tuple[TwoStageBCTrainingState, TwoStageBCTrainingMetrics]:
        """Run ``bc_epochs`` epochs of stage-2 importance-sampling PPO.

        The shared std is frozen (``stop_gradient``) and linearly
        interpolated from the stage-1 (clipped) std toward ``PPO_std`` by
        ``alpha`` in ``[0, 1]`` (0 -> stage-1 std, 1 -> PPO_std). ``alpha``
        is a traced scalar, so this stays jit-able without recompilation and
        the caller can impose any anneal schedule.

        ``transitions`` must already be batched into mini-batches, i.e. have
        leading shape ``(num_mini_batches, mini_batch_size, ...)``.
        """
        std_logits = self.annealed_std_logits(training_state.stage1_std, alpha)
        (policy_params, policy_opt_state, final_loss, key), _ = jax.lax.scan(
            lambda carry, _: self._train_stage2_epoch(carry, transitions, std_logits),
            (
                training_state.policy_params,
                training_state.policy_opt_state,
                training_state.ema_loss,
                key,
            ),
            length=self.configs.bc_epochs,
        )

        new_training_state = TwoStageBCTrainingState(
            policy_params=policy_params,
            policy_opt_state=policy_opt_state,
            ema_loss=final_loss,
            step_num=training_state.step_num + 1,
            stage1_std=training_state.stage1_std,
        )
        return new_training_state, TwoStageBCTrainingMetrics(bc_loss=final_loss)

    @partial(jax.jit, static_argnames=("self",))
    def _train_stage1_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float, RNGKey],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float, RNGKey], Any]:
        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss, key = carry
            grad, loss = jax.grad(self._stage1_loss_fn, has_aux=True)(
                policy_params, mini_batch
            )
            new_loss = loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss
            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)
            return (new_params, new_opt_state, new_loss, key), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None

    @partial(jax.jit, static_argnames=("self",))
    def _train_stage2_epoch(
        self,
        carry: Tuple[Params, optax.OptState, float, RNGKey],
        transitions: PPOTransition,
        std_logits: jax.Array,
    ) -> Tuple[Tuple[Params, optax.OptState, float, RNGKey], Any]:
        def scan_step(carry, mini_batch):
            policy_params, policy_opt_state, current_loss, key = carry
            grad, loss = jax.grad(self._stage2_loss_fn, has_aux=True)(
                policy_params, mini_batch, std_logits
            )
            new_loss = loss * (1 - self.ema_alpha) + self.ema_alpha * current_loss
            updates, new_opt_state = self._policy_optimizer.update(
                grad, policy_opt_state
            )
            new_params = optax.apply_updates(policy_params, updates)
            return (new_params, new_opt_state, new_loss, key), None

        final_carry, _ = jax.lax.scan(scan_step, carry, transitions)
        return final_carry, None
