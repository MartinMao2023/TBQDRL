from flax.struct import dataclass
from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from optax.losses import sigmoid_binary_cross_entropy

from data_struct.distillation_transitions import GMMDistillationTransition
from data_struct import PPOTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper
from networks import GC_multi_Policy, GC_Selector


@dataclass
class CombinedDistillConfigs:
    action_learning_rate: float = 3e-4
    selector_learning_rate: float = 3e-4
    stage1_epochs: int = 16
    stage2_epochs: int = 1
    mini_batch_size: int = 2048
    # Stage-1 IS upper clip: stop the update once (new_log_p - old_log_p) exceeds this.
    clip_log_ratio: float = 0.2
    # Number of teacher-aligned components. The rest (component_num - k1) are
    # the relocated/demo components fit by the NLL->IS term.
    k1: int = None


class CombinedDistillTrainingState(PyTreeNode):
    """Training state for the two-stage decoupled GMM distillation learner."""

    policy_params: Params       # GC_multi_Policy: means + shared std
    selector_params: Params     # GC_Selector: weight_logits
    policy_opt_state: optax.OptState
    selector_opt_state: optax.OptState
    ema_loss: jax.Array
    step_num: int


class CombinedDistillMetrics(PyTreeNode):
    bc_loss: jax.Array


class CombinedDistill:
    """Two-stage decoupled GMM distillation.

    Student = GC_multi_Policy (means + shared std) + GC_Selector (weight_logits),
    with k1 teacher-aligned components and the rest (component_num - k1) relocated
    demo components.

    Both stages share a KL term (student first-k1 vs teacher, frozen teacher std)
    and a balance regularizer on the selector's first-k1 vs back weight masses.

    Stage 1 (warmup): demo term on the back components morphs from
    advantage-weighted NLL into advantage-weighted importance sampling with an
    upper clip, via a per-epoch beta that ramps 0 -> 1 (final 20% pure IS). The
    shared std is learned but floored at the teacher std.

    Stage 2 (fine-tune): demo term is the IS surrogate only (beta = 1); the std
    is annealed from the stage-1-end std toward the teacher std (frozen) and at
    the end the student's std_logits are set to the teacher's.

    Per-step data is a tuple (GMMDistillationTransition, PPOTransition)
    whose leading two axes carry the data size. :meth:`distill` reshapes both
    to (num_mini_batches, mini_batch_size, ...) — with num_mini_batches derived
    from the data — and checks the two leading sizes match.
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: GC_multi_Policy,
        selector_network: GC_Selector,
        teacher_std_logits: jax.Array,
        bc_configs: CombinedDistillConfigs,
    ):
        self._env = env
        self.configs = bc_configs
        self._policy_network = policy_network
        self._selector_network = selector_network
        self._teacher_std_logits = teacher_std_logits
        self._teacher_std = nn.sigmoid(teacher_std_logits)                     # (d,)
        self._teacher_inv_var = jnp.square(jnp.exp(-teacher_std_logits) + 1)   # (d,)

        component_num = policy_network.component_num
        k1 = bc_configs.k1
        if k1 is None:
            k1 = component_num // 2
        if not (1 <= k1 < component_num):
            raise Exception(
                f"k1 must be in [1, component_num); "
                f"got k1={k1}, component_num={component_num}"
            )
        self.k1 = k1

        self._max_ratio = jnp.exp(bc_configs.clip_log_ratio)

        def make_optimizer(learning_rate):
            return optax.adam(learning_rate=learning_rate)

        self._policy_optimizer = optax.inject_hyperparams(make_optimizer)(
            learning_rate=bc_configs.action_learning_rate
        )
        self._selector_optimizer = optax.inject_hyperparams(make_optimizer)(
            learning_rate=bc_configs.selector_learning_rate
        )

        # Stage 1: floored learnable std (max(ppo_std, std_logits2)); beta is
        # passed per epoch by the caller.
        stage1_demo_std_fn = lambda std2, ov: jnp.maximum(ov, std2)
        self._stage1_loss_fn = self._build_loss_fn(
            stage1_demo_std_fn,
            fixed_beta=None,
            fixed_override=self._teacher_std_logits,
        )

        # Stage 2: scheduled frozen std (the override); IS only (beta = 1).
        stage2_demo_std_fn = lambda std2, ov: ov
        self._stage2_loss_fn = self._build_loss_fn(
            stage2_demo_std_fn,
            fixed_beta=1.0,
            fixed_override=None,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _build_loss_fn(
        self,
        demo_std_fn: Callable[[jax.Array, jax.Array], jax.Array],
        fixed_beta,
        fixed_override,
    ) -> Callable:
        """Build a stage loss fn with no if-else inside.

        demo_std_fn maps (std_logits2, std_logits_override) -> demo_std_logits.
        fixed_beta / fixed_override partial one of those two args; the other
        remains a per-call argument.
        """
        policy_network = self._policy_network
        selector_network = self._selector_network
        k1 = self.k1
        teacher_inv_var = self._teacher_inv_var
        teacher_std = self._teacher_std
        max_ratio = self._max_ratio

        def loss_fn(
            policy_params: Params,
            selector_params: Params,
            transitions: Tuple[GMMDistillationTransition, PPOTransition],
            key: RNGKey,
            beta: jax.Array,
            std_logits_override: jax.Array,
        ) -> Tuple[float, jax.Array]:
            gmm_t, demo_t = transitions

            action_means1, _ = policy_network.apply(
                policy_params, gmm_t.obs, gmm_t.zs
            )  # B, K, d
            w_logits1 = selector_network.apply(
                selector_params, gmm_t.obs, gmm_t.zs
            )  # B, K
            action_means2, std_logits2 = policy_network.apply(
                policy_params, demo_t.obs, demo_t.zs
            )  # B, K, d ; d
            w_logits2 = selector_network.apply(
                selector_params, demo_t.obs, demo_t.zs
            )  # B, K

            # ---- KL: student first-k1 vs teacher (all components, teacher std) ----
            student_means_k1 = action_means1[:, :k1]
            student_w_k1 = w_logits1[:, :k1]
            student_w_k1_sg = jax.lax.stop_gradient(student_w_k1)
            # selector trained only through the bounded softmax weighting (not via log_q)
            component_weights_k1 = nn.softmax(student_w_k1)     # B, k1
            sampled = jax.random.normal(key, student_means_k1.shape) * teacher_std \
                + student_means_k1                                        # B, k1, d  (non-sg: landing)
            d_q = jnp.square(
                sampled[:, :, None, :]
                - jax.lax.stop_gradient(student_means_k1)[:, None, :, :]
            )  # B, k1, k1, d
            log_q = student_w_k1_sg[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * d_q, axis=-1)     # B, k1, k1
            d_p = jnp.square(
                sampled[:, :, None, :] - gmm_t.action_means[:, None, :, :]
            )  # B, k1, Kt, d
            log_p = gmm_t.component_logits[:, None, :] \
                - 0.5 * jnp.sum(teacher_inv_var * d_p, axis=-1)    # B, k1, Kt
            log_ratio_kl = (
                nn.logsumexp(log_q, axis=-1)
                - nn.logsumexp(student_w_k1_sg, axis=-1, keepdims=True)
            ) - (
                nn.logsumexp(log_p, axis=-1)
                - nn.logsumexp(gmm_t.component_logits, axis=-1, keepdims=True)
            )  # B, k1
            kl = jnp.mean(jnp.sum(log_ratio_kl * component_weights_k1, axis=-1))

            # ---- Demo term on the back components (slice [:, k1:]) ----
            means_k2 = action_means2[:, k1:]                       # B, k2, d
            w_k2 = w_logits2[:, k1:]                               # B, k2
            demo_std_logits = demo_std_fn(std_logits2, std_logits_override)
            inv_var = jnp.square(jnp.exp(-demo_std_logits) + 1)    # (d,)
            log_norm = jnp.sum(
                nn.log_sigmoid(demo_std_logits), axis=-1, keepdims=True
            )  # (1,)
            per_comp = w_k2 - 0.5 * jnp.sum(
                inv_var * jnp.square(means_k2 - demo_t.actions[:, None, :]),
                axis=-1,
            )  # B, k2
            new_log_p_k2 = (
                nn.logsumexp(per_comp, axis=-1, keepdims=True)
                - nn.logsumexp(w_k2, axis=-1, keepdims=True)
                - log_norm
            )  # B, 1

            nll = -jnp.mean(demo_t.weights * new_log_p_k2)
            ratio = jnp.exp(new_log_p_k2 - demo_t.log_likelihood)   # B, 1
            ratio = jnp.minimum(max_ratio, ratio)
            surrogate = -jnp.mean(demo_t.weights * ratio)
            demo_term = (1.0 - beta) * nll + beta * surrogate

            # ---- Balance regularizer on the selector's weight_logits ----
            log_wr1 = nn.logsumexp(w_logits1[:, :k1], axis=-1) \
                - nn.logsumexp(w_logits1[:, k1:], axis=-1)         # B
            log_wr2 = nn.logsumexp(w_logits2[:, :k1], axis=-1) \
                - nn.logsumexp(w_logits2[:, k1:], axis=-1)         # B
            balance = jnp.mean(
                sigmoid_binary_cross_entropy(log_wr1, 0.5)
                + sigmoid_binary_cross_entropy(log_wr2, 0.5)
            )

            average_diff = 0.5 * (jnp.mean(
                jnp.abs(nn.sigmoid(log_wr1) - 0.5) + jnp.abs(nn.sigmoid(log_wr2) - 0.5)
                ))

            # loss = kl + demo_term + 
            loss = balance + demo_term + kl
            return loss, jnp.array([kl, demo_term, average_diff])

        if fixed_beta is not None:
            loss_fn = partial(loss_fn, beta=fixed_beta)
        if fixed_override is not None:
            loss_fn = partial(loss_fn, std_logits_override=fixed_override)
        return loss_fn

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def init(self, key: RNGKey) -> CombinedDistillTrainingState:
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_params = {
            **policy_params,
            "params": {
                **policy_params["params"],
                "std_logits": self._teacher_std_logits,
            },
        }
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        selector_params = self._selector_network.init(subkey, obs=fake_obs, z=fake_zs)
        selector_opt_state = self._selector_optimizer.init(selector_params)

        return CombinedDistillTrainingState(
            policy_params=policy_params,
            selector_params=selector_params,
            policy_opt_state=policy_opt_state,
            selector_opt_state=selector_opt_state,
            ema_loss=jnp.array([1.0, 5.0, 0.0]),  # kl, demo_term, balance
            step_num=0,
        )

    # ------------------------------------------------------------------
    # Std anneal (stage 2), mirroring two_stage_bc
    # ------------------------------------------------------------------
    def annealed_std_logits(self, stage1_std: jax.Array, alpha: jax.Array) -> jax.Array:
        std = (1.0 - alpha) * stage1_std + alpha * self._teacher_std
        return jax.lax.stop_gradient(-jnp.log(1.0 / std - 1.0 + 1e-8))

    def stage1_final_std(self, training_state: CombinedDistillTrainingState) -> jax.Array:
        std_logits = training_state.policy_params["params"]["std_logits"]
        return jnp.maximum(self._teacher_std, nn.sigmoid(std_logits))

    # ------------------------------------------------------------------
    # Stage 1: one-epoch helper (beta per gradient step from carry)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def _train_stage1_epoch(
        self,
        carry,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
        coefficient: jax.Array,
        ema_alpha: jax.Array,
    ):
        def scan_step(carry, mini_batch):
            (
                policy_params, policy_opt_state,
                selector_params, selector_opt_state,
                current_loss, grad_step_num, key,
            ) = carry
            key, subkey = jax.random.split(key)
            beta = jnp.clip(grad_step_num * coefficient, 0.0, 1.0)

            (p_grad, s_grad), loss = jax.grad(
                self._stage1_loss_fn, has_aux=True, argnums=(0, 1)
            )(policy_params, selector_params, mini_batch, subkey, beta)

            new_loss = loss * (1 - ema_alpha) + ema_alpha * current_loss
            p_updates, new_p_opt = self._policy_optimizer.update(p_grad, policy_opt_state)
            new_p = optax.apply_updates(policy_params, p_updates)
            s_updates, new_s_opt = self._selector_optimizer.update(s_grad, selector_opt_state)
            new_s = optax.apply_updates(selector_params, s_updates)

            return (new_p, new_p_opt, new_s, new_s_opt, new_loss, grad_step_num + 1, key), None

        final, _ = jax.lax.scan(scan_step, carry, transitions)
        return final, None

    # ------------------------------------------------------------------
    # Stage 1: outer scan over stage1_epochs (coefficient from distill)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def state1_update(
        self,
        training_state: CombinedDistillTrainingState,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
        key: RNGKey,
        coefficient: jax.Array,
        ema_alpha: jax.Array,
    ) -> Tuple[CombinedDistillTrainingState, CombinedDistillMetrics]:
        init = (
            training_state.policy_params, training_state.policy_opt_state,
            training_state.selector_params, training_state.selector_opt_state,
            training_state.ema_loss,
            jnp.array(0, dtype=jnp.int32),
            key,
        )
        final, _ = jax.lax.scan(
            lambda x, _: self._train_stage1_epoch(x, transitions, coefficient, ema_alpha),
            init,
            length=self.configs.stage1_epochs,
        )

        num_mini_batches = transitions[0].obs.shape[0]
        new_state = CombinedDistillTrainingState(
            policy_params=final[0],
            policy_opt_state=final[1],
            selector_params=final[2],
            selector_opt_state=final[3],
            ema_loss=final[4],
            step_num=training_state.step_num + self.configs.stage1_epochs * num_mini_batches,
        )
        return new_state, CombinedDistillMetrics(bc_loss=final[4])

    # ------------------------------------------------------------------
    # Stage 2: one-epoch helper (std annealed per gradient step from carry)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def _train_stage2_epoch(
        self,
        carry,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
        stage1_end_std: jax.Array,
        coefficient: jax.Array,
        ema_alpha: jax.Array,
    ):
        def scan_step(carry, mini_batch):
            (
                policy_params, policy_opt_state,
                selector_params, selector_opt_state,
                current_loss, grad_step_num, key,
            ) = carry
            key, subkey = jax.random.split(key)
            alpha = grad_step_num * coefficient
            std_logits_override = self.annealed_std_logits(stage1_end_std, alpha)

            (p_grad, s_grad), loss = jax.grad(
                self._stage2_loss_fn, has_aux=True, argnums=(0, 1)
            )(policy_params, selector_params, mini_batch, subkey, std_logits_override=std_logits_override)

            new_loss = loss * (1 - ema_alpha) + ema_alpha * current_loss
            p_updates, new_p_opt = self._policy_optimizer.update(p_grad, policy_opt_state)
            new_p = optax.apply_updates(policy_params, p_updates)
            s_updates, new_s_opt = self._selector_optimizer.update(s_grad, selector_opt_state)
            new_s = optax.apply_updates(selector_params, s_updates)

            return (new_p, new_p_opt, new_s, new_s_opt, new_loss, grad_step_num + 1, key), None

        final, _ = jax.lax.scan(scan_step, carry, transitions)
        return final, None

    # ------------------------------------------------------------------
    # Stage 2: outer scan over stage2_epochs (coefficient from distill)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def state2_update(
        self,
        training_state: CombinedDistillTrainingState,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
        key: RNGKey,
        stage1_end_std: jax.Array,
        coefficient: jax.Array,
        ema_alpha: jax.Array,
    ) -> Tuple[CombinedDistillTrainingState, CombinedDistillMetrics]:
        init = (
            training_state.policy_params, training_state.policy_opt_state,
            training_state.selector_params, training_state.selector_opt_state,
            training_state.ema_loss,
            jnp.array(0, dtype=jnp.int32),
            key,
        )
        final, _ = jax.lax.scan(
            lambda x, _: self._train_stage2_epoch(
                x, transitions, stage1_end_std, coefficient, ema_alpha
            ),
            init,
            length=self.configs.stage2_epochs,
        )

        new_state = CombinedDistillTrainingState(
            policy_params=final[0],
            policy_opt_state=final[1],
            selector_params=final[2],
            selector_opt_state=final[3],
            ema_loss=final[4],
            step_num=training_state.step_num + self.configs.stage2_epochs * transitions[0].obs.shape[0],
        )
        return new_state, CombinedDistillMetrics(bc_loss=final[4])

    # ------------------------------------------------------------------
    # distill: Python orchestrator combining both stages
    # ------------------------------------------------------------------
    def distill(
        self,
        training_state: CombinedDistillTrainingState,
        transitions: Tuple[GMMDistillationTransition, PPOTransition],
        key: RNGKey,
    ) -> Tuple[CombinedDistillTrainingState, CombinedDistillMetrics]:
        """Run stage 1 then stage 2, with the shape check, beta schedule and
        std anneal handled here. Not jitted; calls the jitted stage updates.
        """
        gmm_t, demo_t = transitions
        mini_batch_size = self.configs.mini_batch_size

        # data_size is the product of the leading two axes; we don't know it in
        # advance, so num_mini_batches is derived from the data, not from config.
        # Reshape both buffers to (num_mini_batches, mini_batch_size, *feat).
        def _reshape(t):
            return jax.tree.map(
                lambda x: jnp.reshape(x, (-1, mini_batch_size, *x.shape[2:])), t
            )
        gmm_t = _reshape(gmm_t)
        demo_t = _reshape(demo_t)
        transitions = (gmm_t, demo_t)

        # Shape check after reshape: both buffers must yield the same
        # (num_mini_batches, mini_batch_size) so the per-epoch scan lines up.
        assert gmm_t.obs.shape[:2] == demo_t.obs.shape[:2], (
            "teacher (GMMDistillationTransition) and demonstration "
            "(PPOTransition) must have the same leading size"
        )

        s1_epochs = self.configs.stage1_epochs
        s2_epochs = self.configs.stage2_epochs
        num_mini_batches = gmm_t.obs.shape[0]
        s1_total = s1_epochs * num_mini_batches
        s2_total = s2_epochs * num_mini_batches
        # coefficient computed in distill and passed into the state_updates;
        # beta/alpha per gradient step = grad_step_num * coefficient (clipped).
        coefficient1 = jnp.float32(1.25 / (s1_total - 1))
        coefficient2 = jnp.float32(1.0 / (s2_total - 1))
        ema_alpha = jnp.exp(jnp.array(-2.0 / num_mini_batches))

        # ---- Stage 1: beta ramps 0 -> 1 per gradient step, final 20% pure IS ----
        training_state, _ = self.state1_update(
            training_state, transitions, key, coefficient1, ema_alpha
        )
        key, _ = jax.random.split(key)

        # ---- Stage boundary: snapshot the effective std ----
        stage1_end_std = self.stage1_final_std(training_state)

        # ---- Stage 2: IS only, std annealed stage1_end_std -> teacher_std ----
        training_state, _ = self.state2_update(
            training_state, transitions, key, stage1_end_std, coefficient2, ema_alpha
        )
        key, _ = jax.random.split(key)

        # Fix the student's std at the teacher std.
        final_p = {
            **training_state.policy_params,
            "params": {
                **training_state.policy_params["params"],
                "std_logits": self._teacher_std_logits,
            },
        }
        training_state = training_state.replace(policy_params=final_p)

        return training_state, CombinedDistillMetrics(bc_loss=training_state.ema_loss)

