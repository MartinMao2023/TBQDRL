from flax.struct import dataclass
from functools import partial
from typing import Any, Tuple, Optional

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from flax import serialization

from data_struct import PPOTransition
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper


@dataclass
class EnsembleCriticConfigs:
    critic_learning_rate: float = 5e-4
    discount: float = 0.99
    td_lambda_discount: float = 0.95
    rollout_length: int = 32
    vec_env: int = 4096
    mini_batch_size: int = 8192
    # number of rollout iterations collected by collect_rollout_data (m)
    num_data_iterations: int = 32
    resample_prob: float = 0.5
    # number of gradient epochs per fine-tuning iteration (m)
    num_train_epochs: int = 4
    # number of critic snapshots saved over the fine-tuning scan (k). Each
    # fine-tuning iteration uses 1/k of the trajectories, so k must divide
    # num_data_iterations * vec_env.
    num_snapshots: int = 16
    # TD(lambda) targets are clipped to old_v +/- 3 * target_clip_rmse to
    # suppress outlier values. This is the final PPO critic RMSE (real-return
    # space).
    target_clip_rmse: float = 17.62787


class EnsembleCriticState(PyTreeNode):
    """Carries the frozen policy, the trainable critic (warm-started from the
    loaded old critic), the frozen old critic used as the clipping / bootstrap
    reference, and the fixed value-normalization statistics."""

    policy_params: Params          # frozen, only used to collect rollouts
    critic_params: Params          # trainable, warm-started from old_critic
    critic_opt_state: optax.OptState

    old_critic_params: Params      # frozen; clipping reference + iter-1 bootstrap
    moving_mean: jax.Array         # fixed, loaded from disk
    moving_squared_diff: jax.Array  # fixed, loaded from disk
    iteration_num: int


class StackedCritics(PyTreeNode):
    """Ensemble of k critic parameter snapshots produced by fine_tune_scan.

    ``critic_params`` is a params PyTree whose leaves carry a leading axis of
    size k (one per snapshot). The fixed normalization stats are stored
    alongside so ``predict`` is self-contained.
    """

    critic_params: Params
    moving_mean: jax.Array
    moving_squared_diff: jax.Array


class EnsembleCriticMetrics(PyTreeNode):
    critic_error: jax.Array        # per-snapshot EMA RMSE (real-return space)


class EnsembleCriticFinetuner:
    """Fine-tunes a goal-conditioned critic against a frozen goal-conditioned
    policy, saving k snapshots of the critic as it evolves so the prediction
    mean / uncertainty can be modelled afterwards.

    Pipeline:
      * ``collect_rollout_data`` collects raw rollout data once. A trajectory
        is one env's full ``rollout_length`` sequence plus its final state;
        there are ``num_data_iterations * vec_env`` trajectories.
      * ``fine_tune_scan`` shuffles all trajectories and partitions them into k
        disjoint groups (one per fine-tuning iteration; trajectories are never
        broken and each is used exactly once). Each iteration computes clipped
        TD(lambda) targets with the *last* critic, shuffles the transitions,
        trains ``num_train_epochs`` epochs, and snapshots the critic.
      * ``predict`` applies all k snapshots to query states and returns the
        stacked (k, batch, 1) predictions.

    Workflow::

        state = ft.init(key, policy_path, critic_path, mean_path, var_path)
        (states, key), transitions, final_obs, final_zs = \\
            ft.collect_rollout_data(states, state, key)
        state, metrics, ensemble = ft.fine_tune_scan(
            state, transitions, final_obs, final_zs, key)
        preds = ft.predict(ensemble, query_obs, query_zs)   # (k, batch, 1)
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        configs: EnsembleCriticConfigs,
    ):

        self._env = env
        self.configs = configs
        self._policy_network = policy_network
        self._critic_network = critic_network
        # The old critic shares the architecture of the trainable critic
        # (warm start), so the same network module is reused.

        num_traj = configs.num_data_iterations * configs.vec_env
        if num_traj % configs.num_snapshots != 0:
            raise ValueError(
                "num_data_iterations * vec_env (= "
                f"{num_traj}) must be divisible by num_snapshots (="
                f"{configs.num_snapshots})."
            )
        self.trajectories_per_snapshot = num_traj // configs.num_snapshots

        train_transitions = (
            self.trajectories_per_snapshot * configs.rollout_length
        )
        self.mini_batch_num = train_transitions // configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2 / self.mini_batch_num)

        self._critic_optimizer = optax.adam(
            learning_rate=configs.critic_learning_rate,
        )
        self._clip_half_width = 3.0 * configs.target_clip_rmse

        # ------------------------------------------------------------------
        # Rollout collection with the frozen policy (raw transitions only;
        # targets are computed inside fine_tune_scan with the evolving critic).
        # ------------------------------------------------------------------
        @jax.jit
        def rollout_fn(
            policy_params: Params,
            starting_states,
            keys: RNGKey,
        ) -> Tuple[Any, PPOTransition]:

            def play_step_fn(carry):
                state, key = carry
                obs, z = self._env.get_obs(state)
                action_mean, std_logits = self._policy_network.apply(
                    policy_params, obs, z
                )
                action_std = nn.sigmoid(std_logits)

                key, subkey = jax.random.split(key)
                noise = action_std * jax.random.normal(subkey, action_mean.shape)
                action = jnp.clip(action_mean + noise, -1.0, 1.0)

                state, transition_info = self._env.step(state, action)

                transition = PPOTransition(
                    obs=obs,
                    actions=action,
                    zs=z,
                    log_likelihood=jnp.zeros((1,)),
                    rewards=transition_info.reward,
                    td_lambda_returns=jnp.zeros((1,)),
                    gaes=jnp.zeros((1,)),
                    dones=transition_info.done,
                    truncations=transition_info.truncation,
                    weights=jnp.zeros((1,)),
                )
                return (state, key), transition

            (final_states, _), transitions = jax.lax.scan(
                lambda x, _: jax.vmap(play_step_fn)(x),
                (starting_states, keys),
                length=self.configs.rollout_length,
            )

            return final_states, transitions

        self._rollout_fn = rollout_fn

        # ------------------------------------------------------------------
        # Critic loss (regression toward the stored, clipped TD(lambda) target).
        # ------------------------------------------------------------------
        def critic_loss_fn(critic_params: Params, transitions: PPOTransition):
            estimated_v = self._critic_network.apply(
                critic_params, transitions.obs, transitions.zs
            )
            weights = 1 / (1 + jnp.square(transitions.weights))
            loss = jnp.average(
                jnp.square(estimated_v - transitions.td_lambda_returns),
                weights=weights,
            )
            return loss, jnp.sqrt(loss)

        self._critic_loss_fn = critic_loss_fn

    # ------------------------------------------------------------------
    # Initialisation: load frozen policy + old critic (and fixed stats); the
    # trainable critic is warm-started from the loaded old critic params.
    # ------------------------------------------------------------------
    def init(
        self,
        key: RNGKey,
        policy_path: str,
        critic_path: str,
        mean_path: Optional[str] = None,
        var_path: Optional[str] = None,
    ) -> EnsembleCriticState:

        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_template = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        with open(policy_path, "rb") as f:
            policy_params = serialization.from_bytes(policy_template, f.read())

        key, subkey = jax.random.split(key)
        critic_template = self._critic_network.init(subkey, obs=fake_obs, z=fake_zs)
        with open(critic_path, "rb") as f:
            old_critic_params = serialization.from_bytes(
                critic_template, f.read()
            )

        # Warm start: the trainable critic begins from the loaded old critic.
        critic_params = old_critic_params
        critic_opt_state = self._critic_optimizer.init(critic_params)

        if mean_path is not None:
            moving_mean = jnp.load(mean_path)
        else:
            moving_mean = jnp.array(0.0)
        if var_path is not None:
            moving_squared_diff = jnp.load(var_path)
        else:
            moving_squared_diff = jnp.array(1.0)

        return EnsembleCriticState(
            policy_params=policy_params,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            old_critic_params=old_critic_params,
            moving_mean=moving_mean,
            moving_squared_diff=moving_squared_diff,
            iteration_num=0,
        )

    # ------------------------------------------------------------------
    # Data collection: m rollout iterations with probabilistic task
    # resampling. Returns RAW transitions (m, rollout_length, vec_env, ...) and
    # per-iteration final (obs, zs). No targets are computed here; fine_tune_scan
    # computes them with the evolving critic.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def collect_rollout_data(
        self,
        starting_states,
        training_state: EnsembleCriticState,
        key: RNGKey,
    ) -> Tuple[Tuple[Any, RNGKey], PPOTransition, jax.Array, jax.Array]:

        policy_params = training_state.policy_params
        resample_prob = self.configs.resample_prob
        vec_env = self.configs.vec_env

        def data_step(carry, _):
            states, key = carry

            key, subkey = jax.random.split(key)
            resample_mask = jax.random.bernoulli(
                subkey, resample_prob, shape=(vec_env,)
            )
            resampled_states = jax.vmap(self._env.resample_task_state)(states)
            states = jax.tree.map(
                lambda a, b: jax.vmap(jax.lax.select)(resample_mask, a, b),
                resampled_states,
                states,
            )

            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, num=vec_env)
            final_states, transitions = self._rollout_fn(
                policy_params, states, subkeys
            )
            final_obs, final_zs = self._env.get_obs(final_states)

            return (final_states, key), (transitions, final_obs, final_zs)

        (final_states, key), (transitions, final_obs, final_zs) = jax.lax.scan(
            data_step,
            (starting_states, key),
            length=self.configs.num_data_iterations,
        )

        return (final_states, key), transitions, final_obs, final_zs

    # ------------------------------------------------------------------
    # Fine-tuning scan: k iterations.
    #   * Reorganise the collected data into trajectories (one env's full
    #     rollout_length sequence + its final state); N = m * vec_env of them.
    #   * Shuffle the trajectories and split into k disjoint groups (each
    #     trajectory used exactly once, never broken, final state paired).
    #   * Per iteration: compute clipped TD(lambda) targets with the CURRENT
    #     critic, shuffle the transitions, train num_train_epochs epochs, and
    #     emit a snapshot of the trained critic.
    # The trained critic is carried forward so the next iteration bootstraps
    # from it. Returns (new_state, metrics, StackedCritics).
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def fine_tune_scan(
        self,
        training_state: EnsembleCriticState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        key: RNGKey,
    ) -> Tuple[EnsembleCriticState, EnsembleCriticMetrics, StackedCritics]:

        moving_mean = training_state.moving_mean
        moving_std = jnp.sqrt(training_state.moving_squared_diff)
        m_data = self.configs.num_data_iterations
        vec_env = self.configs.vec_env
        k = self.configs.num_snapshots
        num_traj = m_data * vec_env
        p_traj = self.trajectories_per_snapshot

        # --- Reorganise into trajectory-centric layout: (N, T, ...) ---
        def to_traj_layout(x: jax.Array) -> jax.Array:
            # (m, T, V, d...) -> (m, V, T, d...) -> (N, T, d...)
            x = jnp.transpose(
                x, [0, 2, 1] + list(range(3, x.ndim))
            )
            return jnp.reshape(x, (num_traj,) + x.shape[2:])

        traj_trans = jax.tree.map(to_traj_layout, transitions)
        traj_final_obs = jnp.reshape(final_obs, (num_traj,) + final_obs.shape[2:])
        traj_final_zs = jnp.reshape(final_zs, (num_traj,) + final_zs.shape[2:])

        # --- Old-critic V (real-return space) and weights, computed once. ---
        v_old = self._critic_network.apply(
            training_state.old_critic_params, traj_trans.obs, traj_trans.zs
        )  # (N, T, 1)
        final_v_old = self._critic_network.apply(
            training_state.old_critic_params, traj_final_obs, traj_final_zs
        )  # (N, 1)
        v_old_real = v_old * moving_std + moving_mean
        final_v_old_real = final_v_old * moving_std + moving_mean
        _, traj_weights = self._td_lambda_per_trajectory(
            final_v_old_real, v_old_real, traj_trans
        )  # (N, T, 1)

        # --- Shuffle trajectories and partition into k groups. ---
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_traj)
        traj_trans = jax.tree.map(lambda x: x[perm], traj_trans)
        traj_final_obs = traj_final_obs[perm]
        traj_final_zs = traj_final_zs[perm]
        v_old_real = v_old_real[perm]
        traj_weights = traj_weights[perm]

        def to_groups(x: jax.Array) -> jax.Array:
            return jnp.reshape(x, (k, p_traj) + x.shape[1:])

        grouped_trans = jax.tree.map(to_groups, traj_trans)
        grouped_final_obs = to_groups(traj_final_obs)
        grouped_final_zs = to_groups(traj_final_zs)
        grouped_v_old = to_groups(v_old_real)
        grouped_weights = to_groups(traj_weights)

        # --- Scan over the k groups. ---
        def scan_step(carry, group):
            critic_params, opt_state, error_ema, key = carry
            (
                group_trans,
                group_final_obs,
                group_final_zs,
                group_v_old,
                group_weights,
            ) = group

            # Clipped TD(lambda) targets with the current critic.
            td_norm = self._compute_group_targets(
                critic_params,
                group_trans,
                group_final_obs,
                group_final_zs,
                group_v_old,
                moving_mean,
                moving_std,
            )
            group_trans = group_trans.replace(
                td_lambda_returns=td_norm,
                gaes=td_norm,
                weights=group_weights,
            )

            # Train num_train_epochs epochs (transitions reshuffled each epoch).
            (critic_params, opt_state, error_ema, key), _ = self._train_epochs(
                (critic_params, opt_state, error_ema, key), group_trans
            )

            return (critic_params, opt_state, error_ema, key), (
                critic_params,
                error_ema,
            )

        init_carry = (
            training_state.critic_params,
            training_state.critic_opt_state,
            jnp.array(0.0),
            key,
        )
        final_carry, (snapshots, error_history) = jax.lax.scan(
            scan_step,
            init_carry,
            (grouped_trans, grouped_final_obs, grouped_final_zs,
             grouped_v_old, grouped_weights),
        )
        new_critic_params, new_opt_state, _, _ = final_carry

        new_training_state = training_state.replace(
            critic_params=new_critic_params,
            critic_opt_state=new_opt_state,
            iteration_num=training_state.iteration_num + k,
        )

        metrics = EnsembleCriticMetrics(critic_error=error_history * moving_std)
        ensemble = StackedCritics(
            critic_params=snapshots,
            moving_mean=moving_mean,
            moving_squared_diff=training_state.moving_squared_diff,
        )

        return new_training_state, metrics, ensemble

    # ------------------------------------------------------------------
    # Ensemble prediction: apply all k saved critics to query (obs, zs) and
    # return the STACKED (k, batch, 1) predictions in real-return space.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def predict(
        self,
        ensemble: StackedCritics,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:

        moving_std = jnp.sqrt(ensemble.moving_squared_diff)
        preds = jax.vmap(
            lambda p: self._critic_network.apply(p, obs, zs)
        )(ensemble.critic_params)  # (k, batch, 1)
        return preds * moving_std + ensemble.moving_mean

    # ------------------------------------------------------------------
    # Per-trajectory TD(lambda) returns. Inputs are leading-axis N (or P)
    # trajectories with v_values (N, T, 1) and final_v (N, 1). Returns
    # (td_lambda_returns, weights) of shape (N, T, 1).
    # ------------------------------------------------------------------
    def _td_lambda_per_trajectory(
        self,
        final_v_real: jax.Array,
        v_values_real: jax.Array,
        transitions: PPOTransition,
    ) -> Tuple[jax.Array, jax.Array]:

        td, weights = jax.vmap(
            self.calculate_td_lambda_returns,
            in_axes=(0, 0, 0, 0, 0),
        )(
            final_v_real[..., None],        # (N, 1, 1)
            v_values_real[..., None],       # (N, T, 1, 1)
            transitions.rewards[..., None],
            transitions.dones[..., None],
            transitions.truncations[..., None],
        )  # (N, T, 1, 1)
        return td[..., 0], weights[..., 0]  # (N, T, 1)

    # ------------------------------------------------------------------
    # Target computation for one group of trajectories with the current critic,
    # clipping the TD(lambda) return to old_v +/- 3 * target_clip_rmse before
    # re-normalizing into the critic's output space.
    # ------------------------------------------------------------------
    def _compute_group_targets(
        self,
        critic_params: Params,
        group_trans: PPOTransition,
        group_final_obs: jax.Array,
        group_final_zs: jax.Array,
        group_v_old_real: jax.Array,
        moving_mean: jax.Array,
        moving_std: jax.Array,
    ) -> jax.Array:

        v_values = self._critic_network.apply(
            critic_params, group_trans.obs, group_trans.zs
        )  # (P, T, 1)
        final_v = self._critic_network.apply(
            critic_params, group_final_obs, group_final_zs
        )  # (P, 1)
        v_values = v_values * moving_std + moving_mean
        final_v = final_v * moving_std + moving_mean

        td_real, _ = self._td_lambda_per_trajectory(
            final_v, v_values, group_trans
        )  # (P, T, 1)

        clipped = jnp.clip(
            # td_real * 0.25 + 0.75 * group_v_old_real,
            td_real,
            group_v_old_real - self._clip_half_width,
            group_v_old_real + self._clip_half_width,
        )
        return (clipped - moving_mean) / (moving_std + 1e-6)

    # ------------------------------------------------------------------
    # Run num_train_epochs epochs of mini-batch gradient descent on a fixed
    # transition set (the per-iteration group). Transitions are reshuffled
    # every epoch (after target computation + clipping).
    # ------------------------------------------------------------------
    def _train_epochs(
        self,
        carry: Tuple[Params, optax.OptState, jax.Array, RNGKey],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, jax.Array, RNGKey], Any]:

        def epoch_step(carry, _):
            params, opt_state, error_ema, key = carry
            key, subkey = jax.random.split(key)
            batched = transitions.shuffle(subkey)
            batched = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (-1, self.configs.mini_batch_size, *x.shape[1:]),
                ),
                batched,
            )
            (params, opt_state, error_ema), _ = self.train_critic(
                (params, opt_state, error_ema), batched
            )
            return (params, opt_state, error_ema, key), None

        final_carry, _ = jax.lax.scan(
            epoch_step, carry, None, length=self.configs.num_train_epochs
        )
        return final_carry, None

    @partial(jax.jit, static_argnames=("self",))
    def train_critic(
        self,
        carry: Tuple[Params, optax.OptState, jax.Array],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, jax.Array], Any]:
        """One mini-batch pass over ``transitions`` (leading axis = number of
        mini-batches)."""

        def scan_train_critic(carry, transition_data):
            (
                current_critic_params,
                current_critic_opt_state,
                current_critic_error,
            ) = carry

            critic_gradient, critic_error = jax.grad(
                self._critic_loss_fn, has_aux=True
            )(current_critic_params, transition_data)

            new_critic_error = (
                critic_error * (1 - self.ema_alpha)
                + self.ema_alpha * current_critic_error
            )

            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_gradient, current_critic_opt_state
            )
            new_critic_params = optax.apply_updates(
                current_critic_params, critic_updates
            )

            return (new_critic_params, new_critic_opt_state, new_critic_error), None

        final_carry, _ = jax.lax.scan(
            scan_train_critic,
            carry,
            transitions,
        )

        return final_carry, None

    # ------------------------------------------------------------------
    # TD(lambda) return computation (identical logic to PPO). Expects, per
    # rollout iteration, v_values of shape (rollout_length, V, 1) and final_v
    # of shape (V, 1); V is 1 here (one trajectory), and the callers vmap over
    # the trajectory axis.
    # ------------------------------------------------------------------
    def calculate_td_lambda_returns(
        self,
        final_v_value: jax.Array,
        v_values: jax.Array,
        rewards: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:

        discount = self.configs.discount
        td_lambda_discount = self.configs.td_lambda_discount

        def scan_calculate_td_lambda(carry, data):
            (last_td_lambda_value, last_value, last_weight) = carry
            reward, v_value, done, truncate = data

            current_td_lambda_value = reward + (1 - done) * discount * (
                (1 - td_lambda_discount) * last_value
                + td_lambda_discount * last_td_lambda_value
            )
            current_td_lambda_value = jnp.where(
                truncate, v_value, current_td_lambda_value
            )
            weight = jnp.where(
                truncate > 0.5,
                1.0,
                (1 - done) * discount * (1 + (last_weight - 1) * td_lambda_discount),
            )

            return (current_td_lambda_value, v_value, weight), (
                current_td_lambda_value,
                weight,
            )

        _, (td_lambda_values, weights) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (final_v_value, final_v_value, jnp.ones_like(final_v_value)),
            (rewards, v_values, termination, truncation),
            reverse=True,
        )

        return td_lambda_values, weights
