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
class CriticFineTuneConfigs:
    critic_learning_rate: float = 5e-4
    discount: float = 0.99
    td_lambda_discount: float = 0.95
    rollout_length: int = 32
    vec_env: int = 256
    mini_batch_size: int = 4096
    # number of rollout iterations produced per generate_rollout_data call (m)
    num_data_iterations: int = 8
    # per-env probability of resampling a new task at the start of each
    # rollout iteration (matches main_ant_mo.py)
    resample_prob: float = 0.5


class CriticFineTuneState(PyTreeNode):
    """Carries the frozen policy, the trainable (new-architecture) critic, the
    frozen stored (old-architecture) critic used to compute value targets, and
    the (fixed) normalization statistics of the value targets."""

    policy_params: Params          # frozen, only used to generate rollouts
    critic_params: Params          # trainable, new architecture
    critic_opt_state: optax.OptState

    old_critic_params: Params      # frozen, old architecture; computes V targets
    moving_mean: jax.Array         # fixed, loaded from disk
    moving_squared_diff: jax.Array  # fixed, loaded from disk
    iteration_num: int


class CriticFineTuneMetrics(PyTreeNode):
    critic_error: jax.Array


class CriticFineTuner:
    """Fine-tunes a goal-conditioned critic against a frozen goal-conditioned
    policy.

    Workflow (the caller drives the outer repetition, e.g. with a Python loop
    or a jax.lax.scan, re-using the latest critic each iteration)::

        state = fine_tuner.init(key, policy_path, critic_path, mean_path, var_path)
        # Generate data once; initial targets come from the frozen old critic.
        (states, key), transitions, final_obs, final_zs = \\
            fine_tuner.generate_rollout_data(states, state, key)
        # Repeatedly fine-tune on the SAME data: each call trains one epoch,
        # then re-bootstraps targets with the updated critic. Feed the returned
        # transitions back in for the next iteration.
        for _ in range(num_bootstrap_steps):
            state, metrics, transitions = fine_tuner.state_update(
                state, transitions, final_obs, final_zs, key)
    """

    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        old_critic_network: nn.Module,
        configs: CriticFineTuneConfigs,
    ):

        self._env = env
        self.configs = configs
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._old_critic_network = old_critic_network

        total_transitions = (
            configs.num_data_iterations
            * configs.vec_env
            * configs.rollout_length
        )
        self.mini_batch_num = total_transitions // configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2 / self.mini_batch_num)

        self._critic_optimizer = optax.adam(
            learning_rate=configs.critic_learning_rate,
        )

        # ------------------------------------------------------------------
        # Rollout collection with the frozen policy.
        # The policy always returns (action_mean, std_logits); we use the
        # policy's own learnable std via sigmoid(std_logits).
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
        # Critic loss (operates on normalized targets, as in PPO).
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
    # Initialisation: load saved policy + critic (and fixed value stats).
    # ------------------------------------------------------------------
    def init(
        self,
        key: RNGKey,
        policy_path: str,
        critic_path: str,
        mean_path: Optional[str] = None,
        var_path: Optional[str] = None,
    ) -> CriticFineTuneState:
        """Load the frozen policy and the frozen old-architecture critic from
        disk (``critic_path`` points at the stored trained critic), and
        randomly initialize the new-architecture critic."""

        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_template = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        with open(policy_path, "rb") as f:
            policy_params = serialization.from_bytes(policy_template, f.read())

        # Stored / old critic: loaded from disk (frozen, computes V targets).
        key, subkey = jax.random.split(key)
        old_critic_template = self._old_critic_network.init(
            subkey, obs=fake_obs, z=fake_zs
        )
        with open(critic_path, "rb") as f:
            old_critic_params = serialization.from_bytes(
                old_critic_template, f.read()
            )

        # New-architecture critic: fresh random init (trainable).
        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, obs=fake_obs, z=fake_zs)
        critic_opt_state = self._critic_optimizer.init(critic_params)

        if mean_path is not None:
            moving_mean = jnp.load(mean_path)
        else:
            moving_mean = jnp.array(0.0)
        if var_path is not None:
            moving_squared_diff = jnp.load(var_path)
        else:
            moving_squared_diff = jnp.array(1.0)

        return CriticFineTuneState(
            policy_params=policy_params,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            old_critic_params=old_critic_params,
            moving_mean=moving_mean,
            moving_squared_diff=moving_squared_diff,
            iteration_num=0,
        )

    # ------------------------------------------------------------------
    # Data generation: m rollout iterations, with probabilistic task
    # resampling. Returns transitions of shape (m, rollout_length, vec_env,
    # ...) plus the per-iteration final-state (obs, zs) used for bootstrapping.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def generate_rollout_data(
        self,
        starting_states,
        training_state: CriticFineTuneState,
        key: RNGKey,
    ) -> Tuple[Tuple[Any, RNGKey], PPOTransition, jax.Array, jax.Array]:

        policy_params = training_state.policy_params
        resample_prob = self.configs.resample_prob
        vec_env = self.configs.vec_env

        def data_step(carry, _):
            states, key = carry

            # Probabilistically resample a new task per environment, carrying
            # the (possibly new) task forward into the rollout.
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

        # First-pass targets: compute v_values and TD(lambda) returns with the
        # FROZEN old critic, and store the result in both td_lambda_returns and
        # gaes (in the new critic's normalized output space).
        td_lambda_returns_norm, weights = self._compute_normalized_targets_old(
            training_state.old_critic_params,
            transitions,
            final_obs,
            final_zs,
            training_state.moving_mean,
            jnp.sqrt(training_state.moving_squared_diff),
        )
        transitions = transitions.replace(
            td_lambda_returns=td_lambda_returns_norm,
            gaes=td_lambda_returns_norm,
            weights=weights,
        )

        return (final_states, key), transitions, final_obs, final_zs

    # ------------------------------------------------------------------
    # One fine-tuning epoch on the current (fixed) dataset:
    #   1. train the new critic by regression toward transitions.td_lambda_returns
    #      (one mini-batch pass),
    #   2. recompute v_values / TD(lambda) returns with the UPDATED critic,
    #   3. set transitions.td_lambda_returns to the latest result, and update
    #      transitions.gaes <- 0.5 * gaes + 0.5 * td_lambda_returns
    #      (a moving average of the targets, for future evaluation).
    # The updated transitions are returned so the caller can feed them back in
    # for the next bootstrapping iteration.
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: CriticFineTuneState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        key: RNGKey,
    ) -> Tuple[CriticFineTuneState, CriticFineTuneMetrics, PPOTransition]:

        moving_mean = training_state.moving_mean
        moving_std = jnp.sqrt(training_state.moving_squared_diff)

        # 1. Train one epoch by regression toward the current td_lambda_returns.
        #    Shuffle a copy into mini-batches; keep `transitions` intact for the
        #    recompute below.
        key, subkey = jax.random.split(key)
        train_transitions = transitions.shuffle(subkey)
        train_transitions = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (-1, self.configs.mini_batch_size, *x.shape[1:]),
            ),
            train_transitions,
        )

        final_carry, _ = self.train_critic(
            (training_state.critic_params, training_state.critic_opt_state, 0.0),
            train_transitions,
        )
        new_critic_params, new_critic_opt_state, critic_error = final_carry

        # 2. Recompute v_values / TD(lambda) returns with the UPDATED critic.
        td_lambda_returns_norm, _ = self._compute_normalized_targets_new(
            new_critic_params,
            transitions,
            final_obs,
            final_zs,
            moving_mean,
            moving_std,
        )

        alpha = jnp.exp(-0.2 * training_state.iteration_num)

        # 3. Refresh the targets and update the moving-average gaes.
        transitions = transitions.replace(
            # td_lambda_returns=td_lambda_returns_norm,
            td_lambda_returns=(0.5 * transitions.td_lambda_returns + 0.5 * td_lambda_returns_norm) * (1 -alpha) + alpha * transitions.gaes,
            # gaes=0.5 * transitions.gaes + 0.5 * td_lambda_returns_norm,
        )

        new_training_state = training_state.replace(
            critic_params=new_critic_params,
            critic_opt_state=new_critic_opt_state,
            iteration_num=training_state.iteration_num + 1,
        )

        metrics = CriticFineTuneMetrics(
            critic_error=critic_error * moving_std,
        )

        return new_training_state, metrics, transitions

    # ------------------------------------------------------------------
    # Target computation with the FROZEN old critic (used in
    # generate_rollout_data). The network is referenced via self so it stays
    # static at trace time.
    # ------------------------------------------------------------------
    def _compute_normalized_targets_old(
        self,
        critic_params: Params,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        moving_mean: jax.Array,
        moving_std: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:

        v_values = self._old_critic_network.apply(
            critic_params, transitions.obs, transitions.zs
        )  # (m, T, V, 1)
        final_v = self._old_critic_network.apply(
            critic_params, final_obs, final_zs
        )  # (m, V, 1)
        return self._td_lambda_from_values(
            v_values, final_v, transitions, moving_mean, moving_std
        )

    # ------------------------------------------------------------------
    # Target computation with the trainable new critic (used in state_update
    # to re-bootstrap targets after each training epoch).
    # ------------------------------------------------------------------
    def _compute_normalized_targets_new(
        self,
        critic_params: Params,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        moving_mean: jax.Array,
        moving_std: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:

        v_values = self._critic_network.apply(
            critic_params, transitions.obs, transitions.zs
        )  # (m, T, V, 1)
        final_v = self._critic_network.apply(
            critic_params, final_obs, final_zs
        )  # (m, V, 1)
        return self._td_lambda_from_values(
            v_values, final_v, transitions, moving_mean, moving_std
        )

    # ------------------------------------------------------------------
    # Network-free core: given already-computed (normalized) v_values and
    # final_v, denormalize to real-return space, compute per-iteration
    # TD(lambda) returns (vmap over m), and normalize them back into the
    # critic's output space. Returns (normalized td_lambda_returns, weights).
    # ------------------------------------------------------------------
    def _td_lambda_from_values(
        self,
        v_values: jax.Array,
        final_v: jax.Array,
        transitions: PPOTransition,
        moving_mean: jax.Array,
        moving_std: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:

        v_values = v_values * moving_std + moving_mean
        final_v = final_v * moving_std + moving_mean

        td_lambda_returns, weights = jax.vmap(
            self.calculate_td_lambda_returns,
            in_axes=(0, 0, 0, 0, 0),
        )(
            final_v,
            v_values,
            transitions.rewards,
            transitions.dones,
            transitions.truncations,
        )  # (m, T, V, 1)

        td_lambda_returns_norm = (td_lambda_returns - moving_mean) / (
            moving_std + 1e-6
        )
        return td_lambda_returns_norm, weights

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
    # rollout iteration, v_values of shape (rollout_length, vec_env, 1) and
    # final_v of shape (vec_env, 1). state_update vmaps this over m.
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
