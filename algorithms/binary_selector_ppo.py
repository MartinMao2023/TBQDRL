"""Binary selector PPO for the relocation ablation.

A binary "selection" policy is trained with PPO to decide, at each state,
whether to follow the frozen Gaussian PPO *teacher* or the frozen *relocated-
only* GMM *student*.  The teacher and the student share the same std.

The rollout gates between the two as follows:
  1. the GMM student's OWN mixture weights (``weight_logits``) are used to
     sample one component mean ``s_mean`` from the student (a single realized
     student action mean);
  2. the binary selector emits 2 logits (softmax over {teacher, student});
  3. one of ``[t_mean, s_mean]`` is selected by that softmax;
  4. the shared-std Gaussian noise is added to the chosen mean and clipped to
     [-1, 1].

The PPO loss only updates the selector's logits.  ``transitions.actions``
holds the two fixed per-option Gaussian log-probs of the sampled action
(teacher option, student option), so ``policy_loss_fn`` only needs to
recompute the selection logits.

Structure is adapted from ``algorithms/binary_ppo.py`` (categorical PPO
loss, critic loss, GAE, training scans) with two changes:

  * the rollout *gates* between the frozen teacher and the frozen GMM student
    via a binary softmax instead of stepping a binary-action env;
  * the critic is scaled by running reward stats (``moving_mean`` /
    ``moving_squared_diff``) ported from ``algorithms/gmm_ppo.py`` so a fresh
    critic can be warm-started from the teacher's mean/var.
"""

from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from data_struct import PPOTransition
from data_struct.states import GeneralizedState
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper


@dataclass
class SelectorPPOConfigs:
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    clip_ratio: float = 0.2
    entropy_gain: float = 0.01
    discount: float = 0.99
    td_lambda_discount: float = 0.95
    rollout_length: int = 64
    vec_env: int = 256
    mini_batch_size: int = 1024
    critic_epochs: int = 4
    policy_epochs: int = 4


class PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    policy_params: Params           # selector (softmax) params
    critic_params: Params           # fresh critic params

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    step_num: int
    iteration_num: int
    moving_mean: jnp.ndarray
    moving_squared_diff: jnp.ndarray


class RolloutMetrics(PyTreeNode):
    average_reward: float
    average_return: float
    average_lifespan: float


class TrainingMetrics(PyTreeNode):
    critic_error: float
    approx_kl: float
    clip_fraction: float


class AuxData(PyTreeNode):
    """Contains auxiliary information for monitoring"""

    rollout_data: RolloutMetrics
    training_data: TrainingMetrics


class SelectorPPO:
    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: nn.Module,
        critic_network: nn.Module,
        ppo_configs: SelectorPPOConfigs,
        teacher_network: nn.Module,
        teacher_params: Params,
        student_network: nn.Module,
        student_params: Params,
    ):

        self._env = env
        self.configs = ppo_configs

        self._policy_network = policy_network        # selector (softmax head)
        self._critic_network = critic_network        # fresh critic

        # Frozen base policies the selector gates between.
        self._teacher_network = teacher_network
        self._teacher_params = teacher_params
        self._student_network = student_network
        self._student_params = student_params

        self.mini_batch_num = (
            ppo_configs.vec_env * ppo_configs.rollout_length
        ) // ppo_configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2 / self.mini_batch_num)

        if ppo_configs.clip_ratio > 0:
            self._clip_log_ratio = jnp.log(1 + ppo_configs.clip_ratio)
        else:
            raise ValueError("invalid clip ratio")

        self._policy_optimizer = optax.adam(
            learning_rate=ppo_configs.policy_learning_rate,
        )
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate,
        )

        # Base-policy stds are frozen (taken from their loaded std_logits).
        # Teacher and GMM share the same std, so teacher_std is used as the
        # shared noise scale for the selected action mean.
        teacher_std_logits = teacher_params["params"]["std_logits"]
        self._teacher_std = jax.nn.sigmoid(teacher_std_logits)
        student_std_logits = student_params["params"]["std_logits"]
        self._student_std = jax.nn.sigmoid(student_std_logits)
        teacher_std = self._teacher_std

        @jax.jit
        def rollout_fn(
            policy_params: Params,
            starting_states: GeneralizedState,
            keys: RNGKey,
        ) -> Tuple[GeneralizedState, PPOTransition]:

            def play_step_fn(carry):
                state, key = carry
                obs, z = env.get_obs(state)

                # --- candidate action means (all FROZEN) ---
                t_mean, _ = teacher_network.apply(teacher_params, obs, z)              # (d,)
                s_means, weight_logits, _ = student_network.apply(student_params, obs, z)  # (C, d), (C,)

                # first sample one GMM component mean using the student's OWN
                # mixture weights (weight_logits); this realizes the "student"
                # action mean the selector will gate against the teacher.
                key, gkey = jax.random.split(key)
                s_mean = jax.random.choice(
                    gkey, a=s_means, p=nn.softmax(weight_logits)
                )  # (d,)

                # --- binary selector: option 0 = teacher, option 1 = GMM-sampled student ---
                selection_logits = policy_network.apply(policy_params, obs, z)  # (2,)
                means = jnp.stack([t_mean, s_mean], axis=0)                     # (2, d)

                # select an action mean by the binary softmax, then add shared-std noise
                key, ckey = jax.random.split(key)
                selected_mean = jax.random.choice(
                    ckey, a=means, p=nn.softmax(selection_logits)
                )  # (d,)

                # selected_mean = s_mean

                key, nkey = jax.random.split(key)
                action = jnp.clip(
                    selected_mean + teacher_std * jax.random.normal(nkey, selected_mean.shape),
                    -1.0, 1.0,
                )
                action = jnp.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

                # per-option log p(a) of the sampled action:
                #   option 0 (teacher, PPO Gaussian): single Gaussian log N(a | t_mean, std)
                log_p_teacher = -0.5 * jnp.sum(
                    jnp.square(action - t_mean) / (teacher_std**2 + 1e-6)
                )  # scalar
                #   option 1 (student, GMM): full GMM log p(a) over ALL components
                log_gauss_components = -0.5 * jnp.sum(
                    jnp.square(action - s_means) / (teacher_std**2 + 1e-6),
                    axis=-1,
                )  # (C,)
                log_p_student = nn.logsumexp(weight_logits + log_gauss_components) \
                    - nn.logsumexp(weight_logits)  # scalar
                per_option_log = jnp.stack([log_p_teacher, log_p_student])  # (2,)

                # mixture log p(a) = logsumexp_i [w_i + log p_i(a)] - logsumexp_i [w_i]
                log_likelihood = nn.logsumexp(selection_logits + per_option_log) \
                    - nn.logsumexp(selection_logits)  # scalar

                state, transition_info = env.step(state, action)

                transition = PPOTransition(
                    obs=obs,
                    actions=per_option_log,  # (2,) fixed per-option log-probs (teacher Gaussian, student GMM)
                    zs=z,
                    log_likelihood=jnp.array([log_likelihood]),
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
                length=ppo_configs.rollout_length,
            )
            return final_states, transitions

        self._rollout_fn = rollout_fn

        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:

            estimated_v = critic_network.apply(
                critic_params, transitions.obs, transitions.zs
            )
            weights = 1 / (1 + jnp.square(transitions.weights))
            loss = jnp.average(
                jnp.square(estimated_v - transitions.td_lambda_returns),
                weights=weights,
            )
            return loss, jnp.sqrt(loss)

        self._critic_loss_fn = critic_loss_fn

        def policy_loss_fn(
            policy_params: Params,
            transitions: PPOTransition,
        ) -> float:

            selection_logits = policy_network.apply(
                policy_params, transitions.obs, transitions.zs
            )  # batch x (C+1)
            # transitions.actions holds the (fixed) per-component Gaussian
            # log-probs of the sampled action; only the selection logits are
            # recomputed here (the component means / std are frozen).
            new_log_likelihood = nn.logsumexp(
                selection_logits + transitions.actions, axis=-1, keepdims=True
            ) - nn.logsumexp(selection_logits, axis=-1, keepdims=True)  # batch x 1

            # selection (categorical) entropy
            entropy = nn.logsumexp(selection_logits, axis=-1, keepdims=True) \
                - jnp.sum(
                    nn.softmax(selection_logits, axis=-1) * selection_logits,
                    axis=-1, keepdims=True,
                )  # batch x 1

            log_ratio = new_log_likelihood - transitions.log_likelihood
            ratio = jnp.exp(log_ratio)
            gaes = transitions.gaes
            loss_cond = jax.lax.stop_gradient(
                log_ratio * gaes <= self._clip_log_ratio * jnp.abs(gaes)
            )
            clip_fraction = 1 - jnp.mean(loss_cond)
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

            loss = jnp.mean(
                jnp.where(loss_cond, -gaes * ratio, 0.0)
                - ppo_configs.entropy_gain * entropy
            )
            return loss, (approx_kl, clip_fraction)

        self._policy_loss_fn = policy_loss_fn

    def init(
        self,
        key: RNGKey,
    ) -> PPOTrainingState:

        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, obs=fake_obs, z=fake_zs)
        critic_opt_state = self._critic_optimizer.init(critic_params)

        training_state = PPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            step_num=0,
            iteration_num=0,
            moving_mean=jnp.zeros((1,)),
            moving_squared_diff=jnp.ones((1,)),
        )
        return training_state

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: PPOTrainingState,
        transitions: PPOTransition,
    ) -> Tuple[PPOTrainingState, TrainingMetrics]:

        (critic_params, critic_opt_state, final_critic_error), _ = jax.lax.scan(
            lambda x, _: partial(self.train_critic, transitions=transitions)(x),
            (training_state.critic_params, training_state.critic_opt_state, 0.0),
            length=self.configs.critic_epochs,
        )

        (policy_params, policy_opt_state, final_approx_kl, final_clip_fraction), _ = jax.lax.scan(
            lambda x, _: partial(
                self.train_policy,
                transitions=transitions,
            )(x),
            (training_state.policy_params, training_state.policy_opt_state, 0.0, 0.0),
            length=self.configs.policy_epochs,
        )

        step_num = training_state.step_num + 1

        new_training_state = PPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            step_num=step_num,
            iteration_num=training_state.iteration_num,
            moving_mean=training_state.moving_mean,
            moving_squared_diff=training_state.moving_squared_diff,
        )
        training_data = TrainingMetrics(
            critic_error=final_critic_error,
            approx_kl=final_approx_kl,
            clip_fraction=final_clip_fraction,
        )
        return new_training_state, training_data

    @partial(jax.jit, static_argnames=("self",))
    def train_policy(
        self,
        carry: Tuple[Params, optax.OptState, float, float],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float, float], Any]:

        def scan_train_policy(carry, transition_data):
            (
                current_policy_params,
                current_policy_opt_state,
                current_approx_kl,
                current_clip_fraction,
            ) = carry

            policy_gradient, (approx_kl, clip_fraction) = jax.grad(
                self._policy_loss_fn, has_aux=True
            )(current_policy_params, transition_data)

            new_approx_kl = approx_kl * (1 - self.ema_alpha) + \
                self.ema_alpha * current_approx_kl
            new_clip_fraction = clip_fraction * (1 - self.ema_alpha) + \
                self.ema_alpha * current_clip_fraction

            policy_updates, new_policy_opt_state = self._policy_optimizer.update(
                policy_gradient, current_policy_opt_state)
            new_policy_params = optax.apply_updates(current_policy_params, policy_updates)

            new_carry = (
                new_policy_params,
                new_policy_opt_state,
                new_approx_kl,
                new_clip_fraction,
            )
            return new_carry

        def cond_scan_train_policy(carry, transition_data):
            approx_kl = carry[-2]
            new_carry = jax.lax.cond(
                approx_kl > 0.0125,
                lambda x: x,
                lambda x: scan_train_policy(x, transition_data),
                carry,
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            cond_scan_train_policy,
            carry,
            transitions,
        )
        return final_carry, None

    @partial(jax.jit, static_argnames=("self",))
    def train_critic(
        self,
        carry: Tuple[Params, optax.OptState, float],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:

        def scan_train_critic(carry, transition_data):
            (
                current_critic_params,
                current_critic_opt_state,
                current_critic_error,
            ) = carry

            critic_gradient, critic_error = jax.grad(
                self._critic_loss_fn, has_aux=True
            )(current_critic_params, transition_data)

            new_critic_error = critic_error * (1 - self.ema_alpha) + \
                self.ema_alpha * current_critic_error

            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_gradient, current_critic_opt_state)
            new_critic_params = optax.apply_updates(current_critic_params, critic_updates)

            return (new_critic_params, new_critic_opt_state, new_critic_error), None

        final_carry, _ = jax.lax.scan(
            scan_train_critic,
            carry,
            transitions,
        )
        return final_carry, None

    @partial(jax.jit, static_argnames=("self",))
    def calculate_v(
        self,
        critic_params: Params,
        transitions: PPOTransition,
    ) -> jnp.ndarray:

        def scan_calculate_v(transition: PPOTransition):
            v_value = self._critic_network.apply(
                critic_params, transition.obs, transition.zs
            )
            return None, v_value

        _, v_values = jax.lax.scan(
            lambda _, x: jax.vmap(scan_calculate_v)(x),
            None,
            transitions,
        )
        return v_values

    @partial(jax.jit, static_argnames=("self",))
    def _process_gaes(self, gaes: jnp.ndarray) -> jnp.ndarray:
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        mask = jnp.abs(gaes - gae_mean) < 3 * gae_std
        corrected_mean = jnp.mean(gaes, where=mask)
        clipped_values = jnp.clip(gaes, corrected_mean - 3 * gae_std, corrected_mean + 3 * gae_std)
        corrected_std = jnp.std(clipped_values, ddof=1)
        gaes = jnp.clip(gaes, corrected_mean - 5 * corrected_std, corrected_mean + 5 * corrected_std)
        offset = jnp.clip(-jnp.mean(gaes), min=0.0)
        return (gaes + offset) / (corrected_std + 1e-6)

    @partial(jax.jit, static_argnames=("self",))
    def calculate_td_lambda_returns(
        self,
        final_v_value: jnp.ndarray,
        v_values: jnp.ndarray,
        rewards: jnp.ndarray,
        termination: jnp.ndarray,
        truncation: jnp.ndarray,
    ) -> jnp.ndarray:

        discount = self.configs.discount
        td_lambda_discount = self.configs.td_lambda_discount

        def scan_calculate_td_lambda(carry, data):
            (last_td_lambda_value, last_value, last_weight) = carry
            reward, v_value, done, truncate = data
            current_td_lambda_value = reward + (1 - done) * discount * (
                (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
            )
            current_td_lambda_value = jnp.where(truncate, v_value, current_td_lambda_value)
            weight = jnp.where(
                truncate > 0.5,
                1.0,
                (1 - done) * discount * (1 + (last_weight - 1) * td_lambda_discount),
            )
            return (current_td_lambda_value, v_value, weight), (current_td_lambda_value, weight)

        _, (td_lambda_values, weights) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (final_v_value, final_v_value, jnp.ones_like(final_v_value)),
            (rewards, v_values, termination, truncation),
            reverse=True,
        )
        return td_lambda_values, weights

    @partial(jax.jit, static_argnames=("self",))
    def evaluate_rollout(
        self,
        final_v: jax.Array,
        transitions: PPOTransition,
    ) -> RolloutMetrics:

        discount = self.configs.discount
        average_reward = jnp.mean(transitions.rewards)

        def scan_evaluation(carry, data: PPOTransition):
            (v_value, lifespan) = carry
            new_v_value = data.rewards + (1 - data.dones) * discount * v_value
            new_v_value = jnp.where(data.truncations, data.td_lambda_returns, new_v_value)
            new_lifespan = 1 + (1 - data.dones) * lifespan
            new_carry = (new_v_value, new_lifespan)
            return new_carry, new_carry

        (initial_v_value, initial_lifespan), (v_values, lifespans) = jax.lax.scan(
            scan_evaluation,
            (final_v, jnp.zeros_like(final_v)),
            transitions,
            reverse=True,
        )
        rollout_data = RolloutMetrics(
            average_reward=average_reward,
            average_return=jnp.mean(v_values),
            average_lifespan=jnp.mean(initial_lifespan),
        )
        return rollout_data

    @partial(jax.jit, static_argnames=("self",))
    def train(
        self,
        starting_states: GeneralizedState,
        training_state: PPOTrainingState,
        key: RNGKey,
    ) -> Tuple[Tuple[GeneralizedState, PPOTrainingState, RNGKey], AuxData]:

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self.configs.vec_env)
        final_states, transitions = self._rollout_fn(
            training_state.policy_params,
            starting_states,
            subkeys,
        )
        final_obs, final_zs = self._env.get_obs(final_states)

        final_v = self._critic_network.apply(training_state.critic_params, final_obs, final_zs)
        final_v = final_v * jnp.sqrt(training_state.moving_squared_diff) + training_state.moving_mean
        v_values = self.calculate_v(training_state.critic_params, transitions)
        v_values = v_values * jnp.sqrt(training_state.moving_squared_diff) + training_state.moving_mean

        td_lambda_returns, weights = self.calculate_td_lambda_returns(
            final_v,
            v_values,
            transitions.rewards,
            transitions.dones,
            transitions.truncations,
        )

        iteration_num = training_state.iteration_num + 1
        alpha = 1 / iteration_num
        new_mean = training_state.moving_mean * (1 - alpha) + jnp.mean(td_lambda_returns) * alpha
        new_squared_diff = training_state.moving_squared_diff * (1 - alpha) + \
            jnp.mean(jnp.square(td_lambda_returns - new_mean)) * alpha

        gaes = self._process_gaes(td_lambda_returns - v_values)

        transitions = transitions.replace(
            td_lambda_returns=td_lambda_returns,
            gaes=gaes,
            weights=weights,
        )
        rollout_data = self.evaluate_rollout(final_v, transitions)

        transitions = transitions.replace(
            td_lambda_returns=(td_lambda_returns - new_mean) / (1e-6 + jnp.sqrt(new_squared_diff)),
        )

        key, subkey = jax.random.split(key)
        transitions = transitions.shuffle(subkey)
        transitions = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (
                    -1,
                    self.configs.mini_batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions,
        )

        training_state = training_state.replace(
            iteration_num=iteration_num,
            moving_mean=new_mean,
            moving_squared_diff=new_squared_diff,
        )

        new_training_state, training_data = self.state_update(training_state, transitions)
        aux_data = AuxData(rollout_data=rollout_data, training_data=training_data)
        return (final_states, new_training_state, key), aux_data
