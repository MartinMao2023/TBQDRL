"""Fine-tune a value critic on packed ``PPOTransition`` data.

Critic learning follows ``algorithms/ppo.py``: a few epochs of weighted MSE
on minibatches, with TD(lambda) targets clipped by a running ``moving_mse``.
``moving_mean`` and ``moving_std`` are kept fixed (no return-normalization
update).
"""

from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import optax
from flax.struct import PyTreeNode, dataclass
from jax import numpy as jnp

from custom_types import Params, RNGKey
from data_struct import PPOTransition


zero = jnp.float32(0.0)
one = jnp.float32(1.0)


@dataclass
class CriticFineTuningConfigs:
    critic_learning_rate: float = 5e-4
    critic_epochs: int = 2
    mini_batch_size: int = 4096
    discount: float = 0.99
    gae_lambda: float = 0.95
    trajectory_num_per_iter: int = 4096
    target_learning_rate: float = 0.25


class CriticBatch(PyTreeNode):
    obs: jax.Array
    zs: jax.Array
    target_values: jax.Array
    weights: jax.Array


class CriticFineTuningState(PyTreeNode):
    critic_params: Params
    critic_opt_state: optax.OptState
    moving_mean: jax.Array
    moving_std: jax.Array
    moving_mse: jax.Array
    iteration_num: jax.Array


class CriticFineTuningMetrics(PyTreeNode):
    critic_rmse: jax.Array
    moving_mse: jax.Array


class CriticFineTuning:
    def __init__(
        self,
        critic_network: nn.Module,
        configs: CriticFineTuningConfigs,
    ):
        self._critic_network = critic_network
        self.configs = configs
        self._critic_optimizer = optax.adam(
            learning_rate=configs.critic_learning_rate
        )

    def init(
        self,
        critic_params: Params,
        moving_mean: jax.Array,
        moving_std: jax.Array,
        moving_mse: jax.Array,
    ) -> CriticFineTuningState:
        return CriticFineTuningState(
            critic_params=critic_params,
            critic_opt_state=self._critic_optimizer.init(critic_params),
            moving_mean=jnp.asarray(moving_mean, dtype=jnp.float32),
            moving_std=jnp.asarray(moving_std, dtype=jnp.float32),
            moving_mse=jnp.asarray(moving_mse, dtype=jnp.float32),
            iteration_num=jnp.asarray(0, dtype=jnp.int32),
        )

    def _shuffle_data(
        self,
        data: CriticBatch,
        shuffled_indices: jax.Array,
    ) -> CriticBatch:
        mini_batch_size = self.configs.mini_batch_size
        n_mb = shuffled_indices.shape[0]

        def shuffle_leaf(value):
            return value[shuffled_indices].reshape(
                n_mb, mini_batch_size, *value.shape[1:]
            )

        return jax.tree.map(shuffle_leaf, data)

    def _update_critic(
        self,
        critic_params: Params,
        critic_opt_state: optax.OptState,
        training_data: CriticBatch,
        ema_alpha: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        def loss_fn(params: Params, mini_batch: CriticBatch):
            predictions = self._critic_network.apply(
                params, mini_batch.obs, mini_batch.zs
            )
            squared_errors = jnp.square(predictions - mini_batch.target_values)
            loss = jnp.average(squared_errors, weights=mini_batch.weights)
            return loss, jnp.sqrt(loss)

        def update_minibatch(carry, mini_batch):
            params, opt_state, running_rmse = carry
            gradients, rmse = jax.grad(loss_fn, has_aux=True)(params, mini_batch)
            running_rmse = (one - ema_alpha) * rmse + ema_alpha * running_rmse
            updates, opt_state = self._critic_optimizer.update(
                gradients, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state, running_rmse), None

        (critic_params, critic_opt_state, critic_rmse), _ = jax.lax.scan(
            lambda x, _: jax.lax.scan(update_minibatch, x, training_data),
            (critic_params, critic_opt_state, zero),
            length=self.configs.critic_epochs,
        )
        return critic_params, critic_opt_state, critic_rmse

    def calculate_v(
        self,
        training_state: CriticFineTuningState,
        obs: jax.Array,
        zs: jax.Array,
    ) -> jax.Array:
        v_values = self._critic_network.apply(
            training_state.critic_params, obs, zs
        )
        return v_values * training_state.moving_std + training_state.moving_mean

    def _calculate_td_lambda_returns(
        self,
        all_v_values: jax.Array,
        transitions: PPOTransition,
    ) -> Tuple[jax.Array, jax.Array]:
        actual_discount = jnp.where(
            transitions.dones > 0.5, zero, self.configs.discount
        )
        ks = self.configs.gae_lambda * actual_discount
        bs = actual_discount * (
            (1 - self.configs.gae_lambda) * all_v_values[1:, ...]
        )
        init_target = all_v_values[-1]

        def scan_calculate_td_lambda(carry, data):
            last_target, last_bootstrap_portion = carry
            reward, truncation, k, b, v_value = data
            target = jnp.where(
                truncation > 0.5, v_value, reward + k * last_target + b
            )
            bootstrap_portion = jnp.where(
                truncation > 0.5, one, last_bootstrap_portion * k
            )
            new_carry = (target, bootstrap_portion)
            return new_carry, new_carry

        _, (critic_targets, bootstrap_portions) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (init_target, jnp.ones_like(init_target)),
            (
                transitions.rewards,
                transitions.truncations,
                ks,
                bs,
                all_v_values[:-1],
            ),
            reverse=True,
        )
        critic_target_weights = one / (one + jnp.square(bootstrap_portions))
        return critic_targets, critic_target_weights

    @partial(jax.jit, static_argnames=("self",))
    def _calculate_td_lambda_returns_chunked(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
    ) -> jax.Array:
        num_traj = transitions.obs.shape[0]
        rollout_length = transitions.obs.shape[1]
        trajectory_num = self.configs.trajectory_num_per_iter
        chunk_num = num_traj // trajectory_num

        def to_chunks(x):
            x = x.reshape(
                chunk_num, trajectory_num, rollout_length, x.shape[-1]
            )
            return jnp.swapaxes(x, 1, 2)

        chunked_transitions = jax.tree.map(to_chunks, transitions)
        chunked_final_obs = final_obs.reshape(
            chunk_num, trajectory_num, final_obs.shape[-1]
        )
        chunked_final_zs = final_zs.reshape(
            chunk_num, trajectory_num, final_zs.shape[-1]
        )

        def calculate_chunk_returns(_, batch):
            chunk, chunk_final_obs, chunk_final_zs = batch
            all_obs = jnp.concatenate(
                [chunk.obs, chunk_final_obs[None, ...]], axis=0
            )
            all_zs = jnp.concatenate(
                [chunk.zs, chunk_final_zs[None, ...]], axis=0
            )
            all_v_values = self.calculate_v(training_state, all_obs, all_zs)
            td_lambda_returns, _ = self._calculate_td_lambda_returns(
                all_v_values, chunk
            )
            return None, td_lambda_returns

        _, chunked_returns = jax.lax.scan(
            calculate_chunk_returns,
            None,
            (chunked_transitions, chunked_final_obs, chunked_final_zs),
        )
        chunked_returns = jnp.swapaxes(chunked_returns, 1, 2)
        return chunked_returns.reshape(
            num_traj, rollout_length, chunked_returns.shape[-1]
        )

    @partial(jax.jit, static_argnames=("self",))
    def _calculate_gaes_chunked(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
    ) -> jax.Array:
        num_traj = transitions.obs.shape[0]
        rollout_length = transitions.obs.shape[1]
        trajectory_num = self.configs.trajectory_num_per_iter
        chunk_num = num_traj // trajectory_num

        def to_chunks(x):
            x = x.reshape(
                chunk_num, trajectory_num, rollout_length, x.shape[-1]
            )
            return jnp.swapaxes(x, 1, 2)

        chunked_transitions = jax.tree.map(to_chunks, transitions)
        chunked_final_obs = final_obs.reshape(
            chunk_num, trajectory_num, final_obs.shape[-1]
        )
        chunked_final_zs = final_zs.reshape(
            chunk_num, trajectory_num, final_zs.shape[-1]
        )

        def calculate_chunk_gaes(_, batch):
            chunk, chunk_final_obs, chunk_final_zs = batch
            all_obs = jnp.concatenate(
                [chunk.obs, chunk_final_obs[None, ...]], axis=0
            )
            all_zs = jnp.concatenate(
                [chunk.zs, chunk_final_zs[None, ...]], axis=0
            )
            all_v_values = self.calculate_v(training_state, all_obs, all_zs)
            td_lambda_returns, _ = self._calculate_td_lambda_returns(
                all_v_values, chunk
            )
            return None, td_lambda_returns - all_v_values[:-1]

        _, chunked_gaes = jax.lax.scan(
            calculate_chunk_gaes,
            None,
            (chunked_transitions, chunked_final_obs, chunked_final_zs),
        )
        chunked_gaes = jnp.swapaxes(chunked_gaes, 1, 2)
        return chunked_gaes.reshape(
            num_traj, rollout_length, chunked_gaes.shape[-1]
        )

    def calculate_td_lambda_returns(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
    ) -> PPOTransition:
        """Replace TD(lambda) returns using the final critic.

        ``transitions`` have shape ``(trajectory_num, rollout_length, d)``.
        Evaluation is split into ``trajectory_num_per_iter`` trajectories at
        a time to bound the critic's peak memory use.
        """
        trajectory_num = self.configs.trajectory_num_per_iter

        if trajectory_num <= 0:
            raise ValueError("trajectory_num_per_iter must be positive")

        trajectory_final_obs = final_obs.reshape(-1, final_obs.shape[-1])
        trajectory_final_zs = final_zs.reshape(-1, final_zs.shape[-1])
        num_traj = transitions.obs.shape[0]

        if trajectory_final_obs.shape[0] != num_traj:
            raise ValueError(
                f"final_obs contains {trajectory_final_obs.shape[0]} "
                f"trajectories, expected {num_traj}"
            )
        if trajectory_final_zs.shape[0] != num_traj:
            raise ValueError(
                f"final_zs contains {trajectory_final_zs.shape[0]} "
                f"trajectories, expected {num_traj}"
            )
        if num_traj % trajectory_num != 0:
            raise ValueError(
                f"trajectory count {num_traj} is not divisible by "
                f"trajectory_num_per_iter {trajectory_num}"
            )

        td_lambda_returns = self._calculate_td_lambda_returns_chunked(
            training_state,
            transitions,
            trajectory_final_obs,
            trajectory_final_zs,
        )
        return transitions.replace(td_lambda_returns=td_lambda_returns)

    def calculate_gaes(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
    ) -> PPOTransition:
        """Replace GAEs using TD(lambda) returns from the final critic.

        ``transitions`` have shape ``(trajectory_num, rollout_length, d)``.
        Evaluation is split into ``trajectory_num_per_iter`` trajectories at
        a time to bound the critic's peak memory use.
        """
        trajectory_num = self.configs.trajectory_num_per_iter

        if trajectory_num <= 0:
            raise ValueError("trajectory_num_per_iter must be positive")

        trajectory_final_obs = final_obs.reshape(-1, final_obs.shape[-1])
        trajectory_final_zs = final_zs.reshape(-1, final_zs.shape[-1])
        num_traj = transitions.obs.shape[0]

        if trajectory_final_obs.shape[0] != num_traj:
            raise ValueError(
                f"final_obs contains {trajectory_final_obs.shape[0]} "
                f"trajectories, expected {num_traj}"
            )
        if trajectory_final_zs.shape[0] != num_traj:
            raise ValueError(
                f"final_zs contains {trajectory_final_zs.shape[0]} "
                f"trajectories, expected {num_traj}"
            )
        if num_traj % trajectory_num != 0:
            raise ValueError(
                f"trajectory count {num_traj} is not divisible by "
                f"trajectory_num_per_iter {trajectory_num}"
            )

        gaes = self._calculate_gaes_chunked(
            training_state,
            transitions,
            trajectory_final_obs,
            trajectory_final_zs,
        )
        return transitions.replace(gaes=gaes)

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        key: RNGKey,
    ) -> Tuple[CriticFineTuningState, CriticFineTuningMetrics]:
        moving_mean = training_state.moving_mean
        moving_std = training_state.moving_std

        all_obs = jnp.concatenate(
            [transitions.obs, final_obs[None, ...]], axis=0
        )
        all_zs = jnp.concatenate(
            [transitions.zs, final_zs[None, ...]], axis=0
        )
        all_v_values = self.calculate_v(training_state, all_obs, all_zs)
        v_values = all_v_values[:-1]
        critic_targets, critic_target_weights = (
            self._calculate_td_lambda_returns(all_v_values, transitions)
        )

        iteration_num = training_state.iteration_num + 1
        mse_learning_rate = 0.05
        moving_mse = (
            (one - mse_learning_rate) * training_state.moving_mse
            + mse_learning_rate * jnp.mean(jnp.square(critic_targets - v_values))
        )

        target_alpha = self.configs.target_learning_rate
        clipped_critic_targets = jnp.clip(
            critic_targets,
            v_values - 3.0 * jnp.sqrt(moving_mse),
            v_values + 3.0 * jnp.sqrt(moving_mse),
        ) * target_alpha + (1 - target_alpha) * v_values
        normalized_critic_targets = (clipped_critic_targets - moving_mean) / (
            1e-8 + moving_std
        )

        n = transitions.obs.shape[0] * transitions.obs.shape[1]
        mini_batch_size = self.configs.mini_batch_size
        n_mb = n // mini_batch_size
        n_used = n_mb * mini_batch_size
        ema_alpha = jnp.exp(jnp.array(-2.0 / n_mb))

        def flatten(x):
            return x.reshape(n, *x.shape[2:])

        shuffled_indices = jax.random.permutation(key, n)[:n_used].reshape(
            n_mb, mini_batch_size
        )
        training_data = self._shuffle_data(
            CriticBatch(
                obs=flatten(transitions.obs),
                zs=flatten(transitions.zs),
                target_values=flatten(normalized_critic_targets),
                weights=flatten(critic_target_weights),
            ),
            shuffled_indices,
        )

        critic_params, critic_opt_state, critic_rmse = self._update_critic(
            training_state.critic_params,
            training_state.critic_opt_state,
            training_data,
            ema_alpha,
        )

        training_state = training_state.replace(
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            moving_mse=moving_mse,
            iteration_num=iteration_num,
        )
        metrics = CriticFineTuningMetrics(
            critic_rmse=critic_rmse * moving_std,
            moving_mse=moving_mse,
        )
        return training_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def _scan_state_updates(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        key: RNGKey,
    ) -> Tuple[Tuple[CriticFineTuningState, RNGKey], CriticFineTuningMetrics]:
        num_traj = transitions.obs.shape[0]
        T = transitions.obs.shape[1]
        B = self.configs.trajectory_num_per_iter
        k = num_traj // B

        key, perm_key, scan_key = jax.random.split(key, 3)
        perm = jax.random.permutation(perm_key, num_traj)
        transitions = jax.tree.map(lambda x: x[perm], transitions)
        final_obs = final_obs[perm]
        final_zs = final_zs[perm]

        def to_chunks(x):
            y = x.reshape(k, B, T, x.shape[-1])
            return jnp.transpose(y, (0, 2, 1, 3))  # (k, T, B, d)

        transitions = jax.tree.map(to_chunks, transitions)
        final_obs = final_obs.reshape(k, B, final_obs.shape[-1])
        final_zs = final_zs.reshape(k, B, final_zs.shape[-1])

        def scan_step(carry, batch):
            state, key = carry
            trans, fobs, fzs = batch
            key, subkey = jax.random.split(key)
            state, metrics = self.state_update(state, trans, fobs, fzs, subkey)
            return (state, key), metrics

        return jax.lax.scan(
            scan_step,
            (training_state, scan_key),
            (transitions, final_obs, final_zs),
        )

    def train(
        self,
        training_state: CriticFineTuningState,
        transitions: PPOTransition,
        final_obs: jax.Array,
        final_zs: jax.Array,
        key: RNGKey,
    ) -> Tuple[CriticFineTuningState, CriticFineTuningMetrics]:
        """Shuffle trajectories into chunks and scan ``state_update``.

        ``transitions`` are ``(..., rollout_length, vec_env, d)``;
        ``final_obs`` / ``final_zs`` are ``(..., vec_env, d)``.
        """
        rollout_length = transitions.obs.shape[-3]
        batch_per_iter = self.configs.trajectory_num_per_iter

        def to_traj(x):
            x = jnp.swapaxes(x, -3, -2)  # (..., vec_env, rollout_length, d)
            return x.reshape(-1, rollout_length, x.shape[-1])

        transitions = jax.tree.map(to_traj, transitions)
        final_obs = final_obs.reshape(-1, final_obs.shape[-1])
        final_zs = final_zs.reshape(-1, final_zs.shape[-1])

        num_traj = transitions.obs.shape[0]
        if num_traj % batch_per_iter != 0:
            raise ValueError(
                f"trajectory count {num_traj} is not divisible by "
                f"trajectory_num_per_iter {batch_per_iter}"
            )

        (training_state, _), metrics = self._scan_state_updates(
            training_state, transitions, final_obs, final_zs, key
        )

        return training_state, metrics
