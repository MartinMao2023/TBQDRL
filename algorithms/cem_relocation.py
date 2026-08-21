"""Trajectory relocation optimization (Stage 3).

Given a critic and a set of rollout trajectories (collected under some
preference), search for the (preference, start, end) that maximizes the
relocation advantage of each trajectory, then relabel the selected segments
with their new preference and return them as a ``PPOTransition``.

The search is a random-search + local-refinement scheme:
  * each iteration draws ``n_global`` global candidates and ``n_refine``
    candidates around the current best (or all global if the best so far is
    below a discard threshold);
  * the best candidate is updated every iteration;
  * after the search, trajectories containing a done/truncation are forced to
    a discard score, only segments with a positive best score are sliced out
    and concatenated, and the result is tiled back to the input transition
    count.

This module is organised as a ``Relocator`` class. The per-step GAE of the
selected trunks is computed by :meth:`Relocator.calculate_td_lambda_return`
(currently a placeholder); ``reorganize`` currently leaves GAE / weights as
dummy and only slices, concatenates, and marks truncations.
"""

from data_struct.relocation_transitions import MORelocationTransition
from data_struct import PPOTransition
from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
from tools import calculate_coefs_for_trajectory
import flax.linen as nn


class Relocator:
    """Find optimal (preference, start, end) per trajectory, compute the GAE of
    the selected trunks, and re-organize the selected transitions into a
    ``PPOTransition``.

    Workflow:
        1. ``optimize_all``   -> (best_p, best_s, best_e, best_score) per traj
        2. ``calculate_td_lambda_return`` -> per-step GAE of the trunks
        3. ``reorganize``     -> index table / shuffle / concat slices -> PPOTransition
    """

    def __init__(
        self,
        critic_network: nn.module,
        critic_params,
        moving_mean: float,
        moving_std: float,
        config: dict,
    ):
        self.critic_network = critic_network
        self.critic_params = critic_params
        self.moving_mean = moving_mean
        self.moving_std = moving_std

        self.threshold = config.get("threshold", -1000)
        self.num_iters = config.get("num_iters", 256)
        self.n_global = config.get("num_global", 4)
        self.n_refine = config.get("num_refine", 4)
        self.pref_std = config.get("pref_std", 0.1)
        self.idx_radius = config.get("idx_radius", 8)
        self.s_min = config.get("s_min", 0)
        self.s_max = config.get("s_max", 48)
        self.min_len = config.get("min_len", 16)
        self.e_max = config.get("e_max", 64)
        self.chunk_size = config.get("chunk_size", 1024)
        self.n_cand = self.n_global + self.n_refine
        self.trunk_penalty = 10

    # -------------------------------------------------- advantage evaluator
    def _eval_significant_advantage(
        self,
        preference: jax.Array,
        start: int,
        end: int,
        mo_rewards: jax.Array,
        obs: jax.Array,
        last_actions: jax.Array,
    ) -> float:
        total_length = mo_rewards.shape[0]
        reward_coefs, v_coefs = calculate_coefs_for_trajectory(
            total_length, start, end, 4, 0.99, 0.95
        )
        zs = jnp.concatenate(
            [last_actions, jnp.tile(preference, (total_length + 1, 1))], axis=-1
        )
        values = self.critic_network.apply(self.critic_params, obs, zs) \
            * self.moving_std + self.moving_mean
        rewards = jnp.sum(mo_rewards * preference, axis=-1, keepdims=True)
        trunk_length = end - start
        length_penalty = self.trunk_penalty * 2.578434 * (
            trunk_length * 0.125 - 1.578434 * (1 - jnp.exp(-0.06134363 * trunk_length))
        )
        advantage = jnp.sum(rewards * reward_coefs) + jnp.sum(values * v_coefs) \
            - length_penalty
        return advantage

    # ----------------------------------------------------- candidate samplers
    @staticmethod
    def _project(x):
        """Map an unconstrained 5-vector onto the feasible preference patch:
        last three non-negative (abs), then renormalize to unit length."""
        x = jnp.concatenate([x[..., :2], jnp.abs(x[..., 2:])], axis=-1)
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

    def _sample_global(self, k, n):
        kp, ks, ke = jax.random.split(k, 3)
        p = self._project(jax.random.normal(kp, (n, 5)))
        s = jax.random.randint(ks, (n,), self.s_min, self.s_max + 1)
        e = jax.random.randint(ke, (n,), s + self.min_len, self.e_max + 1)
        return p, s, e

    def _sample_refine(self, k, best_p, best_s, best_e, n):
        kp, ks, ke = jax.random.split(k, 3)
        p = self._project(best_p + self.pref_std * jax.random.normal(kp, (n, 5)))
        s_lo = jnp.maximum(best_s - self.idx_radius, self.s_min)
        s_hi = jnp.minimum(best_s + self.idx_radius, self.s_max)
        s = jax.random.randint(ks, (n,), s_lo, s_hi + 1)
        e_lo = jnp.maximum(best_e - self.idx_radius, s + self.min_len)
        e_hi = jnp.minimum(best_e + self.idx_radius, self.e_max)
        e_hi = jnp.maximum(e_hi, e_lo)
        e = jax.random.randint(ke, (n,), e_lo, e_hi + 1)
        return p, s, e

    # ------------------------------------------------ per-trajectory search
    def _optimize_one(self, all_obs_t, all_la_t, mo_rewards_t, key):
        """num_iters random-search + refine for one trajectory.

        Returns (best_p, best_s, best_e, best_score).
        """
        def step(carry, _):
            best_p, best_s, best_e, best_score, key = carry
            key, k_gen, k_next = jax.random.split(key, 3)
            kg, kr = jax.random.split(k_gen, 2)

            g_p, g_s, g_e = self._sample_global(kg, self.n_cand)
            r_p, r_s, r_e = self._sample_refine(kr, best_p, best_s, best_e, self.n_refine)

            mixed_p = jnp.concatenate([g_p[: self.n_global], r_p], axis=0)
            mixed_s = jnp.concatenate([g_s[: self.n_global], r_s], axis=0)
            mixed_e = jnp.concatenate([g_e[: self.n_global], r_e], axis=0)

            below = best_score < self.threshold
            p_batch = jnp.where(below, g_p, mixed_p)
            s_batch = jnp.where(below, g_s, mixed_s)
            e_batch = jnp.where(below, g_e, mixed_e)

            scores = jax.vmap(
                lambda p, s, e: self._eval_significant_advantage(
                    p, s, e, mo_rewards_t, all_obs_t, all_la_t
                )
            )(p_batch, s_batch, e_batch)

            idx = jnp.argmax(scores)
            cand_score = scores[idx]
            improved = cand_score > best_score
            new_best_p = jnp.where(improved, p_batch[idx], best_p)
            new_best_s = jnp.where(improved, s_batch[idx], best_s)
            new_best_e = jnp.where(improved, e_batch[idx], best_e)
            new_best_score = jnp.where(improved, cand_score, best_score)

            return (new_best_p, new_best_s, new_best_e, new_best_score, k_next), None

        init = (
            jnp.zeros(5),
            jnp.int32(self.s_min),
            jnp.int32(self.e_max),
            jnp.array(-jnp.inf, dtype=jnp.float32),
            key,
        )
        carry, _ = jax.lax.scan(step, init, None, length=self.num_iters)
        return carry[0], carry[1], carry[2], carry[3]

    @partial(jax.jit, static_argnames=("self",))
    def optimize_all(self, all_obs_c, all_la_c, mo_r_c, key):
        """Scan over chunks of `chunk_size` trajectories (vmap inside each).

        Args (chunked): all_obs_c, all_la_c, mo_r_c with leading axis
            (num_chunks, chunk_size, ...).
        Returns (best_p, best_s, best_e, best_score), each flattened to
            (num_traj, ...).
        """
        def chunk_step(key, chunk):
            ao, al, mr = chunk
            key, subkey = jax.random.split(key)
            bp, bs, be, bsc = jax.vmap(self._optimize_one)(
                ao, al, mr, jax.random.split(subkey, ao.shape[0])
            )
            return key, (bp, bs, be, bsc)

        _, (bp, bs, be, bsc) = jax.lax.scan(
            chunk_step, key, (all_obs_c, all_la_c, mo_r_c)
        )
        num_traj = bp.shape[0] * bp.shape[1]
        best_p = bp.reshape(num_traj, 5)
        best_s = bs.reshape(num_traj)
        best_e = be.reshape(num_traj)
        best_score = bsc.reshape(num_traj)
        return best_p, best_s, best_e, best_score



    @partial(jax.jit, static_argnames=("self",))
    def compute_lambda_return_and_GAE(
        self,
        critic_params,
        moving_mean,
        moving_std,
        transitions: PPOTransition,
    ) -> Tuple[jax.Array, jax.Array]:
        """Reverse TD(lambda) / GAE on packed transitions reshaped to (M, L).
        """
        discount = 0.99
        gae_lambda = 0.95
        L = self.e_max
        n = transitions.obs.shape[0]
        M = n // L

        def to_time_major(x):
            return jnp.transpose(x.reshape(M, L, x.shape[-1]), (1, 0, 2))

        obs = to_time_major(transitions.obs)
        zs = to_time_major(transitions.zs)
        rewards = to_time_major(transitions.rewards)
        truncations = to_time_major(transitions.truncations)

        def step1(carry, data):
            last_target, last_v_value, t = carry
            obs_t, zs_t, reward, truncation = data
            v_value = (
                self.critic_network.apply(critic_params, obs_t, zs_t)
                * moving_std + moving_mean
            )
            bootstrap = (truncation > 0.5) | (t == 0)
            target = jnp.where(
                bootstrap,
                v_value,
                reward + discount * (
                    gae_lambda * last_target + (1 - gae_lambda) * last_v_value
                ),
            )
            return (target, v_value, t + 1), v_value


        def step2(carry, data):
            last_target, last_v_value = carry
            reward, truncation, v_value = data
            bootstrap = truncation > 0.5
            target = jnp.where(
                bootstrap,
                v_value,
                reward + discount * (
                    gae_lambda * last_target + (1 - gae_lambda) * last_v_value
                ),
            )
            gae = target - v_value
            return (target, v_value), (target, gae)

        init = (jnp.zeros((M, 1)), jnp.zeros((M, 1)), jnp.int32(0))
        starting_carry, v_values = jax.lax.scan(
            step1, init, (obs, zs, rewards, truncations), reverse=True,
        )
        init_target = jnp.roll(starting_carry[0], -1, axis=0)
        init_last_v = jnp.roll(starting_carry[1], -1, axis=0)

        _, (td_lambda, gae) = jax.lax.scan(
            step2, (init_target, init_last_v), (rewards, truncations, v_values), reverse=True,
        )

        def to_flat(x):
            return jnp.transpose(x, (1, 0, 2)).reshape(n, x.shape[-1])

        return to_flat(td_lambda), to_flat(gae)
    

    # ------------------------------------------------- post-processing (Python)
    def reorganize(
        self,
        obs_t: jax.Array,
        la_t: jax.Array,
        act_t: jax.Array,
        dones_t: jax.Array,
        truncs_t: jax.Array,
        mo_r_t: jax.Array,
        best_p: jax.Array,
        best_s: jax.Array,
        best_e: jax.Array,
        best_score: jax.Array,
        shuf_key: jax.Array,
        max_data_size=None,
    ) -> PPOTransition:
        """Build a shuffle table, concat selected slices until ``n_used``.

        Table columns are ``(traj_idx, start, end, if_use)`` with inclusive
        ``end`` clipped to the last transition index (``rollout_len - 1``;
        ``best_e`` may point at the extra bootstrap state in ``all_obs``).
        ``if_use`` is 0 when ``best_score`` is non-positive or the trajectory
        contains a done / truncation. Trajectories are shuffled, then usable
        ``(i, s, e)`` pairs are collected (last ``e`` cut to fit ``n_used``).
        Fields are packed to a 3D tensor and sliced with ``s:e+1``. Truncation
        is 1.0 at each segment's last step (``cumsum(lengths) - 1``).
        """
        num_traj, rollout_len = obs_t.shape[0], obs_t.shape[1]
        last_t = rollout_len - 1  # 63 when rollout_len is 64

        if max_data_size is None:
            n_used = num_traj * rollout_len
        else:
            n_used = max_data_size

        has_term = (dones_t.sum(axis=1) + truncs_t.sum(axis=1))[:, 0] > 0
        if_use = (best_score > 0) & (~has_term)
        end_clipped = jnp.minimum(best_e, jnp.int32(last_t))
        table = jnp.stack(
            [
                jnp.arange(num_traj, dtype=jnp.int32),
                best_s.astype(jnp.int32),
                end_clipped.astype(jnp.int32),
                if_use.astype(jnp.int32),
            ],
            axis=1,
        )  # (num_traj, 4)

        n_traj_sel = float(jnp.sum(if_use))
        seg_lens = jnp.where(if_use, end_clipped - best_s + 1, 0)
        n_trans_sel = float(jnp.sum(seg_lens))
        print(
            f"portion trajectories selected: {n_traj_sel / num_traj:.4f}  "
            f"({n_traj_sel:.0f}/{num_traj})"
        )
        print(
            f"portion transitions selected: {n_trans_sel / n_used:.4f}  "
            f"({n_trans_sel:.0f}/{n_used})"
        )
        if n_traj_sel > 0:
            avg_sel = float(jnp.mean(best_score[if_use]))
            print(f"avg best_score (selected): {avg_sel:.4f}")
        else:
            print("avg best_score (selected): N/A (none selected)")

        dummy_step = jnp.zeros((num_traj, rollout_len, 1))
        scalar_r = jnp.sum(mo_r_t * best_p[:, None, :], axis=-1, keepdims=True)
        packed = PPOTransition(
            obs=obs_t,
            actions=act_t,
            zs=jnp.concatenate(
                [
                    la_t,
                    jnp.broadcast_to(
                        best_p[:, None, :],
                        (num_traj, rollout_len, best_p.shape[-1]),
                    ),
                ],
                axis=-1,
            ),
            log_likelihood=dummy_step,
            rewards=scalar_r,
            td_lambda_returns=dummy_step,
            gaes=dummy_step,
            dones=dummy_step,
            truncations=dummy_step,
            weights=dummy_step,
        )

        table = np.asarray(table[jax.random.permutation(shuf_key, num_traj)], dtype=np.int32)
        used = table[table[:, 3] == 1]
        n_rows = used.shape[0]
        del table

        if n_rows == 0:
            print("no usable trunks; returning dummy batch")
            dummy = jnp.zeros((n_used, 1))
            return PPOTransition(
                obs=jnp.zeros((n_used, obs_t.shape[-1])),
                actions=jnp.zeros((n_used, act_t.shape[-1])),
                zs=jnp.zeros((n_used, packed.z_dim)),
                log_likelihood=dummy,
                rewards=dummy,
                td_lambda_returns=dummy,
                gaes=dummy,
                dones=dummy,
                truncations=dummy,
                weights=dummy,
            )

        flat3d = np.asarray(
            packed.flatten().reshape(num_traj, rollout_len, packed.flatten_dim)
        )
        pairs = []
        n = 0
        k = 0
        print("starting loop")
        while n < n_used:
            i, s, e = int(used[k, 0]), int(used[k, 1]), int(used[k, 2])
            L = e - s + 1  # inclusive end
            if n + L > n_used:
                e = s + (n_used - n) - 1
                L = e - s + 1
            pairs.append((i, s, e))
            n += L
            k += 1
            if k == n_rows:
                k = 0
        print("loop end")

        lengths = np.array([e - s + 1 for _, s, e in pairs], dtype=np.int32)
        truncation = np.zeros((n_used, 1), dtype=np.float32)
        truncation[np.cumsum(lengths) - 1] = 1.0

        flat = jnp.asarray(
            np.concatenate([flat3d[i, s : e + 1] for i, s, e in pairs], axis=0)
        )
        out = PPOTransition.from_flatten(flat, packed)
        return out.replace(truncations=jnp.asarray(truncation))


    def relocate(
        self,
        transitions: MORelocationTransition,
        final_info: Tuple[jax.Array, jax.Array],
        key,
        max_data_size=None,
    ) -> PPOTransition:
        final_obs, final_last_actions = final_info
        all_obs = jnp.concatenate(
            [transitions.obs, jnp.expand_dims(final_obs, 1)], axis=1
        )  # (num_iter, 65, vec_env, obs_dim)
        all_last_actions = jnp.concatenate(
            [transitions.last_actions, jnp.expand_dims(final_last_actions, 1)], axis=1
        )

        num_iter, rollout_len, n_env = transitions.obs.shape[:3]
        num_traj = num_iter * n_env
        num_chunks = num_traj // self.chunk_size

        def to_traj(x):
            return jnp.transpose(x, (0, 2, 1, 3)).reshape(
                num_traj, x.shape[1], x.shape[3]
            )

        all_obs_t = to_traj(all_obs)
        all_la_t = to_traj(all_last_actions)
        mo_r_t = to_traj(transitions.mo_rewards)
        obs_t = all_obs_t[:, :rollout_len, :]
        la_t = all_la_t[:, :rollout_len, :]
        act_t = to_traj(transitions.actions)
        dones_t = to_traj(transitions.dones)
        truncs_t = to_traj(transitions.truncations)

        all_obs_c = all_obs_t.reshape(
            num_chunks, self.chunk_size, *all_obs_t.shape[1:]
        )
        all_la_c = all_la_t.reshape(
            num_chunks, self.chunk_size, *all_la_t.shape[1:]
        )
        mo_r_c = mo_r_t.reshape(num_chunks, self.chunk_size, *mo_r_t.shape[1:])

        key, opt_key, shuf_key = jax.random.split(key, 3)
        best_p, best_s, best_e, best_score = self.optimize_all(
            all_obs_c, all_la_c, mo_r_c, opt_key
        )

        print("optimization complete")
        del all_obs_c, all_obs_t, all_obs, all_la_t

        organized_transitions = self.reorganize(
            obs_t, la_t, act_t, dones_t, truncs_t, mo_r_t,
            best_p, best_s, best_e, best_score, shuf_key, max_data_size,
        ) # (n_used, ...)

        del obs_t, la_t

        print("reorganized")
        lambda_returns, gaes = self.compute_lambda_return_and_GAE(
            self.critic_params,
            self.moving_mean,
            self.moving_std,
            organized_transitions,
        )
        mean_gae = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        # clipped_gaes = jnp.clip(gaes, 0.0, mean_gae + 3 * gae_std)

        print("GAE compute complete")
        print("average GAE:", mean_gae, "GAE std:", gae_std)
        organized_transitions = organized_transitions.replace(
            td_lambda_returns=lambda_returns,
            gaes=gaes,
            weights=jnp.ones_like(gaes),
        )

        return organized_transitions


def relocate(
    transitions: MORelocationTransition,
    final_info: Tuple[jax.Array, jax.Array],
    critic_network: nn.module,
    critic_params,
    moving_mean: float,
    moving_std: float,
    config: dict,
    key,
    max_data_size=None,
) -> PPOTransition:
    """Backward-compatible thin wrapper around ``Relocator``."""
    return Relocator(
        critic_network, critic_params, moving_mean, moving_std, config
    ).relocate(transitions, final_info, key, max_data_size)



def eval_return_and_GAE(
    transitions: PPOTransition,
    critic_network: nn.module,
    critic_params,
    moving_mean,
    moving_std,
    config: dict,
    ):

    return Relocator(
            critic_network, critic_params, moving_mean, moving_std, config
        ).compute_lambda_return_and_GAE(
            critic_params,
            moving_mean,
            moving_std,
            transitions,
        )


