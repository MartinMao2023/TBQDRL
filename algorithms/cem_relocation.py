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
(currently a placeholder); until that is filled in, ``reorganize`` falls back
to broadcasting the scalar trajectory score as the per-step weight.
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
        3. ``reorganize``     -> slice / concat / tile / shuffle -> PPOTransition
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

    # ------------------------------------------------- per-step GAE (placeholder)
    def calculate_td_lambda_return(
        self,
        v_values: jax.Array,
        rewards: jax.Array,
        end_indices: jax.Array,
    ) -> jax.Array:
        """Per-step GAE-lambda of the relocated trunks. PLACEHOLDER.

        Inputs are time-major to allow a jax.lax.scan over time with a
        jax.vmap over the batch inside it:
            v_values       (T+1, B, 1)  critic values incl. the final state
            rewards        (T,   B, 1)  scalarized rewards (mo_rewards . best_p)
            start_indices  (B,)        trunk start per trajectory
            end_indices    (B,)        trunk end per trajectory (bootstrap idx)

        Returns:
            gae            (T, B, 1)   GAE = td_lambda - v, zero outside
                                       [start, end); bootstrap = v_values[end].

        Implementation notes (to be filled in):
          * reverse jax.lax.scan over time, jax.vmap over batch;
          * done / truncation are ignored (trajectories containing them are
            discarded before this stage);
          * position mask: td[t+1] = v_values[end] for any t+1 >= end.
        """
        discount = 0.99
        td_lambda_discount = 0.95
        rollout_length = rewards.shape[0]   # T
        batch_size = rewards.shape[1]       # B

        # Per-batch bootstrap value v_values[end] (the trunk's terminal state),
        # used only to seed the reverse scan's initial carry.
        v_end = v_values[end_indices, jnp.arange(batch_size)]  # (B, 1)

        def scan_calculate_td_lambda(carry, data):
            # carry (per batch): last_td = td[t+1], last_value = v[t+1],
            # t = current step index, end = trunk end index.
            last_td, t, end = carry
            reward, v_value = data  # reward[t], v[t]

            # standard TD(lambda) recursion (no done/trunc: such trajs discarded)
            td = reward + discount * (
                (1 - td_lambda_discount) * v_value
                + td_lambda_discount * last_td
            )

            td = jnp.where(t >= end, last_td, td)

            return (td, t - 1, end), td

        _, td_lambda_values = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (
                v_end,                                                       # last_td                                                      # last_value
                jnp.full((batch_size,), rollout_length - 1, dtype=jnp.int32),  # t = T-1
                end_indices,                                                  # end per batch
            ),
            (rewards, v_values[1:]),
            reverse=True,
        )  # (T, B, 1)

        return td_lambda_values
    
    # ------------------------------------------------- post-processing (Python)
    def reorganize(
        self,
        obs_t: jax.Array,
        la_t: jax.Array,
        act_t: jax.Array,
        ll_t: jax.Array,
        dones_t: jax.Array,
        truncs_t: jax.Array,
        best_p: jax.Array,
        best_s: jax.Array,
        best_e: jax.Array,
        best_score: jax.Array,
        gae_t: jax.Array,
        shuf_key: jax.Array,
        max_data_size=None,
    ) -> PPOTransition:
        """Slice the selected trunks, concat, tile back to `n_used`, shuffle.

        ``best_score`` drives selection (positive) and the done/trunc
        discard; the per-step weight of each selected trunk is the
        corresponding slice of ``gae_t`` (the per-step GAE =
        ``td_lambda_return - v_values`` produced by
        :meth:`calculate_td_lambda_return`).
        """
        num_traj, rollout_len = obs_t.shape[0], obs_t.shape[1]
        obs_dim = obs_t.shape[-1]
        act_dim = act_t.shape[-1]
        pref_dim = best_p.shape[-1]

        if max_data_size is None:
            n_used = num_traj * rollout_len
        else:
            n_used = max_data_size

        # discard trajectories that contain any done / truncation
        has_term = (dones_t.sum(axis=1) + truncs_t.sum(axis=1)) > 0  # (N, 1)
        best_score = jnp.where(has_term[:, 0], jnp.float32(-1000.0), best_score)

        selected_mask = best_score > 0
        n_traj_sel = float(jnp.sum(selected_mask))
        seg_lens = jnp.where(selected_mask, best_e - best_s, 0)
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
            avg_sel = float(jnp.mean(best_score[selected_mask]))
            print(f"avg best_score (selected): {avg_sel:.4f}")
        else:
            print("avg best_score (selected): N/A (none selected)")

        obs_np = np.asarray(obs_t)
        act_np = np.asarray(act_t)
        la_np = np.asarray(la_t)
        ll_np = np.asarray(ll_t)
        bp_np = np.asarray(best_p)
        bs_np = np.asarray(best_s).astype(int)
        be_np = np.asarray(best_e).astype(int)
        bsc_np = np.asarray(best_score).astype(float)
        gae_np = np.asarray(gae_t)  # (num_traj, rollout_len, 1)

        obs_list, z_list, act_list, w_list, ll_list = [], [], [], [], []
        for i in range(num_traj):
            sc = bsc_np[i]
            if sc > 0:
                s = int(bs_np[i])
                e = int(be_np[i])
                L = e - s
                obs_list.append(obs_np[i, s:e])
                act_list.append(act_np[i, s:e])
                seg_la = la_np[i, s:e]
                seg_ll = ll_np[i, s:e]
                seg_p = np.broadcast_to(bp_np[i], (L, pref_dim))
                z_list.append(np.concatenate([seg_la, seg_p], axis=-1))
                # per-step weight: the GAE of the selected trunk.
                w_list.append(gae_np[i, s:e])
                ll_list.append(seg_ll)

        if len(obs_list) > 0:
            obs_sel = np.concatenate(obs_list, axis=0)
            z_sel = np.concatenate(z_list, axis=0)
            act_sel = np.concatenate(act_list, axis=0)
            w_sel = np.concatenate(w_list, axis=0)
            ll_sel = np.concatenate(ll_list, axis=0)
        else:
            obs_sel = obs_np.reshape(n_used, obs_dim)
            act_sel = act_np.reshape(n_used, act_dim)
            la_flat = la_np.reshape(n_used, act_dim)
            ll_flat = ll_np.reshape(n_used, 1)
            p_flat = np.broadcast_to(
                bp_np, (num_traj, rollout_len, pref_dim)
            ).reshape(n_used, pref_dim)
            z_sel = np.concatenate([la_flat, p_flat], axis=-1)
            w_sel = np.zeros((n_used, 1), dtype=np.float32)
            ll_sel = ll_flat

        N_sel = obs_sel.shape[0]
        repeats = n_used // N_sel + 1
        obs_tiled = np.tile(obs_sel, (repeats, 1))[:n_used]
        z_tiled = np.tile(z_sel, (repeats, 1))[:n_used]
        act_tiled = np.tile(act_sel, (repeats, 1))[:n_used]
        w_tiled = np.tile(w_sel, (repeats, 1))[:n_used]
        ll_tiled = np.tile(ll_sel, (repeats, 1))[:n_used]

        seed_int = int(np.asarray(shuf_key).flat[0])
        rng = np.random.default_rng(seed_int)
        perm = rng.permutation(n_used)
        weights = jnp.asarray(w_tiled[perm])
        dummy = jnp.zeros((n_used, 1))

        print("average GAE:", jnp.mean(weights))

        return PPOTransition(
            obs=jnp.asarray(obs_tiled[perm]),
            actions=jnp.asarray(act_tiled[perm]),
            zs=jnp.asarray(z_tiled[perm]),
            log_likelihood=jnp.asarray(ll_tiled[perm]),
            rewards=dummy,
            td_lambda_returns=dummy,
            gaes=dummy,
            dones=dummy,
            truncations=dummy,
            weights=weights / (1e-6 + jnp.mean(weights)),
        )

    # ---------------------------------------------- per-trajectory values / rewards
    @partial(jax.jit, static_argnames=("self",))
    def _compute_values(self, all_obs_t, all_la_t, mo_r_t, best_p):
        """Critic values (incl. final state) and scalarized rewards per traj.

        all_obs_t (N, 65, obs_dim), all_la_t (N, 65, act_dim),
        mo_r_t (N, 64, mo_dim), best_p (N, 5).
        Returns v_values (N, 65, 1), rewards (N, 64, 1).
        """
        zs = jnp.concatenate(
            [all_la_t, jnp.broadcast_to(best_p[:, None, :], all_la_t.shape[:2] + (best_p.shape[-1],))],
            axis=-1,
        )  # (N, 65, z_dim)
        v_values = self.critic_network.apply(self.critic_params, all_obs_t, zs) \
            * self.moving_std + self.moving_mean  # (N, 65, 1)
        rewards = jnp.sum(mo_r_t * best_p[:, None, :], axis=-1, keepdims=True)  # (N, 64, 1)
        return v_values, rewards

    # ------------------------------------------------- orchestration
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
        ll_t = to_traj(transitions.log_likelihood)
        dones_t = to_traj(transitions.dones)
        truncs_t = to_traj(transitions.truncations)

        all_obs_c = all_obs_t.reshape(
            num_chunks, self.chunk_size, *all_obs_t.shape[1:]
        )
        all_la_c = all_la_t.reshape(
            num_chunks, self.chunk_size, *all_la_t.shape[1:]
        )
        mo_r_c = mo_r_t.reshape(num_chunks, self.chunk_size, *mo_r_t.shape[1:])

        key, opt_key, gae_key, shuf_key = jax.random.split(key, 4)
        best_p, best_s, best_e, best_score = self.optimize_all(
            all_obs_c, all_la_c, mo_r_c, opt_key
        )

        # ---- per-step GAE of the selected trunks ----
        # v_values (N, 65, 1), rewards (N, 64, 1) under the best preference.
        v_values, rewards = self._compute_values(all_obs_t, all_la_t, mo_r_t, best_p)
        # time-major (T+1, B, 1) / (T, B, 1) for the reverse scan + vmap.
        v_tm = jnp.transpose(v_values, (1, 0, 2))      # (65, N, 1)
        r_tm = jnp.transpose(rewards, (1, 0, 2))        # (64, N, 1)
        td_lambda_return = self.calculate_td_lambda_return(
            v_tm, r_tm, best_e
        )  # (64, N, 1)
        gae_tm = jnp.clip(td_lambda_return - v_tm[:-1], min=0)  # (64, N, 1)
        gae_t = jnp.transpose(gae_tm, (1, 0, 2))         # (N, 64, 1)

        return self.reorganize(
            obs_t, la_t, act_t, ll_t, dones_t, truncs_t,
            best_p, best_s, best_e, best_score, gae_t, shuf_key, max_data_size,
        )


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


