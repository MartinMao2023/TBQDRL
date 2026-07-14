"""Trajectory relocation optimization (Stage 3).

Given a critic and a set of rollout trajectories (collected under some
preference), search for the (preference, start, end) that maximizes the
relocation advantage of each trajectory, then relabel the selected segments
with their new preference and return them as a ``PPOTransition``.

The search is a simple random-search + local-refinement scheme:
  * each iteration draws ``n_global`` global candidates and ``n_refine``
    candidates around the current best (or all global if the best so far is
    below a discard threshold);
  * the best candidate is updated and emitted every iteration;
  * after the search, trajectories containing a done/truncation are forced to a
    discard score, only segments with a positive best score are sliced out and
    concatenated, and the result is tiled back to the input transition count.
"""

from data_struct.relocation_transitions import MORelocationTransition
from data_struct import PPOTransition
from typing import Tuple
import jax.numpy as jnp
import jax
import numpy as np
from tools import calculate_coefs_for_trajectory
import flax.linen as nn


def relocate(
    transitions: MORelocationTransition,
    final_info: Tuple[jax.Array, jax.Array],
    critic_network: nn.module,
    critic_params,
    config: dict,
    key,
    ) -> PPOTransition:
    """Relabel trajectories to new goals by maximizing the relocation
    advantage, and select which segments to keep.

    Inputs:
        transitions     MORelocationTransition; trajectory structure preserved,
                        shape (num_iter, rollout_length, vec_env, ...).
        final_info      Tuple (final_obs, final_last_actions), each
                        (num_iter, vec_env, ...); concatenated onto the
                        trajectory data to form the per-trajectory 65 states.
        critic_network  value network used to evaluate advantages.
        critic_params   critic parameters.
        config          relocation hyperparameters (see defaults below).
        key             RNGKey.

    Returns a ``PPOTransition`` of length equal to the input transition count
    (``num_iter * rollout_length * vec_env``), holding the relabeled selected
    segments tiled to that size and shuffled.
    """

    # ------------------------------------------------------------------ config
    threshold = config.get("threshold", -1000)
    num_iters = config.get("num_iters", 128)
    n_global = config.get("num_global", 4)
    n_refine = config.get("num_refine", 4)
    pref_std = config.get("pref_std", 0.2)
    idx_radius = config.get("idx_radius", 12)
    s_min = config.get("s_min", 0)
    s_max = config.get("s_max", 48)
    min_len = config.get("min_len", 16)
    e_max = config.get("e_max", 64)
    chunk_size = config.get("chunk_size", 1024)
    n_cand = n_global + n_refine  # 8
    trunk_penalty = 20

    # ------------------------------------------------------ advantage evaluator
    def eval_significant_advantage(
        preference: jax.Array,
        start: int,
        end: int,
        mo_rewards: jax.Array,
        obs: jax.Array,
        last_actions: jax.Array,
        ) -> float:

        total_length = mo_rewards.shape[0]

        reward_coefs, v_coefs = calculate_coefs_for_trajectory(total_length, start, end, 4, 0.99, 0.95)

        zs = jnp.concatenate([last_actions, jnp.tile(preference, (total_length + 1, 1))], axis=-1)
        values = critic_network.apply(critic_params, obs, zs) * 99.61838 + 203.00978
        rewards = jnp.sum(mo_rewards * preference, axis=-1, keepdims=True) # (64, 1)
        trunk_length = end - start
        length_penalty = trunk_penalty * 2.578434 * (trunk_length * 0.125 - 1.578434 * (1 - jnp.exp(-0.06134363 * trunk_length)))

        advantage = jnp.sum(rewards * reward_coefs) + jnp.sum(values * v_coefs) - length_penalty

        return advantage

    # --------------------------------------------------------- candidate samplers
    def project(x):
        """Map an unconstrained 5-vector onto the feasible preference patch:
        last three non-negative (abs), then renormalize to unit length."""
        x = jnp.concatenate([x[..., :2], jnp.abs(x[..., 2:])], axis=-1)
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

    def sample_global(k, n):
        kp, ks, ke = jax.random.split(k, 3)
        p = project(jax.random.normal(kp, (n, 5)))
        s = jax.random.randint(ks, (n,), s_min, s_max + 1)        # [0, 48]
        e = jax.random.randint(ke, (n,), s + min_len, e_max + 1)  # [s+16, 64]
        return p, s, e

    def sample_refine(k, best_p, best_s, best_e, n):
        kp, ks, ke = jax.random.split(k, 3)
        p = project(best_p + pref_std * jax.random.normal(kp, (n, 5)))
        s_lo = jnp.maximum(best_s - idx_radius, s_min)
        s_hi = jnp.minimum(best_s + idx_radius, s_max)
        s = jax.random.randint(ks, (n,), s_lo, s_hi + 1)
        e_lo = jnp.maximum(best_e - idx_radius, s + min_len)
        e_hi = jnp.minimum(best_e + idx_radius, e_max)
        e_hi = jnp.maximum(e_hi, e_lo)  # guard against an empty range
        e = jax.random.randint(ke, (n,), e_lo, e_hi + 1)
        return p, s, e

    # ----------------------------------------------------- per-trajectory search
    def optimize_one(all_obs_t, all_last_actions_t, mo_rewards_t, key):
        """32-iteration random-search + refine for one trajectory.

        Emits (best_p, best_s, best_e, best_score) every iteration.
        """
        def step(carry, _):
            best_p, best_s, best_e, best_score, key = carry
            key, k_gen, k_next = jax.random.split(key, 3)
            kg, kr = jax.random.split(k_gen, 2)

            # Always draw both global and refine candidates so the RNG is
            # consumed identically regardless of the mode (no cond-on-RNG).
            g_p, g_s, g_e = sample_global(kg, n_cand)
            r_p, r_s, r_e = sample_refine(kr, best_p, best_s, best_e, n_refine)

            mixed_p = jnp.concatenate([g_p[:n_global], r_p], axis=0)
            mixed_s = jnp.concatenate([g_s[:n_global], r_s], axis=0)
            mixed_e = jnp.concatenate([g_e[:n_global], r_e], axis=0)

            below = best_score < threshold
            p_batch = jnp.where(below, g_p, mixed_p)
            s_batch = jnp.where(below, g_s, mixed_s)
            e_batch = jnp.where(below, g_e, mixed_e)

            scores = jax.vmap(lambda p, s, e: eval_significant_advantage(
                p, s, e, mo_rewards_t, all_obs_t, all_last_actions_t
            ))(p_batch, s_batch, e_batch)

            idx = jnp.argmax(scores)
            cand_score = scores[idx]
            improved = cand_score > best_score
            new_best_p = jnp.where(improved, p_batch[idx], best_p)
            new_best_s = jnp.where(improved, s_batch[idx], best_s)
            new_best_e = jnp.where(improved, e_batch[idx], best_e)
            new_best_score = jnp.where(improved, cand_score, best_score)

            emit = (new_best_p, new_best_s, new_best_e, new_best_score)
            return (new_best_p, new_best_s, new_best_e, new_best_score, k_next), emit

        init = (
            jnp.zeros(5),
            jnp.int32(s_min),
            jnp.int32(e_max),
            jnp.array(-jnp.inf, dtype=jnp.float32),
            key,
        )
        carry, curve = jax.lax.scan(step, init, None, length=num_iters)
        return carry, curve

    # ------------------------------------------- jitted batch optimizer (scan)
    @jax.jit
    def optimize_all(all_obs_c, all_la_c, mo_r_c, key):
        """Scan over the 32 chunks of 1024 trajectories (vmap inside each)."""
        def chunk_step(key, chunk):
            ao, al, mr = chunk
            key, subkey = jax.random.split(key)
            carry, curve = jax.vmap(optimize_one)(
                ao, al, mr, jax.random.split(subkey, ao.shape[0])
            )
            return key, (carry, curve)

        key, (carries, curves) = jax.lax.scan(
            chunk_step, key, (all_obs_c, all_la_c, mo_r_c)
        )
        best_p = carries[0].reshape(-1, 5)
        best_s = carries[1].reshape(-1)
        best_e = carries[2].reshape(-1)
        best_score = carries[3].reshape(-1)
        score_curve = curves[3].reshape(-1, num_iters)  # (num_traj, num_iters)
        return best_p, best_s, best_e, best_score, score_curve

    # ----------------------------------------------- build the 65-state bundles
    final_obs, final_last_actions = final_info
    all_obs = jnp.concatenate(
        [transitions.obs, jnp.expand_dims(final_obs, 1)], axis=1
    )  # (num_iter, 65, vec_env, obs_dim)
    all_last_actions = jnp.concatenate(
        [transitions.last_actions, jnp.expand_dims(final_last_actions, 1)], axis=1
    )  # (num_iter, 65, vec_env, act_dim)

    num_iter, rollout_len, n_env = transitions.obs.shape[:3]
    num_traj = num_iter * n_env
    num_chunks = num_traj // chunk_size

    def to_traj(x):
        # (num_iter, step, vec_env, F) -> (num_traj, step, F)
        return jnp.transpose(x, (0, 2, 1, 3)).reshape(num_traj, x.shape[1], x.shape[3])

    all_obs_t = to_traj(all_obs)              # (N, 65, obs_dim)
    all_la_t = to_traj(all_last_actions)      # (N, 65, act_dim)
    mo_r_t = to_traj(transitions.mo_rewards)  # (N, 64, mo_dim)
    obs_t = all_obs_t[:, :rollout_len, :]    # (N, 64, obs_dim)
    la_t = all_la_t[:, :rollout_len, :]      # (N, 64, act_dim)
    act_t = to_traj(transitions.actions)      # (N, 64, act_dim)
    dones_t = to_traj(transitions.dones)      # (N, 64, 1)
    truncs_t = to_traj(transitions.truncations)  # (N, 64, 1)

    # (num_chunks, chunk_size, step, F)
    all_obs_c = all_obs_t.reshape(num_chunks, chunk_size, *all_obs_t.shape[1:])
    all_la_c = all_la_t.reshape(num_chunks, chunk_size, *all_la_t.shape[1:])
    mo_r_c = mo_r_t.reshape(num_chunks, chunk_size, *mo_r_t.shape[1:])

    key, opt_key, shuf_key = jax.random.split(key, 3)
    best_p, best_s, best_e, best_score, score_curve = optimize_all(
        all_obs_c, all_la_c, mo_r_c, opt_key
    )

    # print("relocation per-iteration mean best_score:")
    # print(jnp.mean(score_curve, axis=0))  # (num_iters,)

    # discard trajectories that contain any done / truncation
    has_term = (dones_t.sum(axis=1) + truncs_t.sum(axis=1)) > 0  # (N, 1)
    best_score = jnp.where(has_term[:, 0], jnp.float32(-1000.0), best_score)

    # selection metrics (after discarding done/truncation trajectories)
    n_used = num_traj * rollout_len  # input transition count (excl. final)
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

    # ------------------------------------------------- post-processing (Python)
    obs_dim = obs_t.shape[-1]
    act_dim = act_t.shape[-1]
    pref_dim = best_p.shape[-1]

    obs_np = np.asarray(obs_t)
    act_np = np.asarray(act_t)
    la_np = np.asarray(la_t)
    bp_np = np.asarray(best_p)
    bs_np = np.asarray(best_s).astype(int)
    be_np = np.asarray(best_e).astype(int)
    bsc_np = np.asarray(best_score).astype(float)

    obs_list, z_list, act_list, w_list = [], [], [], []
    for i in range(num_traj):
        sc = bsc_np[i]
        if sc > 0:
            s = int(bs_np[i])
            e = int(be_np[i])
            L = e - s
            obs_list.append(obs_np[i, s:e])
            act_list.append(act_np[i, s:e])
            seg_la = la_np[i, s:e]
            seg_p = np.broadcast_to(bp_np[i], (L, pref_dim))
            z_list.append(np.concatenate([seg_la, seg_p], axis=-1))
            w_list.append(np.full((L, 1), sc, dtype=np.float32))

    if len(obs_list) > 0:
        obs_sel = np.concatenate(obs_list, axis=0)
        z_sel = np.concatenate(z_list, axis=0)
        act_sel = np.concatenate(act_list, axis=0)
        w_sel = np.concatenate(w_list, axis=0)
    else:
        # no trajectory selected: fall back to the full input relabeled with
        # the best preferences and zero weights (harmless for advantage-NLL).
        obs_sel = obs_np.reshape(n_used, obs_dim)
        act_sel = act_np.reshape(n_used, act_dim)
        la_flat = la_np.reshape(n_used, act_dim)
        p_flat = np.broadcast_to(
            bp_np, (num_traj, rollout_len, pref_dim)
        ).reshape(n_used, pref_dim)
        z_sel = np.concatenate([la_flat, p_flat], axis=-1)
        w_sel = np.zeros((n_used, 1), dtype=np.float32)

    # restore size: tile to >= n_used then slice (nearest multiple + 1)
    N_sel = obs_sel.shape[0]
    repeats = n_used // N_sel + 1
    obs_tiled = np.tile(obs_sel, (repeats, 1))[:n_used]
    z_tiled = np.tile(z_sel, (repeats, 1))[:n_used]
    act_tiled = np.tile(act_sel, (repeats, 1))[:n_used]
    w_tiled = np.tile(w_sel, (repeats, 1))[:n_used]

    # shuffle
    seed_int = int(np.asarray(shuf_key).flat[0])
    rng = np.random.default_rng(seed_int)
    perm = rng.permutation(n_used)
    obs_f = obs_tiled[perm]
    z_f = z_tiled[perm]
    act_f = act_tiled[perm]
    w_f = w_tiled[perm]
    weights=jnp.asarray(w_f)

    dummy = jnp.zeros((n_used, 1))
    demonstrate = PPOTransition(
        obs=jnp.asarray(obs_f),
        actions=jnp.asarray(act_f),
        zs=jnp.asarray(z_f),
        log_likelihood=dummy,
        rewards=dummy,
        td_lambda_returns=dummy,
        gaes=dummy,
        dones=dummy,
        truncations=dummy,
        weights=weights/(1e-6 + jnp.mean(weights)),
    )
    return demonstrate
