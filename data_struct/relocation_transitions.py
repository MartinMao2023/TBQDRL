from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from custom_types import Action, Observation, RNGKey


class MORelocationTransition(flax.struct.PyTreeNode):
    """Rollout buffer for the relocation pipeline (multi-objective).

    Stores teacher rollouts collected under a *preference* task.  The full
    conditioning vector is ``z = concat(last_action, preference)``; both the
    full ``z`` and the ``last_action`` part are stored so that relocation can
    re-assemble ``z`` with a *new* preference while keeping the executed
    action and the ``last_action`` unchanged.

    The multi-objective reward vector ``mo_rewards`` (one entry per
    objective) is stored instead of a scalar reward, so that the scalar
    reward under any (possibly relocated) preference can be reconstructed as
    ``sum(mo_rewards * preference)`` when computing the relocation advantage.

    By convention every field has at least one trailing dimension, so e.g.
    ``dones`` has shape ``(..., 1)`` to match the critic output.

    Shapes (with batch prefix ``...``):
        obs              (..., obs_dim)
        zs               (..., z_dim)           concat(last_action, preference)
        actions          (..., action_dim)
        last_actions     (..., action_dim)      the last_action part of z
        mo_rewards       (..., mo_reward_dim)   per-objective reward vector
        weights          (..., 1)               truncation weight (as in PPO)
        dones            (..., 1)
        truncations      (..., 1)
        td_lambda_returns(..., 1)               filled in Stage 1/3
        log_likelihood  (..., 1)               log p_teacher(a) of the sampled action
    """

    obs: Observation
    zs: jax.Array
    actions: Action
    last_actions: jax.Array
    mo_rewards: jax.Array
    weights: jax.Array
    dones: jax.Array
    truncations: jax.Array
    td_lambda_returns: jax.Array
    # log p_teacher(a | obs, z): the teacher (Gaussian PPO) log-prob of the
    # sampled action, recorded so the two-stage distiller can use it as the
    # old_log_likelihood of the importance-sampling stage.
    log_likelihood: jax.Array

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self.obs.shape[-1]

    @property
    def z_dim(self) -> int:
        return self.zs.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[-1]

    @property
    def mo_reward_dim(self) -> int:
        return self.mo_rewards.shape[-1]

    @property
    def flatten_dim(self) -> int:
        return (
            self.observation_dim
            + self.z_dim
            + self.action_dim          # actions
            + self.action_dim          # last_actions
            + self.mo_reward_dim       # mo_rewards
            + 1                        # weights
            + 1                        # dones
            + 1                        # truncations
            + 1                        # td_lambda_returns
            + 1                        # log_likelihood
        )

    # ------------------------------------------------------------------
    # Flatten / unflatten
    # ------------------------------------------------------------------

    def flatten(self) -> jax.Array:
        """Concatenate all fields into a 2-D array of shape (N, flatten_dim)."""
        batch_shape = self.obs.shape[:-1]
        return jnp.concatenate(
            [
                self.obs,
                self.zs,
                self.actions,
                self.last_actions,
                self.mo_rewards,
                self.weights,
                self.dones,
                self.truncations,
                self.td_lambda_returns,
                self.log_likelihood,
            ],
            axis=-1,
        ).reshape(-1, self.flatten_dim)

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jax.Array,
        transition: "MORelocationTransition",
    ) -> "MORelocationTransition":
        """Reconstruct a MORelocationTransition from a flattened array.

        Args:
            flattened_transition: shape (batch_size, flatten_dim)
            transition: a reference instance (possibly a dummy) carrying the
                dimension metadata needed for unpacking.
        """
        obs_dim = transition.observation_dim
        z_dim = transition.z_dim
        action_dim = transition.action_dim
        mo_reward_dim = transition.mo_reward_dim

        cursor = 0
        obs = flattened_transition[:, cursor : cursor + obs_dim]
        cursor += obs_dim
        zs = flattened_transition[:, cursor : cursor + z_dim]
        cursor += z_dim
        actions = flattened_transition[:, cursor : cursor + action_dim]
        cursor += action_dim
        last_actions = flattened_transition[:, cursor : cursor + action_dim]
        cursor += action_dim
        mo_rewards = flattened_transition[:, cursor : cursor + mo_reward_dim]
        cursor += mo_reward_dim
        weights = flattened_transition[:, cursor : cursor + 1]
        cursor += 1
        dones = flattened_transition[:, cursor : cursor + 1]
        cursor += 1
        truncations = flattened_transition[:, cursor : cursor + 1]
        cursor += 1
        td_lambda_returns = flattened_transition[:, cursor : cursor + 1]
        cursor += 1
        log_likelihood = flattened_transition[:, cursor : cursor + 1]

        return cls(
            obs=obs,
            zs=zs,
            actions=actions,
            last_actions=last_actions,
            mo_rewards=mo_rewards,
            weights=weights,
            dones=dones,
            truncations=truncations,
            td_lambda_returns=td_lambda_returns,
            log_likelihood=log_likelihood,
        )

    @classmethod
    def init_dummy(
        cls,
        observation_dim: int,
        action_dim: int,
        z_dim: int,
        mo_reward_dim: int,
    ) -> "MORelocationTransition":
        """Create a dummy instance to be used as a shape reference."""
        return cls(
            obs=jnp.zeros(shape=(1, observation_dim)),
            zs=jnp.zeros(shape=(1, z_dim)),
            actions=jnp.zeros(shape=(1, action_dim)),
            last_actions=jnp.zeros(shape=(1, action_dim)),
            mo_rewards=jnp.zeros(shape=(1, mo_reward_dim)),
            weights=jnp.zeros(shape=(1, 1)),
            dones=jnp.zeros(shape=(1, 1)),
            truncations=jnp.zeros(shape=(1, 1)),
            td_lambda_returns=jnp.zeros(shape=(1, 1)),
            log_likelihood=jnp.zeros(shape=(1, 1)),
        )

    def shuffle(self, key: RNGKey) -> "MORelocationTransition":
        """Randomly permute transitions along the batch axis."""
        flattened = self.flatten()
        index = jax.random.permutation(key, flattened.shape[0])
        return self.from_flatten(flattened[index], self)
