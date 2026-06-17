from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from custom_types import Action, Observation, RNGKey


class GMMDistillationTransition(flax.struct.PyTreeNode):
    """Distillation buffer for cloning a GMM teacher policy into a student.

    Stores the teacher's per-state GMM outputs (component means and selection
    weights) together with the conditioning state (obs, z).  std_logits are
    intentionally omitted: they are a global (non-state-conditioned) parameter
    shared across all inputs, so they can be extracted directly from the
    teacher's parameter tree outside this buffer.

    Shapes (with batch prefix `...`):
        obs               (..., obs_dim)
        zs                (..., z_dim)
        action_means      (..., k, action_dim)   — teacher's component means
        component_logits (..., k)               — softmax(teacher weight_logits)
    """

    obs: Observation
    zs: jax.Array
    action_means: jax.Array
    component_logits: jax.Array

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
    def num_components(self) -> int:
        return self.action_means.shape[-2]

    @property
    def action_dim(self) -> int:
        return self.action_means.shape[-1]

    @property
    def flatten_dim(self) -> int:
        return (
            self.observation_dim
            + self.z_dim
            + self.num_components * self.action_dim   # flattened means
            + self.num_components                     # weights
        )

    # ------------------------------------------------------------------
    # Flatten / unflatten
    # ------------------------------------------------------------------

    def flatten(self) -> jax.Array:
        """Concatenate all fields into a 2-D array of shape (N, flatten_dim)."""
        batch_shape = self.obs.shape[:-1]
        flat_means = self.action_means.reshape(
            *batch_shape, self.num_components * self.action_dim
        )
        return jnp.concatenate(
            [self.obs, self.zs, flat_means, self.component_logits], axis=-1
        ).reshape(-1, self.flatten_dim)

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jax.Array,
        transition: GMMDistillationTransition,
    ) -> GMMDistillationTransition:
        """Reconstruct a GMMDistillationTransition from a flattened array.

        Args:
            flattened_transition: shape (batch_size, flatten_dim)
            transition: a reference instance (possibly a dummy) that carries
                the dimension metadata needed for unpacking.
        """
        obs_dim = transition.observation_dim
        z_dim = transition.z_dim
        k = transition.num_components
        action_dim = transition.action_dim

        cursor = 0
        obs = flattened_transition[:, cursor : cursor + obs_dim]
        cursor += obs_dim
        zs = flattened_transition[:, cursor : cursor + z_dim]
        cursor += z_dim
        flat_means = flattened_transition[:, cursor : cursor + k * action_dim]
        action_means = flat_means.reshape(-1, k, action_dim)
        cursor += k * action_dim
        component_logits = flattened_transition[:, cursor : cursor + k]

        return cls(
            obs=obs,
            zs=zs,
            action_means=action_means,
            component_logits=component_logits,
        )

    @classmethod
    def init_dummy(
        cls,
        observation_dim: int,
        action_dim: int,
        z_dim: int,
        num_components: int,
    ) -> GMMDistillationTransition:
        """Create a dummy instance to be used as a shape reference."""
        return cls(
            obs=jnp.zeros(shape=(1, observation_dim)),
            zs=jnp.zeros(shape=(1, z_dim)),
            action_means=jnp.zeros(shape=(1, num_components, action_dim)),
            component_logits=jnp.zeros(shape=(1, num_components)),
        )

    def shuffle(self, key: RNGKey) -> GMMDistillationTransition:
        """Randomly permute transitions along the batch axis."""
        flattened = self.flatten()
        index = jax.random.permutation(key, flattened.shape[0])
        return self.from_flatten(flattened[index], self)


# ---------------------------------------------------------------------------


class BCTransition(flax.struct.PyTreeNode):
    """Simple behavior-cloning buffer.

    Records (obs, z, action) tuples collected from any policy for standard
    imitation learning — clone the action distribution without needing the
    teacher's internal GMM structure.

    Shapes (with batch prefix `...`):
        obs     (..., obs_dim)
        zs      (..., z_dim)
        actions (..., action_dim)
    """

    obs: Observation
    zs: jax.Array
    actions: Action

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
    def flatten_dim(self) -> int:
        return self.observation_dim + self.z_dim + self.action_dim

    # ------------------------------------------------------------------
    # Flatten / unflatten
    # ------------------------------------------------------------------

    def flatten(self) -> jax.Array:
        """Concatenate all fields into a 2-D array of shape (N, flatten_dim)."""
        return jnp.concatenate(
            [self.obs, self.zs, self.actions], axis=-1
        ).reshape(-1, self.flatten_dim)

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jax.Array,
        transition: BCTransition,
    ) -> BCTransition:
        """Reconstruct a BCTransition from a flattened array.

        Args:
            flattened_transition: shape (batch_size, flatten_dim)
            transition: a reference instance carrying dimension metadata.
        """
        obs_dim = transition.observation_dim
        z_dim = transition.z_dim

        obs = flattened_transition[:, :obs_dim]
        zs = flattened_transition[:, obs_dim : obs_dim + z_dim]
        actions = flattened_transition[:, obs_dim + z_dim :]

        return cls(obs=obs, zs=zs, actions=actions)

    @classmethod
    def init_dummy(
        cls,
        observation_dim: int,
        action_dim: int,
        z_dim: int,
    ) -> BCTransition:
        """Create a dummy instance to be used as a shape reference."""
        return cls(
            obs=jnp.zeros(shape=(1, observation_dim)),
            zs=jnp.zeros(shape=(1, z_dim)),
            actions=jnp.zeros(shape=(1, action_dim)),
        )

    def shuffle(self, key: RNGKey) -> BCTransition:
        """Randomly permute transitions along the batch axis."""
        flattened = self.flatten()
        index = jax.random.permutation(key, flattened.shape[0])
        return self.from_flatten(flattened[index], self)
