"""Sanity checks for GMMDistillationTransition and BCTransition."""

import jax
import jax.numpy as jnp

from data_struct.distillation_transitions import BCTransition, GMMDistillationTransition
from data_struct import GMMDistillationTransition as G, BCTransition as B  # noqa: F401  (re-export smoke test)


# ---------------------------------------------------------------------------
# GMMDistillationTransition
# ---------------------------------------------------------------------------

def test_gmm_init_dummy():
    d = GMMDistillationTransition.init_dummy(
        observation_dim=8, action_dim=4, z_dim=3, num_components=5
    )
    assert d.obs.shape == (1, 8)
    assert d.zs.shape == (1, 3)
    assert d.action_means.shape == (1, 5, 4)
    assert d.component_weights.shape == (1, 5)


def test_gmm_flatten_dim():
    d = GMMDistillationTransition.init_dummy(
        observation_dim=8, action_dim=4, z_dim=3, num_components=5
    )
    # flatten_dim = obs_dim + z_dim + k*action_dim + k = 8+3+20+5 = 36
    assert d.flatten_dim == 36
    flat = d.flatten()
    assert flat.shape == (1, 36)


def test_gmm_flatten_roundtrip():
    key = jax.random.PRNGKey(0)
    batch = 16
    obs = jax.random.normal(key, (batch, 8))
    zs = jax.random.normal(key, (batch, 3))
    action_means = jax.random.normal(key, (batch, 5, 4))
    component_weights = jax.nn.softmax(jax.random.normal(key, (batch, 5)), axis=-1)

    d = GMMDistillationTransition(
        obs=obs, zs=zs, action_means=action_means, component_weights=component_weights
    )
    dummy = GMMDistillationTransition.init_dummy(
        observation_dim=8, action_dim=4, z_dim=3, num_components=5
    )
    d2 = GMMDistillationTransition.from_flatten(d.flatten(), dummy)

    assert jnp.allclose(d2.obs, obs, atol=1e-6)
    assert jnp.allclose(d2.zs, zs, atol=1e-6)
    assert jnp.allclose(d2.action_means, action_means, atol=1e-6)
    assert jnp.allclose(d2.component_weights, component_weights, atol=1e-6)


def test_gmm_shuffle_preserves_content():
    key = jax.random.PRNGKey(42)
    batch = 32
    obs = jax.random.normal(key, (batch, 8))
    zs = jax.random.normal(key, (batch, 3))
    action_means = jax.random.normal(key, (batch, 5, 4))
    component_weights = jax.nn.softmax(jax.random.normal(key, (batch, 5)), axis=-1)

    d = GMMDistillationTransition(
        obs=obs, zs=zs, action_means=action_means, component_weights=component_weights
    )
    key2 = jax.random.PRNGKey(7)
    d_shuffled = d.shuffle(key2)

    assert d_shuffled.obs.shape == d.obs.shape
    assert d_shuffled.action_means.shape == d.action_means.shape
    assert d_shuffled.component_weights.shape == d.component_weights.shape

    # Rows are a permutation: every original obs row must appear exactly once.
    for i in range(batch):
        matches = jnp.all(jnp.isclose(d_shuffled.obs, obs[i], atol=1e-6), axis=-1)
        assert jnp.sum(matches) == 1, f"obs row {i} not found exactly once after shuffle"


def test_gmm_shuffle_changes_order():
    key = jax.random.PRNGKey(99)
    batch = 64
    obs = jax.random.normal(key, (batch, 8))
    d = GMMDistillationTransition(
        obs=obs,
        zs=jax.random.normal(key, (batch, 3)),
        action_means=jax.random.normal(key, (batch, 5, 4)),
        component_weights=jax.nn.softmax(jax.random.normal(key, (batch, 5)), axis=-1),
    )
    d_shuffled = d.shuffle(jax.random.PRNGKey(1))
    assert not jnp.allclose(d_shuffled.obs, obs), "shuffle left all rows in the same order"


# ---------------------------------------------------------------------------
# BCTransition
# ---------------------------------------------------------------------------

def test_bc_init_dummy():
    bc = BCTransition.init_dummy(observation_dim=8, action_dim=4, z_dim=3)
    assert bc.obs.shape == (1, 8)
    assert bc.zs.shape == (1, 3)
    assert bc.actions.shape == (1, 4)


def test_bc_flatten_dim():
    bc = BCTransition.init_dummy(observation_dim=8, action_dim=4, z_dim=3)
    # flatten_dim = 8 + 3 + 4 = 15
    assert bc.flatten_dim == 15
    flat = bc.flatten()
    assert flat.shape == (1, 15)


def test_bc_flatten_roundtrip():
    key = jax.random.PRNGKey(0)
    batch = 16
    obs = jax.random.normal(key, (batch, 8))
    zs = jax.random.normal(key, (batch, 3))
    actions = jax.random.normal(key, (batch, 4))

    bc = BCTransition(obs=obs, zs=zs, actions=actions)
    dummy = BCTransition.init_dummy(observation_dim=8, action_dim=4, z_dim=3)
    bc2 = BCTransition.from_flatten(bc.flatten(), dummy)

    assert jnp.allclose(bc2.obs, obs, atol=1e-6)
    assert jnp.allclose(bc2.zs, zs, atol=1e-6)
    assert jnp.allclose(bc2.actions, actions, atol=1e-6)


def test_bc_shuffle_preserves_content():
    key = jax.random.PRNGKey(42)
    batch = 32
    obs = jax.random.normal(key, (batch, 8))
    zs = jax.random.normal(key, (batch, 3))
    actions = jax.random.normal(key, (batch, 4))

    bc = BCTransition(obs=obs, zs=zs, actions=actions)
    bc_shuffled = bc.shuffle(jax.random.PRNGKey(7))

    assert bc_shuffled.obs.shape == bc.obs.shape
    assert bc_shuffled.actions.shape == bc.actions.shape

    for i in range(batch):
        matches = jnp.all(jnp.isclose(bc_shuffled.obs, obs[i], atol=1e-6), axis=-1)
        assert jnp.sum(matches) == 1, f"obs row {i} not found exactly once after shuffle"


def test_bc_shuffle_changes_order():
    key = jax.random.PRNGKey(99)
    batch = 64
    obs = jax.random.normal(key, (batch, 8))
    bc = BCTransition(
        obs=obs,
        zs=jax.random.normal(key, (batch, 3)),
        actions=jax.random.normal(key, (batch, 4)),
    )
    bc_shuffled = bc.shuffle(jax.random.PRNGKey(1))
    assert not jnp.allclose(bc_shuffled.obs, obs), "shuffle left all rows in the same order"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_gmm_init_dummy()
    test_gmm_flatten_dim()
    test_gmm_flatten_roundtrip()
    test_gmm_shuffle_preserves_content()
    test_gmm_shuffle_changes_order()
    test_bc_init_dummy()
    test_bc_flatten_dim()
    test_bc_flatten_roundtrip()
    test_bc_shuffle_preserves_content()
    test_bc_shuffle_changes_order()
    print("All tests passed.")
