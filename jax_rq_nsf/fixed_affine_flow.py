"""A learned flow followed by a fixed covariance-derived output transform."""

from __future__ import annotations

import flax.linen as nn
from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp

NF_ARCHITECTURE = "fixed_affine_flow_v1"


class FixedAffineState(PyTreeNode):
    """Replaceable, non-trainable affine data passed as dynamic PyTree leaves.

    Update all three values together, using ``from_numpy`` to recompute a
    consistent inverse and determinant, or ``replace`` for prepared arrays.
    """

    matrix: jax.Array
    inverse_matrix: jax.Array
    log_abs_det: jax.Array

    @classmethod
    def from_numpy(cls, matrix, *, inverse_matrix=None, log_abs_det=None):
        """Compute/validate the affine in host float64, then store float32."""
        import numpy as np

        matrix = np.asarray(matrix, dtype=np.float64)
        if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
                or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
            raise ValueError("Fixed affine matrix must be finite, square, and nonempty")
        sign, actual_logdet = np.linalg.slogdet(matrix)
        if sign == 0 or not np.isfinite(actual_logdet):
            raise ValueError("Fixed affine matrix must be invertible")
        inverse = (np.linalg.solve(matrix, np.eye(len(matrix))) if inverse_matrix is None
                   else np.asarray(inverse_matrix, dtype=np.float64))
        if (inverse.shape != matrix.shape or not np.all(np.isfinite(inverse))
                or not np.allclose(matrix @ inverse, np.eye(len(matrix)), rtol=1e-8, atol=1e-8)):
            raise ValueError("Fixed affine matrix and inverse are inconsistent")
        if log_abs_det is None:
            log_abs_det = actual_logdet
        if (not np.isfinite(log_abs_det)
                or not np.isclose(actual_logdet, log_abs_det, rtol=1e-9, atol=1e-9)):
            raise ValueError("Fixed affine log determinant is inconsistent with its matrix")
        return cls(
            matrix=jnp.asarray(matrix, dtype=jnp.float32),
            inverse_matrix=jnp.asarray(inverse, dtype=jnp.float32),
            log_abs_det=jnp.asarray(log_abs_det, dtype=jnp.float32),
        )


class FixedAffineFlow(nn.Module):
    """Append a replaceable, non-trainable affine to an NSF.

    ``variables['fixed_affine']['state']`` must contain a FixedAffineState.
    The matrix, inverse, and log|det(T)| are runtime PyTree leaves rather than
    static constructor fields. Replace that state to update the transformation
    while retaining the NSF parameters and checkpointing both together.
    ``log_prob`` accepts physical samples and includes the fixed Jacobian.
    For STL use ``sample_with_core`` and ``log_prob_core`` directly, so the
    entropy calculation never transforms and then numerically inverts T.
    """

    core: nn.Module

    @property
    def config(self):
        return self.core.config

    def setup(self):
        dimension = self.config.dimension
        self._affine = self.get_variable("fixed_affine", "state")
        if not isinstance(self._affine, FixedAffineState):
            raise ValueError("A FixedAffineState is required in variables['fixed_affine']['state']")
        if (self._affine.matrix.shape != (dimension, dimension)
                or self._affine.inverse_matrix.shape != (dimension, dimension)
                or self._affine.log_abs_det.shape != ()):
            raise ValueError("Fixed affine matrices and log determinant must match the flow dimension")

    def _check_shape(self, values):
        if values.ndim < 1 or values.shape[-1] != self.config.dimension:
            raise ValueError(f"Expected final dimension {self.config.dimension}, got {values.shape}")

    def to_physical(self, core_samples):
        self._check_shape(core_samples)
        return core_samples @ self._affine.matrix.T

    def to_core(self, samples):
        self._check_shape(samples)
        return samples @ self._affine.inverse_matrix.T

    def entropy_offset(self):
        """H(physical) = H(core) + this constant; KL = core_objective - it."""
        return self._affine.log_abs_det

    def forward(self, base_samples, *, return_boundary_loss=False):
        result = self.core.forward(base_samples, return_boundary_loss=return_boundary_loss)
        if return_boundary_loss:
            core_samples, boundary = result
            return self.to_physical(core_samples), boundary
        return self.to_physical(result)

    def inverse(self, samples, *, return_boundary_loss=False):
        result = self.core.inverse(self.to_core(samples), return_boundary_loss=return_boundary_loss)
        if return_boundary_loss:
            base, log_det, boundary = result
            return base, log_det - self._affine.log_abs_det, boundary
        base, log_det = result
        return base, log_det - self._affine.log_abs_det

    def base_log_prob(self, base_samples, *, include_constant=False):
        return self.core.base_log_prob(base_samples, include_constant=include_constant)

    def log_prob_core(self, core_samples, *, include_constant=False, return_boundary_loss=False):
        return self.core.log_prob(core_samples, include_constant=include_constant,
                                  return_boundary_loss=return_boundary_loss)

    def log_prob(self, samples, *, include_constant=False, return_boundary_loss=False):
        result = self.log_prob_core(self.to_core(samples), include_constant=include_constant,
                                    return_boundary_loss=return_boundary_loss)
        if return_boundary_loss:
            log_prob, boundary = result
            return log_prob - self._affine.log_abs_det, boundary
        return result - self._affine.log_abs_det

    def __call__(self, samples, *, include_constant=False, return_boundary_loss=False):
        return self.log_prob(samples, include_constant=include_constant,
                             return_boundary_loss=return_boundary_loss)

    def sample_with_core(self, key, num_samples, *, return_boundary_loss=False):
        result = self.core.sample(key, num_samples, return_boundary_loss=return_boundary_loss)
        if return_boundary_loss:
            core_samples, boundary = result
            return self.to_physical(core_samples), core_samples, boundary
        return self.to_physical(result), result

    def sample(self, key, num_samples, *, return_boundary_loss=False):
        result = self.sample_with_core(key, num_samples, return_boundary_loss=return_boundary_loss)
        if return_boundary_loss:
            physical, _, boundary = result
            return physical, boundary
        return result[0]


def init_flow(key: jax.Array, core: nn.Module, affine: FixedAffineState):
    """Initialize NSF parameters with a compulsory affine PyTree.

    Core parameters retain their identity/near-identity initialization: the
    reference moments already belong to the fixed outer transformation.
    Only the params collection is initialized; affine state is supplied,
    stored separately, and can later be replaced without rebuilding the flow.
    """
    if not isinstance(affine, FixedAffineState):
        raise TypeError("affine must be a FixedAffineState")
    flow = FixedAffineFlow(core)
    fixed_variables = {"fixed_affine": {"state": affine}}
    _, initialized = flow.apply(
        fixed_variables, jnp.zeros((1, core.config.dimension), dtype=jnp.float32),
        rngs={"params": key}, mutable=["params"],
    )
    return flow, {**fixed_variables, **initialized}


__all__ = ["FixedAffineState", "FixedAffineFlow", "NF_ARCHITECTURE", "init_flow"]
