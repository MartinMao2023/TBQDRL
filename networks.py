""" Implements neural networks models that are commonly found in the RL literature."""

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                    name="Final_layer",
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden
    

class GCMLP(nn.Module):
    """
    Goal-conditioned MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, z], axis=-1)

        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


    


class ComplexGCMLP(nn.Module):
    """MLP module."""

    hidden_layer_sizes: Tuple[Tuple[int, ...], ...]
    output_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal()
    goal_kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal(0.01)
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        main_hidden = obs
        goal_hidden = z
        for hidden_size, goal_hidden_size in self.hidden_layer_sizes:
            main_candidate1 = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=False,
            )(main_hidden) # batch x hidden

            main_candidate2 = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=False,
            )(main_hidden) # batch x hidden

            scale = nn.Dense(
                hidden_size,
                kernel_init=self.goal_kernel_init,
                use_bias=self.bias,
            )(goal_hidden)
            scale = nn.tanh(scale)

            bias = nn.Dense(
                hidden_size,
                kernel_init=self.goal_kernel_init,
                use_bias=self.bias,
            )(goal_hidden)

            main_hidden = self.activation(
                main_candidate1 + main_candidate2 * scale + bias
                )
            goal_hidden = self.activation(
                nn.Dense(
                    goal_hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(goal_hidden)
                )

        if self.kernel_init_final is not None:
            kernel_init = self.kernel_init_final
        else:
            kernel_init = self.kernel_init

        main_hidden = nn.Dense(
            self.output_size,
            kernel_init=kernel_init,
            use_bias=self.bias,
        )(main_hidden)

        extra_bias = nn.Dense(
            self.output_size,
            kernel_init=kernel_init,
            use_bias=False,
        )(goal_hidden)

        main_hidden = main_hidden + extra_bias

        if self.final_activation is not None:
            main_hidden = self.final_activation(main_hidden)

        return main_hidden



class ComplexGCPPO_Policy(nn.Module):
    """MLP module."""

    hidden_layer_sizes: Tuple[Tuple[int, ...], ...]
    action_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal()
    learnable_std: bool = False
    goal_kernel_init: Callable[..., Any] = jax.nn.initializers.orthogonal(0.01)
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        main_hidden = obs
        goal_hidden = z
        for hidden_size, goal_hidden_size in self.hidden_layer_sizes:
            main_candidate1 = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=False,
            )(main_hidden) # batch x hidden

            main_candidate2 = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=False,
            )(main_hidden) # batch x hidden

            scale = nn.Dense(
                hidden_size,
                kernel_init=self.goal_kernel_init,
                use_bias=self.bias,
            )(goal_hidden)
            scale = nn.tanh(scale)

            bias = nn.Dense(
                hidden_size,
                kernel_init=self.goal_kernel_init,
                use_bias=self.bias,
            )(goal_hidden)

            main_hidden = self.activation(
                main_candidate1 + main_candidate2 * scale + bias
                )
            goal_hidden = self.activation(
                nn.Dense(
                    goal_hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(goal_hidden)
                )

        if self.kernel_init_final is not None:
            kernel_init = self.kernel_init_final
        else:
            kernel_init = self.kernel_init

        main_hidden = nn.Dense(
            self.action_dim,
            kernel_init=kernel_init,
            use_bias=self.bias,
        )(main_hidden)

        extra_bias = nn.Dense(
            self.action_dim,
            kernel_init=kernel_init,
            use_bias=False,
        )(goal_hidden)

        action_mean = main_hidden + extra_bias

        if self.final_activation is not None:
            action_mean = self.final_activation(action_mean)

        if self.learnable_std:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(0.0), 
                (self.action_dim,)
            )
        else:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(-2.0), 
                (self.action_dim,)
            )
        return action_mean, std_logits




class PPO_Policy(nn.Module):
    """PPO policy module."""

    hidden_layer_sizes: Tuple[int, ...]
    action_dim: int
    initial_std: jnp.ndarray
    learnable_std: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        for hidden_size in self.hidden_layer_sizes:
            hidden = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            hidden = self.activation(hidden)  # type: ignore

        if self.kernel_init_final is not None:
            kernel_init = self.kernel_init_final
        else:
            kernel_init = self.kernel_init

        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=kernel_init,
            use_bias=self.bias,
        )(hidden)

        if self.final_activation is not None:
            action_mean = self.final_activation(action_mean)

        if self.learnable_std:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(0.0), 
                (self.action_dim,)
            )
        else:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(-2.0), 
                (self.action_dim,)
            )
        return action_mean, std_logits



class GC_PPO_Policy(nn.Module):
    """
    Goal-conditioned PPO policy module.
    """

    hidden_layer_sizes: Tuple[int, ...]
    action_dim: int
    initial_std: jnp.ndarray
    learnable_std: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, z], axis=-1)
        for hidden_size in self.hidden_layer_sizes:
            hidden = nn.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            hidden = self.activation(hidden)  # type: ignore

        if self.kernel_init_final is not None:
            kernel_init = self.kernel_init_final
        else:
            kernel_init = self.kernel_init

        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=kernel_init,
            use_bias=self.bias,
        )(hidden)

        if self.final_activation is not None:
            action_mean = self.final_activation(action_mean)

        if self.learnable_std:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(0.0), 
                (self.action_dim,)
            )
        else:
            std_logits = self.param(
                'std_logits', 
                nn.initializers.constant(-2.0), 
                (self.action_dim,)
            )
        return action_mean, std_logits



class MLPTC(nn.Module):
    """Trajectory-Conditioned MLP module."""

    layer_sizes: Tuple[int, ...]
    offset_sigma: float
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, offset: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, 10*jnp.tanh(offset/self.offset_sigma*0.1), z], axis=-1)
        # hidden = jnp.concatenate([obs, offset, z], axis=-1)
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden



# TD3 networks
class QModule(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class QModuleTC(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    offset_sigma: float
    n_critics: int = 2
    activation: Callable = nn.relu
    final_activation: Callable = None

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, actions: jnp.ndarray, offset: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLPTC(
                layer_sizes=self.hidden_layer_sizes + (1,),
                offset_sigma=self.offset_sigma,
                activation=self.activation,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                final_activation=self.final_activation,
            )(hidden, offset, z)
            res.append(q)
        return jnp.concatenate(res, axis=-1)



