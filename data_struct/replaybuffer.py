from __future__ import annotations
import flax
import jax
import jax.numpy as jnp
from transitions import Transition
from functools import partial
from typing import Tuple

from custom_types import (
    Action,
    Descriptor,
    Done,
    Observation,
    Reward,
    RNGKey,
    StateDescriptor,
)



class ReplayBuffer(flax.struct.PyTreeNode):
    """
    A replay buffer where transitions are flattened before being stored.
    Transitions are unflatenned on the fly when sampled in the buffer.
    data shape: (buffer_size, transition_concat_shape)
    """

    data: jnp.ndarray
    buffer_size: int = flax.struct.field(pytree_node=False)
    occupation: jnp.ndarray
    transition: Transition


    @classmethod
    def init(
        cls,
        buffer_size: int,
        transition: Transition,
    ) -> ReplayBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.

        Args:
            buffer_size: the size of the replay buffer, e.g. 1e6
            transition: a transition object (might be a dummy one) to get
                the dimensions right
        """
        flatten_dim = transition.flatten_dim
        data = jnp.zeros((buffer_size, flatten_dim), dtype=jnp.float32)
        occupation = jnp.zeros(buffer_size, dtype=int)

        return cls(
            data=data,
            buffer_size=buffer_size,
            occupation=occupation,
            transition=transition)


    @partial(jax.jit, static_argnames=("sample_size",))
    def sample(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Transition, RNGKey]:
        """
        Sample a batch of transitions in the replay buffer.
        """
        random_key, subkey = jax.random.split(random_key)

        indices = jax.random.choice(
            subkey, 
            self.buffer_size, 
            shape=(sample_size,), 
            p=self.occupation/jnp.sum(self.occupation)
            )
        
        samples = jnp.take(self.data, indices, axis=0, mode="clip")
        transitions = self.transition.__class__.from_flatten(samples, self.transition)

        return transitions, random_key
    

    @jax.jit
    def insert(self, transitions: Transition) -> ReplayBuffer:
        """
        Insert a batch of transitions in the replay buffer. The transitions are
        flattened before insertion.

        Args:
            transitions: A transition object in which each field is assumed to have
                a shape (batch_size, field_dim).
        """
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1])
        )
        num_transitions = flattened_transitions.shape[0]

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > self.buffer_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {num_transitions}, "
                f"max replay size {self.buffer_size}")

        # roll the data to avoid overlap
        new_data = jnp.roll(self.data, num_transitions, axis=0)
        new_occupation = jnp.roll(self.occupation, num_transitions, axis=0)

        # replace old data by the new one
        new_data = jax.lax.dynamic_update_slice_in_dim(
            new_data,
            flattened_transitions,
            start_index=0,
            axis=0)
        
        new_occupation = jax.lax.dynamic_update_slice_in_dim(
            new_occupation,
            jnp.ones(num_transitions, dtype=int),
            start_index=0,
            axis=0)

        # update the replay buffer
        replay_buffer = self.replace(
            data=new_data,
            occupation=new_occupation)

        return replay_buffer  # type: ignore
    

    @jax.jit
    def random_insert(self, key: RNGKey, transitions: Transition
                      ) -> Tuple[ReplayBuffer, RNGKey]:
        """
        Insert a batch of transitions in the replay buffer. The transitions are
        flattened before insertion.

        Args:
            transitions: A transition object in which each field is assumed to have
                a shape (batch_size, field_dim).
        """
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1]))
        num_transitions = flattened_transitions.shape[0]

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > self.buffer_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {num_transitions}, "
                f"max replay size {self.buffer_size}")
        
        key, subkey = jax.random.split(key)

        # prioritise on filling empty slots
        selection_probability = jax.lax.cond(
            jnp.sum(1 - self.occupation) > num_transitions,
            lambda x: (1 - self.occupation) / jnp.sum(1 - self.occupation),
            lambda x: jnp.ones(self.buffer_size) / self.buffer_size,
            None)

        indices = jax.random.choice(
            subkey, 
            self.buffer_size, 
            shape=(num_transitions,), 
            p=selection_probability, 
            replace=False)
        
        new_data = self.data.at[indices].set(flattened_transitions)
        new_occupation = self.occupation.at[indices].set(1)

        # update the replay buffer
        replay_buffer = self.replace(
            data=new_data,
            occupation=new_occupation)

        return replay_buffer, key  # type: ignore



