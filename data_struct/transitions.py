from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from custom_types import (
    Action,
    Descriptor,
    Done,
    Observation,
    Reward,
    RNGKey,
    StateDescriptor,
)

def shuffle_transitions(key: RNGKey, transitions: PPOTransition) -> PPOTransition:
    flattened_transitions = transitions.flatten()
    num_transitions = flattened_transitions.shape[0]
    index = jax.random.permutation(key, num_transitions)
    transitions = transitions.__class__.from_flatten(flattened_transitions[index], transitions)
    
    return transitions


class TransitionInfo(flax.struct.PyTreeNode):
    """Stores transition information."""
    reward: Reward
    done: Done
    truncation: jnp.ndarray  # Indicates if an episode has reached max time step



class Transition(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    actions: Action
    next_obs: Observation
    rewards: Reward
    dones: Done
    truncations: jnp.ndarray  # Indicates if an episode has reached max time step


    @property
    def observation_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.obs.shape[-1]  # type: ignore

    @property
    def action_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.actions.shape[-1]  # type: ignore
    

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = 2 * self.observation_dim + self.action_dim + 3
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.actions,
                self.next_obs,
                self.rewards,
                self.dones,
                self.truncations,
            ],
            axis=-1,
        )
        # return flatten_transition
        return flatten_transition.reshape(-1, self.flatten_dim)

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: Transition,
    ) -> Transition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim

        obs = flattened_transition[:, :obs_dim]
        actions = flattened_transition[:, obs_dim: (obs_dim + action_dim)]
        next_obs = flattened_transition[:, (obs_dim + action_dim): (2 * obs_dim + action_dim)]
        rewards = flattened_transition[:, (2 * obs_dim + action_dim): (2 * obs_dim + action_dim + 1)]
        dones = flattened_transition[:, -2: -1]
        truncations = flattened_transition[:, -1:]

        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
        )
    

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int) -> Transition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = Transition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            actions=jnp.zeros(shape=(1, action_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1, 1)),
            dones=jnp.zeros(shape=(1, 1)),
            truncations=jnp.zeros(shape=(1, 1)),
        )
        return dummy_transition
    



class PPOTransition(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    actions: Action
    zs: jnp.ndarray # task / history embedding
    log_likelihood: jnp.ndarray # excluding log(2*pi)
    rewards: jnp.ndarray
    td_lambda_returns: jnp.ndarray
    gaes: jnp.ndarray
    dones: Done
    truncations: jnp.ndarray  # Indicates if an episode has reached max time step
    weights: jnp.ndarray  # weight resulting from truncation 

    @property
    def observation_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.obs.shape[-1]  # type: ignore

    @property
    def action_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.actions.shape[-1]  # type: ignore
    

    @property
    def z_dim(self) -> int:
        """
        Returns:
            the dimension of the task embedding
        """
        return self.zs.shape[-1]  # type: ignore


    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = self.observation_dim + self.action_dim + self.z_dim + 7
        return flatten_dim


    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.actions,
                self.zs,
                self.log_likelihood,
                self.rewards,
                self.td_lambda_returns,
                self.gaes,
                self.dones,
                self.truncations,
                self.weights,
            ],
            axis=-1,
        )
        return flatten_transition.reshape(-1, self.flatten_dim)
    

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: PPOTransition,
    ) -> PPOTransition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        task_dim = transition.z_dim

        obs = flattened_transition[:, :obs_dim]
        actions = flattened_transition[:, obs_dim : obs_dim + action_dim]
        zs = flattened_transition[:, obs_dim + action_dim: 
                                     obs_dim + action_dim + task_dim]
        log_likelihood = flattened_transition[:, obs_dim + action_dim + task_dim: 
                                              obs_dim + action_dim + task_dim + 1]
        rewards = flattened_transition[:, obs_dim + action_dim + task_dim + 1: 
                                       obs_dim + action_dim + task_dim + 2]
        td_lambda_returns = flattened_transition[:, obs_dim + action_dim + task_dim + 2: 
                                                 obs_dim + action_dim + task_dim + 3]
        gaes = flattened_transition[:, obs_dim + action_dim + task_dim + 3: 
                                    obs_dim + action_dim + task_dim + 4]
        dones = flattened_transition[:, -3: -2]
        truncations = flattened_transition[:, -2 : -1]
        weights = flattened_transition[:, -1:]


        return cls(
            obs=obs,
            actions=actions,
            zs=zs,
            log_likelihood=log_likelihood,
            rewards=rewards,
            td_lambda_returns=td_lambda_returns,
            gaes=gaes,
            dones=dones,
            truncations=truncations,
            weights=weights,
        )

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int, z_dim: int) -> PPOTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = PPOTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            actions=jnp.zeros(shape=(1, action_dim)),
            action_noises=jnp.zeros(shape=(1, action_dim)),
            action_stds=jnp.zeros(shape=(1, action_dim)),
            zs=jnp.zeros(shape=(1, z_dim)),
            rewards=jnp.zeros(shape=(1, 1)),
            td_lambda_returns=jnp.zeros(shape=(1, 1)),
            gaes=jnp.zeros(shape=(1, 1)),
            dones=jnp.zeros(shape=(1, 1)),
            truncations=jnp.zeros(shape=(1, 1)),
            weights=jnp.zeros(shape=(1, 1)),
        )

        return dummy_transition
    

    def shuffle(self, key: RNGKey) -> PPOTransition:
        flattened_transitions = self.flatten()
        num_transitions = flattened_transitions.shape[0]
        index = jax.random.permutation(key, num_transitions)
        transitions = self.from_flatten(flattened_transitions[index], self)
        
        return transitions







