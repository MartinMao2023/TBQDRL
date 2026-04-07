from __future__ import annotations
import flax
import jax
import jax.numpy as jnp
from .transitions import Transition
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


class QDTransitionInfo(flax.struct.PyTreeNode):
    """Stores transition information."""
    reward: Reward
    fitness_reward: Reward
    done: Done
    truncation: jnp.ndarray  # Indicates if an episode has reached max time step



class QDPPOTransition(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    actions: Action
    zs: jnp.ndarray # task / history embedding
    log_likelihood: jnp.ndarray # excluding log(2*pi)
    rewards: jnp.ndarray
    fitness_rewards: jnp.ndarray
    td_lambda_returns: jnp.ndarray
    fitness_td_lambda_returns: jnp.ndarray
    gaes: jnp.ndarray
    fitness_gaes: jnp.ndarray
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
        flatten_dim = self.observation_dim + self.action_dim + self.z_dim + 10
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
                self.fitness_rewards,
                self.td_lambda_returns,
                self.fitness_td_lambda_returns,
                self.gaes,
                self.fitness_gaes,
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
        transition: QDPPOTransition,
    ) -> QDPPOTransition:
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
        fitness_rewards = flattened_transition[:, obs_dim + action_dim + task_dim + 2: 
                                               obs_dim + action_dim + task_dim + 3]
        td_lambda_returns = flattened_transition[:, obs_dim + action_dim + task_dim + 3: 
                                                 obs_dim + action_dim + task_dim + 4]
        fitness_td_lambda_returns = flattened_transition[:, obs_dim + action_dim + task_dim + 4: 
                                                         obs_dim + action_dim + task_dim + 5]
        gaes = flattened_transition[:, obs_dim + action_dim + task_dim + 5: 
                                    obs_dim + action_dim + task_dim + 6]
        fitness_gaes = flattened_transition[:, -4: -3]
        dones = flattened_transition[:, -3: -2]
        truncations = flattened_transition[:, -2 : -1]
        weights = flattened_transition[:, -1:]


        return cls(
            obs=obs,
            actions=actions,
            zs=zs,
            log_likelihood=log_likelihood,
            rewards=rewards,
            fitness_rewards=fitness_rewards,
            td_lambda_returns=td_lambda_returns,
            fitness_td_lambda_returns=fitness_td_lambda_returns,
            gaes=gaes,
            fitness_gaes=fitness_gaes,
            dones=dones,
            truncations=truncations,
            weights=weights,
        )

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int, z_dim: int) -> QDPPOTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = QDPPOTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            actions=jnp.zeros(shape=(1, action_dim)),
            action_noises=jnp.zeros(shape=(1, action_dim)),
            action_stds=jnp.zeros(shape=(1, action_dim)),
            zs=jnp.zeros(shape=(1, z_dim)),
            rewards=jnp.zeros(shape=(1, 1)),
            fitness_rewards=jnp.zeros(shape=(1, 1)),
            td_lambda_returns=jnp.zeros(shape=(1, 1)),
            fitness_td_lambda_returns=jnp.zeros(shape=(1, 1)),
            gaes=jnp.zeros(shape=(1, 1)),
            fitness_gaes=jnp.zeros(shape=(1, 1)),
            dones=jnp.zeros(shape=(1, 1)),
            truncations=jnp.zeros(shape=(1, 1)),
            weights=jnp.zeros(shape=(1, 1)),
        )

        return dummy_transition
    

    def shuffle(self, key: RNGKey) -> QDPPOTransition:
        flattened_transitions = self.flatten()
        num_transitions = flattened_transitions.shape[0]
        index = jax.random.permutation(key, num_transitions)
        transitions = self.from_flatten(flattened_transitions[index], self)
        
        return transitions



class QDTransition(Transition):
    """Stores data corresponding to a transition collected by a QD algorithm."""

    state_desc: StateDescriptor
    next_state_desc: StateDescriptor

    @property
    def state_descriptor_dim(self) -> int:
        """
        Returns:
            the dimension of the state descriptors.

        """
        return self.state_desc.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: QDTransition,
    ) -> QDTransition:
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
        desc_dim = transition.state_descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (2 * obs_dim + 3 + action_dim + desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

    @classmethod
    def init_dummy(  # type: ignore
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> QDTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = QDTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition


class DCRLTransition(QDTransition):
    """Stores data corresponding to a transition collected by a QD algorithm."""

    desc: Descriptor
    desc_prime: Descriptor

    @property
    def descriptor_dim(self) -> int:
        """
        Returns:
            the dimension of the descriptors.
        """
        return self.state_desc.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.
        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
            + 2 * self.descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
                self.desc,
                self.desc_prime,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: DCRLTransition,
    ) -> DCRLTransition:
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
        state_desc_dim = transition.state_descriptor_dim
        desc_dim = transition.descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (
                2 * obs_dim + 3 + action_dim + state_desc_dim
            ),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + state_desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim
            ),
        ]
        desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + 2 * state_desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + desc_dim
            ),
        ]
        desc_prime = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            desc=desc,
            desc_prime=desc_prime,
        )

    @classmethod
    def init_dummy(  # type: ignore
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> DCRLTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.
        Args:
            observation_dim: observation dimension
            action_dim: action dimension
        Returns:
            a dummy transition
        """
        dummy_transition = DCRLTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            desc=jnp.zeros(shape=(1, descriptor_dim)),
            desc_prime=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition

