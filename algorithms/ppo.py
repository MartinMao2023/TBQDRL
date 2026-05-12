from flax.struct import dataclass

from functools import partial
from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import optax
# from optax.losses import sigmoid_binary_cross_entropy
from jax import numpy as jnp

from data_struct import PPOTransition
from data_struct.states import GeneralizedState
from networks import PPO_Policy
from custom_types import Params, RNGKey
from flax.struct import PyTreeNode
from task_wrappers.base import BaseTaskWrapper



@dataclass
class PPOConfigs:
    policy_learnng_rate_per_std: float = 1e-3 # learning_rate per std
    critic_learning_rate: float = 5e-4
    clip_ratio: float = 0.2
    entropy_gain: float = 0.01
    discount: float = 0.99
    td_lambda_discount: float = 0.95
    rollout_length: int = 64
    vec_env: int = 256
    mini_batch_size: int = 1024
    critic_epochs: int = 4
    policy_epochs: int = 4




class  PPOTrainingState(PyTreeNode):
    """Contains training state for the learner."""

    policy_params: Params
    critic_params: Params

    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    current_std: jax.Array
    step_num: int



class RolloutMetrics(PyTreeNode):
    average_reward: float
    average_return: float
    average_lifespan: float


class TrainingMetrics(PyTreeNode):
    critic_error: float
    approx_kl: float
    clip_fraction: float


class AuxData(PyTreeNode):
    """Contains auxiliary information for monitoring"""

    rollout_data: RolloutMetrics
    training_data: TrainingMetrics
    



class PPO:
    def __init__(
        self,
        env: BaseTaskWrapper,
        policy_network: PPO_Policy,
        critic_network: nn.Module,
        ppo_configs: PPOConfigs,
        std_anneal_fn: Callable,
        ):

        self._env = env
        self.configs = ppo_configs
        self._std_anneal_fn = std_anneal_fn

        self.mini_batch_num = (
            ppo_configs.vec_env * ppo_configs.rollout_length
            ) // ppo_configs.mini_batch_size
        self.ema_alpha = jnp.exp(-2 / self.mini_batch_num)

        self._policy_network = policy_network
        self._critic_network = critic_network
        self._lr_per_std = ppo_configs.policy_learnng_rate_per_std

        if ppo_configs.clip_ratio > 0:
            self._clip_log_ratio = jnp.log(1 + ppo_configs.clip_ratio)
        else:
            raise(ValueError("invalid clip ratio"))

        def make_ppo_optimizer(learning_rate):
            return optax.adam(learning_rate=learning_rate)
        
        initial_std = std_anneal_fn(0)
        rms_std = jnp.sqrt(jnp.mean(initial_std**2))

        self._policy_optimizer = optax.inject_hyperparams(make_ppo_optimizer)(
            learning_rate=rms_std * self._lr_per_std)
        self._critic_optimizer = optax.adam(
            learning_rate=ppo_configs.critic_learning_rate,
        )

        if policy_network.learnable_std:
            std_fn = lambda x, y: nn.sigmoid(x)
        else:
            std_fn = lambda x, y: y * jnp.ones_like(x)

        
        @jax.jit
        def rollout_fn(
            policy_params: Params,
            starting_states: GeneralizedState,
            keys: RNGKey,
            std: jax.Array,
            ) -> Tuple[GeneralizedState, PPOTransition]:

            def play_step_fn(
                carry: Tuple[GeneralizedState, GeneralizedState, float, RNGKey],
                ) -> Tuple[Tuple, PPOTransition]:
                
                state, sampled_state, l, key = carry
                obs, z = env.get_obs(state)
                action_mean, std_logits = policy_network.apply(policy_params, obs, z)
                action_std = std_fn(std_logits, std)

                key, subkey = jax.random.split(key)
                candidate_action_noise = action_std * jax.random.normal(subkey, action_mean.shape)
                action = jnp.clip(action_mean + candidate_action_noise, -1.0, 1.0)
                log_likelihood = -jnp.sum(
                    jnp.log(action_std) + 0.5 * jnp.square(action - action_mean) / (action_std**2 + 1e-6), 
                    keepdims=True,
                    ) # shape of (1,)
                
                state, transition_info = env.step(state, action)
                l = l + 1 - state.env_state.done * l

                key, subkey = jax.random.split(key)
                if_replace = jax.random.uniform(subkey) < 1 / l
                sampled_state = jax.tree.map(
                    lambda x, y: jax.lax.select(if_replace, x, y), 
                    state, 
                    sampled_state
                )

                transition = PPOTransition(
                    obs=obs,
                    actions=action,
                    zs=z, 
                    log_likelihood=log_likelihood,
                    rewards=transition_info.reward,
                    td_lambda_returns=jnp.zeros((1,)),
                    gaes=jnp.zeros((1,)),
                    dones=transition_info.done,
                    truncations=transition_info.truncation,
                    weights=jnp.zeros((1,)),
                    )

                return (state, sampled_state, l, key), transition

            final_carry, transitions = jax.lax.scan(
                lambda x, _: jax.vmap(play_step_fn)(x),
                (starting_states, starting_states, jnp.zeros((ppo_configs.vec_env,)), keys),
                length=ppo_configs.rollout_length,
            )

            final_states, sampled_states = final_carry[:2]

            return final_states, sampled_states, transitions
        
        self._rollout_fn = rollout_fn

        
        def critic_loss_fn(
            critic_params: Params,
            transitions: PPOTransition,
        ) -> float:
            
            estimated_v = critic_network.apply(critic_params, transitions.obs, transitions.zs)
            weights = 1 / (1 + jnp.square(transitions.weights))
            loss = jnp.average(
                jnp.square(estimated_v - transitions.td_lambda_returns), 
                weights=weights)
                        
            return loss, jnp.sqrt(loss)


        self._critic_loss_fn = critic_loss_fn


        if policy_network.learnable_std:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
                std: jnp.ndarray,
            ) -> float:

                action_mean, std_logits = policy_network.apply(policy_params, transitions.obs, transitions.zs)
                entropy = jnp.sum(nn.log_sigmoid(std_logits), axis=-1, keepdims=True) # batch x 1
                new_log_likelihood = -0.5 * jnp.sum(
                    jnp.square(jnp.exp(-std_logits) + 1) * (jnp.square(action_mean - transitions.actions) + 1e-4), 
                    axis=-1, 
                    keepdims=True) - entropy # batch x 1
                log_ratio = new_log_likelihood - transitions.log_likelihood
                ratio = jnp.exp(log_ratio)

                gaes = transitions.gaes
                loss_cond = jax.lax.stop_gradient(log_ratio * gaes <= self._clip_log_ratio * jnp.abs(gaes))
                clip_fraction = 1 - jnp.mean(loss_cond)
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

                loss = jnp.mean((jnp.where(loss_cond, -gaes * ratio, 0.0) - ppo_configs.entropy_gain * entropy))

                return loss, (approx_kl, clip_fraction)
            
        else:
            def policy_loss_fn(
                policy_params: Params,
                transitions: PPOTransition,
                std: jnp.ndarray,
            ) -> float:
                
                action_mean, _ = policy_network.apply(policy_params, transitions.obs, transitions.zs)
                new_log_likelihood = -jnp.sum(
                    jnp.log(std) + 0.5 * jnp.square(action_mean - transitions.actions) / (std**2 + 1e-6), 
                    axis=-1, 
                    keepdims=True) # batch x 1
                log_ratio = new_log_likelihood - transitions.log_likelihood
                ratio = jnp.exp(log_ratio)

                gaes = transitions.gaes
                loss_cond = jax.lax.stop_gradient(log_ratio * gaes <= self._clip_log_ratio * jnp.abs(gaes))
                clip_fraction = 1 - jnp.mean(loss_cond)
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

                loss = jnp.mean(jnp.where(loss_cond, -gaes * ratio, 0.0))
                
                return loss, (approx_kl, clip_fraction)

        self._policy_loss_fn = policy_loss_fn

            
    def init(
        self, 
        key: RNGKey,
    ) -> PPOTrainingState:
        
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_zs = jnp.zeros(shape=(self._env.z_size,))

        key, subkey = jax.random.split(key)
        policy_params = self._policy_network.init(subkey, obs=fake_obs, z=fake_zs)
        policy_opt_state = self._policy_optimizer.init(policy_params)

        key, subkey = jax.random.split(key)
        critic_params = self._critic_network.init(subkey, obs=fake_obs, z=fake_zs)
        critic_opt_state = self._critic_optimizer.init(critic_params)


        initial_std = self._std_anneal_fn(0)

        
        training_state = PPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            current_std=jnp.array(initial_std),
            step_num=0,
            )

        return training_state


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def state_update(
        self, 
        training_state: PPOTrainingState, 
        transitions: PPOTransition, 
    ) -> Tuple[PPOTrainingState, TrainingMetrics]:
        """
        This function can now be Jit-complied.
        """

        (critic_params, critic_opt_state, final_critic_error), _ = jax.lax.scan(
            lambda x, _: partial(self.train_critic, transitions=transitions)(x),
            (training_state.critic_params, training_state.critic_opt_state, 0.0),
            length=self.configs.critic_epochs,
        )

        (policy_params, policy_opt_state, final_approx_kl, final_clip_fraction), _ = jax.lax.scan(
            lambda x, _: partial(
                self.train_policy, 
                transitions=transitions, 
                std=training_state.current_std,
                )(x),
            (training_state.policy_params, training_state.policy_opt_state, 0.0, 0.0),
            length=self.configs.policy_epochs,
        )

        # annealing
        step_num = training_state.step_num + 1
        current_std = self._std_anneal_fn(step_num)
        rms_std = jnp.sqrt(jnp.mean(current_std**2))

        current_learning_rate = rms_std * self._lr_per_std
        policy_opt_state = policy_opt_state._replace(
            hyperparams={**policy_opt_state.hyperparams, 'learning_rate': current_learning_rate}
        )
        
        new_training_state = PPOTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state,
            current_std=current_std,
            step_num=step_num,
        )
        training_data = TrainingMetrics(
            critic_error=final_critic_error,
            approx_kl=final_approx_kl,
            clip_fraction=final_clip_fraction,
            )

        return new_training_state, training_data


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_policy(
        self,
        carry: Tuple[Params, optax.OptState, float, float],
        transitions: PPOTransition,
        std: jnp.ndarray,
        ) -> Tuple[Tuple[Params, optax.OptState, float, float], Any]:
        """
        perform one epoch training of policy network
        """

        def scan_train_policy(carry, transition_data):
            (
                current_policy_params, 
                current_policy_opt_state,
                current_approx_kl,
                current_clip_fraction,
                ) = carry

            policy_gradient, (approx_kl, clip_fraction) = jax.grad(
                self._policy_loss_fn, has_aux=True
                )(current_policy_params, transition_data, std)
    
            new_approx_kl = approx_kl * (1 - self.ema_alpha) + \
                self.ema_alpha * current_approx_kl
            new_clip_fraction = clip_fraction * (1 - self.ema_alpha) + \
                self.ema_alpha * current_clip_fraction
            
            policy_updates, new_policy_opt_state = self._policy_optimizer.update(
                policy_gradient, current_policy_opt_state)
            new_policy_params = optax.apply_updates(current_policy_params, policy_updates)

            new_carry = (
                new_policy_params, 
                new_policy_opt_state,
                new_approx_kl,
                new_clip_fraction,
            )
            
            return new_carry
        
        def cond_scan_train_policy(carry, transition_data):
            approx_kl = carry[-2]
            new_carry = jax.lax.cond(
                # clip_fraction > 0.1,
                approx_kl > 0.0125,
                lambda x: x, # skip update
                lambda x: scan_train_policy(x, transition_data),
                carry
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            cond_scan_train_policy,
            carry,
            transitions,
        )
        
        return final_carry, None


    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train_critic(
        self,
        carry: Tuple[Params, optax.OptState, float],
        transitions: PPOTransition,
    ) -> Tuple[Tuple[Params, optax.OptState, float], Any]:
        """
        perform one epoch training of critic network
        """
        
        def scan_train_critic(carry, transition_data):
            (
                current_critic_params, 
                current_critic_opt_state, 
                current_critic_error,
                ) = carry

            critic_gradient, critic_error = jax.grad(self._critic_loss_fn, has_aux=True)(
                current_critic_params,
                transition_data,
                )
            
            new_critic_error = critic_error * (1 - self.ema_alpha) + self.ema_alpha * current_critic_error
            
            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_gradient, current_critic_opt_state)
            new_critic_params = optax.apply_updates(current_critic_params, critic_updates)
            
            return (new_critic_params, new_critic_opt_state, new_critic_error), None
        
        final_carry, _ = jax.lax.scan(
            scan_train_critic,
            carry,
            transitions,
        )
        
        return final_carry, None
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def calculate_v(
        self,
        critic_params: Params, 
        transitions: PPOTransition,
    ) -> jnp.ndarray:
        
        def scan_calculate_v(
            transition: PPOTransition,
        ) -> Tuple[None, jnp.ndarray]:
            
            v_value = self._critic_network.apply(
                critic_params, transition.obs, transition.zs
                )
            return None, v_value

        _, v_values = jax.lax.scan(
            lambda _, x: jax.vmap(scan_calculate_v)(x),
            None,
            transitions,
            )
        
        return v_values
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def _process_gaes(
        self,
        gaes: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flter the extreme values of GAEs and subtract the mean if needed
        """
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        mask = jnp.abs(gaes - gae_mean) < 3*gae_std
        corrected_mean = jnp.mean(gaes, where=mask)
        clipped_values = jnp.clip(gaes, corrected_mean - 3*gae_std, corrected_mean + 3*gae_std)
        corrected_std = jnp.std(clipped_values, ddof=1)
        gaes = jnp.clip(gaes, corrected_mean - 5*corrected_std, corrected_mean + 5*corrected_std)
        offset = jnp.clip(-jnp.mean(gaes), min=0.0)

        return (gaes + offset) / (corrected_std + 1e-6)
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def train(
        self,
        starting_states: GeneralizedState,
        training_state: PPOTrainingState,
        key: RNGKey,
    ) -> Tuple[Tuple[GeneralizedState, PPOTrainingState, jax.Array, RNGKey], AuxData]:
        """
        Perform one iteration of PPO update
        
        """

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=self.configs.vec_env)
        (
            final_states, sampled_states, transitions
            ) = self._rollout_fn(
                training_state.policy_params, 
                starting_states, 
                subkeys, 
                training_state.current_std,
                )
        final_obs, final_zs = self._env.get_obs(final_states)

        final_v = self._critic_network.apply(training_state.critic_params, final_obs, final_zs)
        v_values = self.calculate_v(training_state.critic_params, transitions)

        td_lambda_returns, weights = self.calculate_td_lambda_returns(
            final_v,
            v_values, 
            transitions.rewards,
            transitions.dones,
            transitions.truncations,
            ) # rollout x parallelize
        
        gaes = self._process_gaes(td_lambda_returns - v_values)

        transitions = transitions.replace(
            td_lambda_returns=td_lambda_returns,
            gaes=gaes,
            weights=weights,
            )
        rollout_data = self.evaluate_rollout(
            final_v,
            transitions,
        )
        
        key, subkey = jax.random.split(key)
        transitions = transitions.shuffle(subkey)
        transitions = jax.tree.map(
            lambda x: jnp.reshape(
                x,
                (
                    -1,
                    self.configs.mini_batch_size,
                    *x.shape[1:],
                ),
            ),
            transitions)
        
        new_training_state, training_data = self.state_update(training_state, transitions)
        aux_data = AuxData(rollout_data=rollout_data, training_data=training_data)

        return (final_states, sampled_states, new_training_state, key), aux_data



    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def calculate_td_lambda_returns(
        self,
        final_v_value: jnp.ndarray,
        v_values: jnp.ndarray, 
        rewards: jnp.ndarray,
        termination: jnp.ndarray, 
        truncation: jnp.ndarray,
    ) -> jnp.ndarray:
        
        discount = self.configs.discount
        td_lambda_discount = self.configs.td_lambda_discount

        def scan_calculate_td_lambda(
            carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], 
            data: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            
            (last_td_lambda_value, last_value, last_weight) = carry
            reward, v_value, done, truncate = data
            current_td_lambda_value = reward + (1 - done) * discount * (
                    (1 - td_lambda_discount) * last_value + td_lambda_discount * last_td_lambda_value
                )
            current_td_lambda_value = jnp.where(truncate, v_value, current_td_lambda_value)
            weight = jnp.where(
                truncate > 0.5, 
                1.0, 
                (1 - done) * discount * (1 + (last_weight - 1) * td_lambda_discount)
                )
            
            return (current_td_lambda_value, v_value, weight), (current_td_lambda_value, weight)
            
        _, (td_lambda_values, weights) = jax.lax.scan(
            jax.vmap(scan_calculate_td_lambda),
            (final_v_value, final_v_value, jnp.ones_like(final_v_value)),
            (rewards, v_values, termination, truncation),
            reverse=True,
        ) # length x batch x d

        return td_lambda_values, weights
    
    

    @partial(
        jax.jit, 
        static_argnames=("self",)
    )
    def evaluate_rollout(
        self,
        final_v: jax.Array,
        transitions: PPOTransition,
        ) -> RolloutMetrics:

        discount = self.configs.discount
        average_reward = jnp.mean(transitions.rewards)

        def scan_evaluation(
            carry: Tuple[jax.Array, jax.Array], 
            data: PPOTransition,
            ) -> Tuple:

            (v_value, lifespan) = carry
            new_v_value = data.rewards + (1 - data.dones) * discount * v_value
            new_v_value = jnp.where(data.truncations, data.td_lambda_returns, new_v_value)
            new_lifespan = 1 + (1 - data.dones) * lifespan

            new_carry = (new_v_value, new_lifespan)

            return new_carry, new_carry
        
        (
            initial_v_value, initial_lifespan
            ), (v_values, lifespans) = jax.lax.scan(
            scan_evaluation,
            (final_v, jnp.zeros_like(final_v)),
            transitions,
            reverse=True,
        )
        rollout_data = RolloutMetrics(
            average_reward=average_reward,
            average_return=jnp.mean(v_values),
            average_lifespan=jnp.mean(lifespans),
            )
        
        return rollout_data





