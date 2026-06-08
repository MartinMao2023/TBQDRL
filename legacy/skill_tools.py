import jax
import jax.numpy as jnp
import flax.linen as nn
from brax import envs
from typing import Any, Tuple, List
from custom_types import Params, RNGKey, Env, EnvState
from flax import serialization
from networks import GCMLP, GC_PPO_Policy, ComplexGCPPO_Policy
from trajectory_encoder import TaskState, CircularEncoder, LineEncoder
from buffer import PPOTransition


def evaluate_deviation(offset: jnp.ndarray):
    squared_distance = jnp.sum(offset**2)
    reward = jnp.exp(-squared_distance*0.5)
    fail = jnp.where(squared_distance > 8, 1.0, 0.0)

    return reward, fail



def build_line_task_state(current_v, angle, key, speed):

    v = speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    deviation = 0.5 * (v - current_v)

    task_state = TaskState(
        deviation=deviation,
        task=v,
        t=0.0,
        z=jnp.concatenate([
            deviation,
            v,
        ]),
        key=key,
        r=1.0
    )

    return task_state


def build_circle_task_state(current_v, r, key, speed):

    v = jnp.array([-speed, 0]) # (2,)
    deviation = 0.5 * (v - current_v)
    key, subkey = jax.random.split(key)
    omega = speed / r

    sign_positive = jax.random.bernoulli(subkey, 0.5)
    omega = jnp.where(sign_positive, omega, -omega)

    task = jnp.concatenate([
        v,
        jnp.array([omega,])
        ])

    task_state = TaskState(
        deviation=deviation,
        task=task,
        t=0.0,
        z=jnp.concatenate([
            deviation,
            task,
        ]),
        key=key,
        r=1.0,
    )

    return task_state


def calculate_return(rewards: jnp.ndarray, dones: jnp.ndarray):

    def scan_calculate_return(
        last_value: jnp.ndarray, 
        data: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        reward, done = data

        new_value = last_value * (1 - done) + reward

        return new_value, None
    
    values, _ = jax.lax.scan(
        scan_calculate_return,
        jnp.zeros(rewards.shape[1:]), # (batch x 1)
        (rewards, dones),
        reverse=True,
    )

    return values




