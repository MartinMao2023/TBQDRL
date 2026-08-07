import jax
import flax.linen as nn
import jax.numpy as jnp
from brax import envs
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Any, Tuple
from algorithms.gmm_ppo import PPO, PPOConfigs, PPOTrainingState
from networks import GCMLP, GC_Student_PPO_Policy, GC_GMM_PPO_Policy
from flax import serialization
from task_wrappers.ant_wrapper import AntWrapper
from data_struct.states import GeneralizedState
from custom_types import RNGKey, Params


# ---------------------------------------------------------------------------
# Hyper-parameters  (mirrors main_ant.py)
# ---------------------------------------------------------------------------
vec_env        = 4096
mini_batch_size = 8192
num_iterations  = 2000
policy_epochs   = 4
critic_epochs   = 4
policy_learning_rate_per_std = 8e-4
critic_learning_rate         = 5e-4
rollout_length  = 32

description = {
    "task": "PPO fine-tuning from BC-distilled student",
    "policy": "GC_Student_PPO_Policy (distilled from GMM teacher)",
    "critic_init": "teacher GMM critic (output/GMM/critic.msgpack)",
    "policy_learning_rate": policy_learning_rate_per_std,
    "critic_learning_rate": critic_learning_rate,
    "vec_env": vec_env,
    "mini_batch_size": mini_batch_size,
    "rollout_length": rollout_length,
    "iterations": num_iterations,
    "policy_epochs": policy_epochs,
    "critic_epochs": critic_epochs,
}

wandb.init(
    entity="airl-lab",
    project="TBQDRL",
    config=description,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
env = AntWrapper(env)

# ---------------------------------------------------------------------------
# Network definitions — must match main_test_BC.py exactly so that the saved
# student parameters deserialise into the correct tree structure.
# ---------------------------------------------------------------------------
actor_hidden_layers:  Tuple[int, ...] = (128, 128)
critic_hidden_layers: Tuple[int, ...] = (64, 64)

# component_means: same seed as main_test_BC.py (624234)
seed = 624234
key_cm = jax.random.PRNGKey(seed)
key_cm, subkey_cm = jax.random.split(key_cm)
component_means = jnp.concatenate([
    jnp.zeros(env.action_size),
    jax.random.normal(subkey_cm, shape=(3 * env.action_size)) * 0.25,
])

policy_network = GC_GMM_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=4,
    component_means=component_means,   # structure-equivalent; values don't affect param tree
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.softplus,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
)

# ---------------------------------------------------------------------------
# PPO (same config as main_ant.py)
# ---------------------------------------------------------------------------
ppo_config = PPOConfigs(
    policy_learnng_rate_per_std=policy_learning_rate_per_std,
    critic_learning_rate=critic_learning_rate,
    clip_ratio=0.2,
    entropy_gain=0.001,
    discount=0.99,
    td_lambda_discount=0.95,
    rollout_length=rollout_length,
    vec_env=vec_env,
    mini_batch_size=mini_batch_size,
    critic_epochs=critic_epochs,
    policy_epochs=policy_epochs,
)

ppo = PPO(
    env=env,
    policy_network=policy_network,
    critic_network=critic_network,
    ppo_configs=ppo_config,
    std_anneal_fn=lambda x: 1.0,
)

# ---------------------------------------------------------------------------
# Initialise training state (gives us fresh optimiser states with zero
# momentum — correct starting point since the loaded params are new to Adam).
# ---------------------------------------------------------------------------
seed = 4242
random_key = jax.random.PRNGKey(seed)
random_key, subkey = jax.random.split(random_key)
ppo_training_state = ppo.init(subkey)

# ---------------------------------------------------------------------------
# Load student policy params
# Use the same init key / network definition so the pytree structure matches.
# ---------------------------------------------------------------------------
fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs  = jnp.zeros(shape=(env.z_size,))

random_key, subkey = jax.random.split(random_key)
student_template = policy_network.init(subkey, obs=fake_obs, z=fake_zs)
with open("output/GMM/student_policy.msgpack", "rb") as f:
    student_params = serialization.from_bytes(student_template, f.read())

# ---------------------------------------------------------------------------
# Load teacher GMM params to extract std_logits.
# The BC training did not copy std_logits into the student, so we do it here.
# (component_means don't affect the saved parameter structure, so the same
#  architecture with any component_means value produces a valid template.)
# ---------------------------------------------------------------------------
teacher_network = GC_GMM_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.softplus,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=4,
    component_means=component_means,   # structure-equivalent; values don't affect param tree
)
teacher_template = teacher_network.init(
    jax.random.PRNGKey(4242), obs=fake_obs, z=fake_zs
)
with open("output/GMM/policy.msgpack", "rb") as f:
    teacher_params = serialization.from_bytes(teacher_template, f.read())

# Transplant teacher's shared std_logits into student params.
# teacher_std_logits = teacher_params["params"]["std_logits"]
# student_params = {
#     **student_params,
#     "params": {
#         **student_params["params"],
#         "std_logits": teacher_std_logits,
#     },
# }

# ---------------------------------------------------------------------------
# Load teacher critic params
# ---------------------------------------------------------------------------
random_key, subkey = jax.random.split(random_key)
critic_template = critic_network.init(subkey, obs=fake_obs, z=fake_zs)
with open("output/GMM/critic.msgpack", "rb") as f:
    critic_params = serialization.from_bytes(critic_template, f.read())

# ---------------------------------------------------------------------------
# Patch training state: swap in loaded params and fix current_std so the
# first rollout uses the student's actual std (not the random-init default).
# ---------------------------------------------------------------------------
# student_current_std = jax.nn.sigmoid(teacher_std_logits)   # matches policy params

ppo_training_state = ppo_training_state.replace(
    # policy_params=student_params,
    policy_params=teacher_params,
    critic_params=critic_params,
    # current_std=student_current_std,
)

# print(f"Student std (from teacher std_logits): {student_current_std}")

# ---------------------------------------------------------------------------
# Env reset  (same seed as main_ant.py / main_ant_GMM.py)
# ---------------------------------------------------------------------------
loop_random_key = jax.random.PRNGKey(114514)
loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)
carry = (states, ppo_training_state, loop_random_key)

# ---------------------------------------------------------------------------
# Training loop  (identical structure to main_ant.py)
# ---------------------------------------------------------------------------
@jax.jit
def training_loop(
    carry: Tuple[GeneralizedState, PPOTrainingState, RNGKey],
    _: None,
) -> Tuple[Tuple, Tuple]:

    states, ppo_training_state, loop_random_key = carry

    (final_states, sampled_states, ppo_training_state, loop_random_key), aux_data = ppo.train(
        states,
        ppo_training_state,
        loop_random_key,
    )
    vs = jnp.sqrt(jnp.sum(sampled_states.env_state.obs[:, 13:15]**2, axis=-1))

    loop_random_key, subkey = jax.random.split(loop_random_key)
    ps = jax.random.bernoulli(subkey, p=0.5, shape=(vec_env,))

    new_states = jax.tree.map(
        lambda a, b: jax.vmap(jax.lax.select)(ps, a, b),
        sampled_states,
        final_states,
    )

    new_carry = (new_states, ppo_training_state, loop_random_key)

    return new_carry, (
        aux_data.training_data.critic_error,
        aux_data.training_data.approx_kl,
        aux_data.training_data.clip_fraction,
        aux_data.rollout_data.average_return,
        jnp.mean(vs),
    )


log_period = 10

for i in range(int(num_iterations / log_period)):

    (
        states,
        ppo_training_state,
        loop_random_key,
    ), (
        iteration_critic_error,
        iteration_approx_kl,
        iteration_clip_fraction,
        iteration_mean_return,
        iteration_mean_v,
    ) = jax.lax.scan(
        training_loop,
        carry,
        length=log_period,
    )

    wandb.log({
        "critic_RMSE":           jnp.mean(iteration_critic_error),
        "approx_kl":             jnp.mean(iteration_approx_kl),
        "clip_fraction":         jnp.mean(iteration_clip_fraction),
        "iteration mean return": jnp.mean(iteration_mean_return),
        "iteration_mean_v":      jnp.mean(iteration_mean_v),
    })

    print(
        f"[{(i + 1) * log_period:4d}/{num_iterations}]  "
        f"v={float(jnp.mean(iteration_mean_v)):.3f}  "
        f"return={float(jnp.mean(iteration_mean_return)):.3f}  "
        f"kl={float(jnp.mean(iteration_approx_kl)):.4f}"
    )

    carry = (states, ppo_training_state, loop_random_key)

wandb.finish()
