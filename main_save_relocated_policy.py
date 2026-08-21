import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import flax.linen as nn
import jax
import jax.numpy as jnp
import wandb
from brax import envs
from flax import serialization

from typing import Tuple
from custom_types import Params, RNGKey
from networks import GC_PPO_Policy, GC_combined_multi_Policy, GC_combined_Selector, GCMLP, GC_multi_Policy, GC_Selector
from task_wrappers.ant_mo_wrapper import AntMOWrapper
from task_wrappers.humanoid_mo_wrapper import HumanoidMOWrapper
from tools import (
    build_on_policy_rollout,
    with_policy_action_mean,
)
# from algorithms.relocation import relocate
from algorithms.cem_relocation import relocate
from data_struct.states import GeneralizedState
from data_struct import (
    MORelocationTransition,
    GMMDistillationTransition,
    PPOTransition,
)

from algorithms.combined_distill import CombinedDistill, CombinedDistillConfigs
from algorithms.gaussian_cloning import GaussianCloningConfigs, GaussianCloning
from algorithms.test_gmm_ppo import PPO, PPOConfigs
# from algorithms.warmup_gmm_ppo import PPO, PPOConfigs
# from algorithms.test_ppo import PPO, PPOConfigs


# PPO configs
vec_env = 4096
mini_batch_size = 8192
num_iterations = 1000
policy_epochs = 4
critic_epochs = 4
policy_learning_rate_per_std = 5e-4  # unified (used for the GMM student distil lr)
selector_learning_rate = 1e-4
critic_learning_rate = 5e-4
ppo_rollout_length = 32
# collection configs


env_name = "ant"
# env_name = "humanoid"


rollout_length = 64
relocation_config = {
    # TODO: relocation hyperparameters (candidate preferences, advantage
    # threshold, segment length, etc.)
}
bc_mini_batch_size = 2048
num_stage1_epochs = 32
num_stage2_epochs = 4
data_size = 2097152



if env_name == "ant":
    num_collect_iterations = 16
    env = envs.create(env_name="ant", episode_length=4096, backend="mjx", auto_reset=True)
    env = AntMOWrapper(env)
else:
    num_collect_iterations = 16 # <----------------    change to 8
    env = envs.create(env_name="humanoid", episode_length=4096, backend="mjx", auto_reset=True, action_repeat=2)
    env = HumanoidMOWrapper(env)

critic_hidden_layers: Tuple[int, ...] = (128, 128)
actor_hidden_layers: Tuple[int, ...] = (256, 256)
policy_network = GC_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
    learnable_std=True,
)

critic_network = GCMLP(
    layer_sizes=critic_hidden_layers + (1,),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    activation=nn.silu,
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
)

seed = 4455
loop_random_key = jax.random.PRNGKey(seed)
loop_random_key, subkey = jax.random.split(loop_random_key)


if env_name == "ant":
    folder_path = "output/MORL/test3"
else:
    folder_path = "output/MORL/huamnoid_mo_corrected" # <--- humanoid

with open(folder_path + "/policy.msgpack", "rb") as f:
    encoded_bytes = f.read()

fake_obs = jnp.zeros(shape=(env.observation_size,))
fake_zs = jnp.zeros(shape=(env.z_size,))

policy_template = policy_network.init(subkey, obs=fake_obs, z=fake_zs)
policy_params = serialization.from_bytes(policy_template, encoded_bytes)

with open(folder_path + "/critic.msgpack", "rb") as f:
    encoded_bytes = f.read()

critic_template = critic_network.init(subkey, obs=fake_obs, z=fake_zs)
critic_params = serialization.from_bytes(critic_template, encoded_bytes)

moving_mean = jnp.load(folder_path + "/mean.npy")
moving_std = jnp.sqrt(jnp.load(folder_path + "/var.npy"))

loop_random_key, subkey = jax.random.split(loop_random_key)
k1 = 0
k2 = 1
component_num = k1 + k2
if component_num > 1:
    component_means = jnp.concatenate([
        jnp.zeros(env.action_size),
        jax.random.normal(subkey, shape=((component_num - 1) * env.action_size)) * 0.01,
    ])
else:
    component_means = jnp.zeros(env.action_size)

student_hidden_layers = (256, 256)
student_action_network = GC_multi_Policy(
    hidden_layer_sizes=(256, 256),
    action_dim=env.action_size,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
    learnable_std=True,
    component_num=component_num,
    component_means=component_means,   
)

student_policy_network = GC_PPO_Policy(
    hidden_layer_sizes=actor_hidden_layers,
    action_dim=env.action_size,
    initial_std=0.1 * jnp.ones(env.action_size),
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
    final_activation=jnp.tanh,
    learnable_std=True,
)


student_selection_network = GC_Selector(
    hidden_layer_sizes=(128, 128),
    component_num=component_num,
    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2)),
    kernel_init_final=jax.nn.initializers.orthogonal(0.01),
    activation=nn.silu,
)


teacher_std_logits = policy_params["params"]["std_logits"]
teacher_std = jax.nn.sigmoid(teacher_std_logits)


expert_demo = False
if expert_demo:
    if env_name == "ant":
        with open("output/MORL/test4/policy.msgpack", "rb") as f:
            encoded_bytes = f.read()
    else:
        with open("output/MORL/test6/policy.msgpack", "rb") as f:
            encoded_bytes = f.read()
    expert_params = serialization.from_bytes(policy_template, encoded_bytes)
else:
    expert_params = policy_params


on_policy_rollout_fn = build_on_policy_rollout(
    env=env,
    policy_network=policy_network,
    policy_params=expert_params, 
    rollout_length=rollout_length,
)


@jax.jit
def collect_on_policy_data(starting_states, key):
    """Outer scan: run a rollout then resample the preference per env, repeat."""

    def outer_step(carry, _):
        states, key = carry
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=vec_env)
        final_states, transitions = on_policy_rollout_fn(states, subkeys)
        final_obs, _ = jax.vmap(env.get_obs)(final_states)
        final_last_actions = final_states.z_state.last_action
        final_info = (final_obs, final_last_actions)
        final_states = jax.vmap(env.resample_task_state)(final_states)
        return (final_states, key), (transitions, final_info)

    (final_states, key), (all_transitions, all_final_info) = jax.lax.scan(
        outer_step,
        (starting_states, key),
        length=num_collect_iterations,
    )
    return final_states, key, all_transitions, all_final_info


loop_random_key, subkey = jax.random.split(loop_random_key)
subkeys = jax.random.split(subkey, num=vec_env)
states = jax.vmap(env.reset)(subkeys)

if num_collect_iterations > 0:
    states, loop_random_key, on_policy_transitions, on_policy_final_info = (
        collect_on_policy_data(states, loop_random_key)
    )

need_distill = True

if num_collect_iterations > 0:
    combined_transitions = on_policy_transitions
    combined_final_info = on_policy_final_info


    loop_random_key, subkey = jax.random.split(loop_random_key)
    flattened = combined_transitions.flatten()
    index = jax.random.permutation(subkey, flattened.shape[0])[:data_size]
    distillation_transitions = combined_transitions.from_flatten(flattened[index], combined_transitions)

    distillation_transitions = with_policy_action_mean(
        distillation_transitions, policy_network, policy_params,
    )
else:
    need_distill = False


need_relocate = True

if need_distill:
    print(combined_transitions.obs.shape)

    if need_relocate:
        loop_random_key, subkey = jax.random.split(loop_random_key)
        relocated_demonstrations = relocate(
            combined_transitions,
            combined_final_info,
            critic_network=critic_network,
            critic_params=critic_params,
            moving_mean=moving_mean,
            moving_std=moving_std,
            config=relocation_config,
            key=subkey,
            max_data_size=data_size,
        )
    else:
        dummy = jnp.ones_like(combined_transitions.td_lambda_returns) 
        relocated_demonstrations = PPOTransition(
            obs=combined_transitions.obs,
            actions=combined_transitions.actions,
            zs=combined_transitions.zs,
            log_likelihood=combined_transitions.log_likelihood,
            rewards=dummy,
            td_lambda_returns=dummy,
            gaes=dummy,
            dones=dummy,
            truncations=dummy,
            weights=dummy,
        )


configs = GaussianCloningConfigs(
    learning_rate=3e-4,
    epochs=32,
    mini_batch_size=2048,
)


bc = GaussianCloning(
    env=env,
    policy_network=student_policy_network,
    cloning_configs=configs,
    initial_std_logits=teacher_std_logits,  # optional
)

if need_distill:
    loop_random_key, subkey = jax.random.split(loop_random_key)
    demo_flat = relocated_demonstrations.shuffle(subkey)            # (N_demo, ...)

    distill_flat = distillation_transitions

    # Both buffers must end up the same size so the per-epoch scan lines up.
    _n = min(distill_flat.obs.shape[0], demo_flat.obs.shape[0])
    _n = (_n // bc_mini_batch_size) * bc_mini_batch_size

    def _batch(x):
        return x[:_n].reshape(-1, bc_mini_batch_size, *x.shape[1:])

    demo_batched = jax.tree.map(_batch, demo_flat)
    distill_batched = jax.tree.map(_batch, distill_flat)

    _distil_batch = distill_batched.obs.shape[:-1]
    gmm_distillation_transitions = GMMDistillationTransition(
        obs=distill_batched.obs,
        zs=distill_batched.zs,
        action_means=distill_batched.actions[..., None, :],  # (..., 1, d)
        component_logits=jnp.zeros((*_distil_batch, 1)),     # (..., 1) -> softmax 1
    )

    loop_random_key, subkey = jax.random.split(loop_random_key)
    bc_training_state = bc.init(subkey)

    print(
        f"Combined distill: {_n} samples | "
        f"{_n // bc_mini_batch_size} mini-batches x {bc_mini_batch_size} | "
        f"stage1 {num_stage1_epochs} epochs + stage2 {num_stage2_epochs} epochs"
    )

    bc_training_state, _metrics = bc.distill(
        bc_training_state,
        demo_batched,
        subkey,
    )

    print(_metrics.bc_loss)

    # Relocated-only distilled student policy (std fixed to teacher by distill).
    relocated_only_policy_params = bc_training_state.policy_params



model_bytes = serialization.to_bytes(relocated_only_policy_params)
if expert_demo:
    if env_name == "ant":
        if need_relocate:
            folder_path = f"./output/MORL/ant_relocated_expert"
        else:
            folder_path = f"./output/MORL/ant_distill_expert"
    else:
        if need_relocate:
            folder_path = f"./output/MORL/humanoid_relocated_expert"
        else:
            folder_path = f"./output/MORL/humanoid_distill_expert"
else:
    if env_name == "ant":
        if need_relocate:
            folder_path = f"./output/MORL/ant_relocated"
        else:
            folder_path = f"./output/MORL/ant_no_relocated"
    else:
        if need_relocate:
            folder_path = f"./output/MORL/humanoid_relocated"
        else:
            folder_path = f"./output/MORL/humanoid_no_relocated"

# if not os.path.exists(folder_path):
#     os.makedirs(folder_path, exist_ok=True)
#     print(f"new folder <{folder_path}> created")

# with open(folder_path + f"/policy.msgpack", "wb") as f:
#     f.write(model_bytes)

