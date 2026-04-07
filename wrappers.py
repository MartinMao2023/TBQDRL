from typing import Callable, Dict, Optional, Tuple

# from brax.base import System
from brax.envs.base import Env, State, Wrapper
# from flax import struct
import jax
from jax import numpy as jp



class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    rng1, rng2 = jax.random.split(rng)
    state = self.env.reset(rng1)
    backup_state = self.env.reset(rng2)
    state.info['first_pipeline_state'] = backup_state.pipeline_state
    state.info['first_obs'] = backup_state.obs
    return state
  

  def refresh_backup_state(self, state: State, rng: jax.Array) -> State:
    backup_state = self.env.reset(rng)
    state.info['first_pipeline_state'] = backup_state.pipeline_state
    state.info['first_obs'] = backup_state.obs
    return state
  

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    pipeline_state = jax.tree.map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(pipeline_state=pipeline_state, obs=obs)
  


# def teleport_to_origin(self, state: envs.State) -> envs.State:
#         """
#         将 Agent 的全局 X, Y 坐标重置为 0，同时保留速度、姿态和其他物理状态。
#         必须显式调用 mjx.kinematics 来刷新物理引擎的派生数据。
#         """
#         # 1. 获取底层的 mjx.Data
#         data = state.pipeline_state
        
#         # 2. 修改 qpos：将前两位 (Global X, Y) 强制设为 0
#         # 注意：这里假设 Ant 的根关节是 Free Joint，qpos[:2] 对应 X, Y
#         new_qpos = data.qpos.at[:2].set(jnp.zeros(2))
        
#         # 3. 替换 qpos 数据
#         new_data = data.replace(qpos=new_qpos)
        
#         # 4. 【关键】调用运动学正解 (Forward Kinematics)
#         # 这一步强制刷新 data.xpos (笛卡尔坐标) 和 data.subtree_com (质心)
#         # 如果不加这行，物理引擎认为身体还在远处，只是关节数值变了，会导致渲染和碰撞检测错乱
#         new_data = mjx.kinematics(self.env.sys, new_data)
        
#         # 5. 返回更新后的 State
#         return state.replace(pipeline_state=new_data)