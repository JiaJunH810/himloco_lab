from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """线速度指令课程：根据速度跟踪奖励自适应扩展速度指令范围。

    核心思路：
    1. 将并行环境按 ID 拆分成"高速组"和"低速组"
    2. 每个 episode 结束时评估两组各自的平均速度跟踪奖励
    3. 只有当两组都达标时，才将指令速度范围向两端扩展 ±0.2 m/s
    4. 受 curriculums_limit_ranges 约束（通常为 [-2, 2] m/s）

    为什么需要分两组？
    - 防止策略只学会低速行走就满足课程条件（高速组拖后腿则不能升级）
    - 防止低速环境被高速指令压垮后拉低整体指标（低速组拖后腿也不能升级）
    - 两组都达标说明策略在不同速度段都有稳定表现，课程升级才安全
    """

    # ---- 1. 获取速度指令管理器及当前参数 ----
    # command_term: UniformLevelVelocityCommand 实例，管理线速度/角速度/朝向指令的采样与下发
    command_term = env.command_manager.get_term("base_velocity")
    # ranges: 当前生效的指令范围，如 lin_vel_x=(-1, 1)，课程学习过程中会被此函数逐步扩展
    ranges = command_term.cfg.ranges
    # limit_ranges: 课程扩展的上下界，如 (-2, 2)，防止速度范围无限扩张
    limit_ranges = command_term.cfg.curriculums_limit_ranges

    # ---- 2. 将环境按 ID 拆分为高速组和低速组 ----
    # 环境 ID 0..num_envs-1 按顺序排布，ID 越小的环境被分配越大的指令速度
    # rel_high_vel_envs=0.2 表示前 80% 环境 (ID < 0.8*num_envs) 为高速组
    #  例: num_envs=4096 时，ID 0~3276 为高速组，ID 3277~4095 为低速组
    low_vel_mask = env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs)
    high_vel_mask = env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs)
    # nonzero(as_tuple=True) 返回 mask 中值为 True 的索引位置
    low_vel_env_ids = env_ids[low_vel_mask.nonzero(as_tuple=True)]
    high_vel_env_ids = env_ids[high_vel_mask.nonzero(as_tuple=True)]

    # ---- 3. 读取奖励项权重，用于计算达标阈值 ----
    # 达标条件: episode 平均奖励 > weight * 0.8，即奖励达到满分的 80%
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # ---- 4. 每个 episode 结束时评估一次 ----
    # common_step_counter: 全局步数计数器，max_episode_length: 每个 episode 的最大步数
    # 取模为 0 时表示一个 episode 刚刚结束，所有环境的累积奖励已完整
    if env.common_step_counter % env.max_episode_length == 0:
        # 4a. 计算低速组的 episode 平均速度跟踪奖励
        # _episode_sums[reward_term_name]: shape (num_envs,) ，每个环境在当前 episode 中的累积 track_lin_vel_xy 奖励
        # 除以 episode_length_s 得到每秒钟平均奖励，消除 episode 长度差异
        reward_low = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids])
            / env.max_episode_length_s
            if len(low_vel_env_ids) > 0
            else 0.0
        )
        # 4b. 计算高速组的 episode 平均速度跟踪奖励（逻辑同上）
        reward_high = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids])
            / env.max_episode_length_s
            if len(high_vel_env_ids) > 0
            else 0.0
        )

        # 4c. 两组都达标 → 扩展指令速度范围
        # delta_command = [-0.2, 0.2] 表示下界减 0.2，上界加 0.2
        # 例如 [-1, 1] → [-1.2, 1.2] → [-1.4, 1.4] → ... 直到达到 limit_ranges
        # clamp 确保不会超出 limit_ranges 的限制
        if reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

    # ---- 5. 返回当前速度范围上界，供 CommandManager 作为 curriculum 缩放因子 ----
    # 返回值 shape: 标量 tensor，内容为 lin_vel_x 的最大值
    # CommandManager 内部用此值对采样的指令做缩放
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upgrade_tile_ratio: float = 0.5,
    grace_period_resets: int = 15,
) -> torch.Tensor:
    """地形等级课程：带升级保护期。

    相比 Isaac Lab 原版的改进：
    1. 升级后 15 个 episode 内禁止降级，打破升-跌-降的死亡循环
    2. 有保护期兜底后，升级门槛保持原地块尺寸/2（4m），需要策略真正走起来
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # 计算从出生点到当前位置的欧氏距离
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    # 升级条件：走了超过 upgrade_tile_ratio 倍的地块宽度（0.5 * 8m = 4m）
    tile_size = terrain.cfg.terrain_generator.size[0]
    move_up = distance > tile_size * upgrade_tile_ratio

    # 降级条件：走的距离 < 指令速度 * 最大 episode 时长 * 0.5
    # 即如果完全跟不上指令速度，就降级
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up  # 升级优先

    # --- 保护期机制 ---
    if not hasattr(env, "_curriculum_grace_counter"):
        env._curriculum_grace_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    # 刚升级的环境获得保护期
    upgraded_ids = env_ids[move_up]
    if len(upgraded_ids) > 0:
        env._curriculum_grace_counter[upgraded_ids] = grace_period_resets

    # 所有被评估的环境保护期减 1
    env._curriculum_grace_counter[env_ids] = torch.clamp(
        env._curriculum_grace_counter[env_ids] - 1, min=0
    )

    # 保护期内禁止降级
    in_grace = env._curriculum_grace_counter[env_ids] > 0
    move_down[in_grace] = False

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
