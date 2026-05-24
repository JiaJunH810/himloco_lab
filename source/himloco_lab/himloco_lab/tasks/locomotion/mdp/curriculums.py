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

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.curriculums_limit_ranges

    low_vel_mask = env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs)
    high_vel_mask = env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs)
    low_vel_env_ids = env_ids[low_vel_mask.nonzero(as_tuple=True)]
    high_vel_env_ids = env_ids[high_vel_mask.nonzero(as_tuple=True)]

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        reward_low = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids])
            / env.max_episode_length_s
            if len(low_vel_env_ids) > 0
            else 0.0
        )
        reward_high = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids])
            / env.max_episode_length_s
            if len(high_vel_env_ids) > 0
            else 0.0
        )

        if reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upgrade_tile_ratio: float = 0.5,
    grace_period_resets: int = 15,
) -> torch.Tensor:
    """地形等级课程：带升级保护期。

    升级条件：行走距离 > 地块尺寸的一半（0.5 * 8m = 4m）
    降级条件：行走距离 < 指令速度要求距离的 50%
    保护期：升级后 15 个 episode 内禁止降级
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    tile_size = terrain.cfg.terrain_generator.size[0]
    move_up = distance > tile_size * upgrade_tile_ratio

    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    if not hasattr(env, "_curriculum_grace_counter"):
        env._curriculum_grace_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    upgraded_ids = env_ids[move_up]
    if len(upgraded_ids) > 0:
        env._curriculum_grace_counter[upgraded_ids] = grace_period_resets

    env._curriculum_grace_counter[env_ids] = torch.clamp(
        env._curriculum_grace_counter[env_ids] - 1, min=0
    )

    in_grace = env._curriculum_grace_counter[env_ids] > 0
    move_down[in_grace] = False

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
