from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformLevelVelocityCommandCfg



class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.

    === 与父类 UniformVelocityCommand 的关键差异 ===

    1. 速度分层：根据 env ID 将环境分为两组
       - 前 rel_high_vel_envs (20%) 的 env：使用全速范围 lin_vel_x
       - 其余 env：使用低速范围 low_vel_env_lin_x_ranges
       这种分区是固定的（按 env ID），不是随机的

    2. 高速 env 的 y 速度：如果重新采样后 x 速度仍然落在低速范围内，
       则将 y 速度清零，确保高速 env 真的跑起来

    3. 最小指令模长过滤：速度指令模长 < min_command_norm 时清零，
       避免给出几乎不动的指令

    4. 保留 heading 跟踪（继承自父类），但 rel_heading_envs 控制
       多少比例的 env 使用 heading 模式
    """

    cfg: UniformLevelVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定的环境重新采样速度指令。

        采样流程：
        1. 所有 env 先从低速 x 范围 (low_vel_env_lin_x_ranges) 采样 x 速度
        2. 所有 env 采样 y 速度和角速度
        3. 如果开了 heading_command，采样目标朝向，按概率决定哪些 env 用 heading 跟踪
        4. 前 rel_high_vel_envs (20%) 的 env 重新从全速范围采样 x 速度
        5. 高速 env 中，如果新 x 速度仍在低速范围内，y 速度清零
        6. 速度模长 < min_command_norm 的指令清零
        """
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)

        # -- 第 1 步：线性速度 x —— 所有 env 先用低速范围
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        # -- 第 2 步：线性速度 y
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- 第 2 步：角速度 z（非 heading 模式时直接使用此值）
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # -- 第 3 步：heading 目标
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # 按 rel_heading_envs 概率决定哪些 env 使用 heading 跟踪，其余用直接角速度
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # -- 第 4 步：高速 env 组 —— env ID 在 [0, num_envs * rel_high_vel_envs) 范围内的
        #    重新从全速范围采样 lin_vel_x
        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
        
        # -- 第 5 步：高速 env 中，如果新 x 速度落在低速范围内，将 y 速度清零
        #    这样可以确保高速 env 真的是"高速直行"，而不是低速绕圈
        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                            (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range

        # -- 第 6 步：速度指令模长 < min_command_norm 的，将 x,y 清零
        #    避免给出几乎不动的指令，让狗原地发呆（直接给0指令更干净）
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > \
                                            self.cfg.min_command_norm).unsqueeze(1)
        

    def _update_command(self):
        """后处理速度指令。

        父类 UniformVelocityCommand._update_command 做两件事：
        1. heading env：用 P 控制器根据 heading 误差计算角速度 ω = stiffness × error
        2. standing env：速度指令全部清零

        本类不覆盖此方法，继承父类行为。
        """
        # 调用父类：处理 heading 跟踪 + standing 清零
        super()._update_command()
