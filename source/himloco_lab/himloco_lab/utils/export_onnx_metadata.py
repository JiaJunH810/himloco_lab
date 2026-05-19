# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export ONNX metadata utilities for HimLoco policies."""

import os
import torch

from isaaclab.envs import ManagerBasedRLEnv


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    """将列表转为 CSV 字符串，数字保留指定位小数。

    Args:
        arr: 输入列表。
        decimals: 小数位数。
        delimiter: 分隔符。

    Returns:
        CSV 格式字符串。
    """
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    path: str,
    policy_filename: str = "policy.onnx",
) -> None:
    """将环境配置参数写入已导出的 ONNX 文件作为元数据。

    写入的元数据包括关节名称、刚度/阻尼、默认关节角度、
    指令/观测名称、动作缩放系数等，便于部署时直接读取而不需要重新配置。

    Args:
        env: Isaac Lab 环境实例。
        path: ONNX 文件所在目录。
        encoder_filename: Encoder ONNX 文件名。
        policy_filename: Policy ONNX 文件名。
    """
    robot = env.scene["robot"]

    # 动作缩放系数
    # scale 可能是单个 float（所有关节相同），也可能是 tensor
    action_term = env.action_manager.get_term("JointPositionAction")
    raw_scale = action_term._scale
    if isinstance(raw_scale, torch.Tensor):
        action_scale = raw_scale[0].cpu().tolist()
    else:
        # 单个 float：广播到所有关节
        action_scale = [raw_scale] * env.action_manager.total_action_dim

    # 构建元数据字典
    metadata = {
        "joint_names": robot.data.joint_names,
        "joint_stiffness": robot.data.default_joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot.data.default_joint_damping[0].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": action_scale,
        "body_names": robot.data.body_names,
    }

    # 写入 policy.onnx
    _write_metadata_to_onnx(os.path.join(path, policy_filename), metadata)

    print(f"[INFO] ONNX metadata attached to: {os.path.join(path, policy_filename)}")


def _write_metadata_to_onnx(onnx_path: str, metadata: dict) -> None:
    """将字典写入 ONNX 文件的 metadata_props 中。

    Args:
        onnx_path: ONNX 文件路径。
        metadata: 元数据字典。
    """
    if not os.path.exists(onnx_path):
        print(f"[WARN] ONNX file not found, skipping metadata: {onnx_path}")
        return

    import onnx  # 懒加载，避免 Isaac Sim 环境中未安装 onnx 时模块无法导入

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
