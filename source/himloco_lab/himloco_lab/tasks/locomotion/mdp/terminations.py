import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def base_too_low(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate when the robot base is too close to the local ground.

    Uses a height scanner to compute local ground elevation, then checks whether
    the robot's root link has dropped below min_height from the ground.

    On flat terrain without a sensor, compares root_pos_z directly to min_height.

    Args:
        env: The environment.
        min_height: Minimum allowed base height above local ground in meters.
        asset_cfg: The asset configuration for the robot.
        sensor_cfg: The height scanner sensor configuration. If None, uses
            root_pos_z directly as the height above world z=0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]

        valid_mask = (
            ~torch.isnan(ray_hits)
            & ~torch.isinf(ray_hits)
            & (torch.abs(ray_hits) < 1e6)
        )
        ray_hits_masked = torch.where(
            valid_mask, ray_hits,
            torch.tensor(float("nan"), device=ray_hits.device),
        )
        ground_z = torch.nanmean(ray_hits_masked, dim=1)

        # fallback: envs where all rays are invalid, use root_pos_z as ground
        all_invalid = torch.isnan(ground_z)
        ground_z[all_invalid] = 0.0

        base_height = asset.data.root_pos_w[:, 2] - ground_z
    else:
        base_height = asset.data.root_pos_w[:, 2]

    return base_height < min_height
