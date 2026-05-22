"""Configuration for Cyborg biped robot.

Reference: robot_lab project cyborg configuration.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from himloco_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from himloco_lab.assets.unitree import UnitreeArticulationCfg


CYBORG_BIPED_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/cyborg/biped_temp_1_0/urdf/biped_temp_1_0_fixed.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    joint_sdk_names=[
        "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch",
        "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll",
        "J_hip_r_roll", "J_hip_r_yaw", "J_hip_r_pitch",
        "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll",
    ],
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_hip_.*_roll",
                "J_hip_.*_yaw",
                "J_hip_.*_pitch",
                "J_knee_.*_pitch",
            ],
            effort_limit_sim={
                "J_hip_.*_roll": 330.0,
                "J_hip_.*_yaw": 330.0,
                "J_hip_.*_pitch": 330.0,
                "J_knee_.*_pitch": 330.0,
            },
            velocity_limit_sim={
                "J_hip_.*_roll": 12.04,
                "J_hip_.*_yaw": 12.04,
                "J_hip_.*_pitch": 12.04,
                "J_knee_.*_pitch": 12.04,
            },
            stiffness={
                "J_hip_.*_roll": 250.0,
                "J_hip_.*_yaw": 120.0,
                "J_hip_.*_pitch": 300.0,
                "J_knee_.*_pitch": 300.0,
            },
            damping={
                "J_hip_.*_roll": 10.0,
                "J_hip_.*_yaw": 10.0,
                "J_hip_.*_pitch": 10.0,
                "J_knee_.*_pitch": 10.0,
            },
            armature={
                ".*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=120.0,
            velocity_limit_sim=11.21,
            joint_names_expr=["J_ankle_.*_pitch", "J_ankle_.*_roll"],
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
        ),
    },
)

CYBORG_BIPED_ACTION_SCALE = {}
for a in CYBORG_BIPED_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            CYBORG_BIPED_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
