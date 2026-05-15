import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp
import himloco_lab.terrains as him_terrains

# 鹅卵石路面地形配置
# 用于训练四足机器人在多样化地形上的运动能力，包含斜坡、台阶、障碍物等多种地形类型
# 按比例混合，每个地形块为8m x 8m，外围有25m的平坦边界
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    # === 地形网格基础参数 ===
    size=(8.0, 8.0),          # 每个地形块的尺寸 [m]（宽 x 长）
    border_width=25.0,        # 地形外围平坦边界的宽度 [m]，边界内坡度为0
    num_rows=10,              # 难度等级行数，从上到下难度递增（10个级别）
    num_cols=20,              # 地形类型列数，每行随机生成20种不同类型的地形块
    horizontal_scale=0.1,     # 高度场网格分辨率 [m]，越小越精细
    vertical_scale=0.005,     # 高度缩放系数 [m]，乘到高度场上控制整体起伏幅度
    slope_threshold=0.75,     # 坡度阈值，超过此值的三角面会应用不同的物理材质
    difficulty_range=(0.0, 1.0),  # 难度映射范围，0.0=最简单, 1.0=最难，随行号线性递增
    use_cache=True,           # 启用地形缓存，避免重复生成
    # === 子地形类型及占比 ===
    sub_terrains={
        # 上坡金字塔斜坡：中间有平台的凸起斜坡，向四周倾斜（占比 5%）
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,            # 占总地形的比例
            slope_range=(0.0, 0.4),     # 坡度范围 [rad]，随难度递增
            platform_width=3.0,         # 顶部平台的宽度 [m]
            border_width=0.0,           # 斜坡区域的地形块内边距
        ),
        # 下坡倒置金字塔斜坡：中间有平台的凹陷斜坡（占比 5%）
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
        ),
        # 带噪␟的粗糙斜坡：在金字塔斜坡上叠加了随机噪␟，模拟不规则路面（占比 20%）
        "hf_slope_with_noise": him_terrains.HfPyramidSlopeWithNoiseCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),                     # 基础斜坡坡度范围
            platform_width=3.0,
            border_width=0.0,
            noise_amplitude_range=(0.01, 0.08),          # 噪␟幅度范围 [m]，随难度增大
            noise_step=0.005,                            # 噪␟采样步长 [m]
            downsampled_scale=0.2,                       # 降采样后的水平分辨率 [m]
        ),
        # 金字塔台阶：一面有阶梯、另一面是斜坡的凸起地形（占比 30%）
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),  # 台阶高度范围 [m]，实际高度 = 0.05 + 0.18 * difficulty
            step_width=0.30,                 # 每级台阶的宽度 [m]
            platform_width=3.0,              # 顶端平台宽度 [m]
            border_width=0.0,
        ),
        # 倒置金字塔台阶：凹陷到地面以下的阶梯状地形（占比 30%）
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.0,
        ),
        # 离散障碍物：地面随机散布柱状/块状障碍物（占比 10%）
        "discrete_obstacles": him_terrains.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            max_height_range=(0.05, 0.15),   # 障碍物高度范围 [m]，实际高度 = 0.05 + 0.1 * difficulty
            obstacle_size_range=(1.0, 2.0),  # 障碍物底边尺寸范围 [m]
            num_obstacles=20,                # 每块地形上障碍物的数量
            platform_width=3.0,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0, 0),
        },
    )

    # interval
    external_force = EventTerm(
        func=mdp.apply_periodic_external_force_torque,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "period_step": 8,
            "force_range": (-30.0, 30.0),
            "torque_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={
            "velocity_range": {"x": (-1, 1), "y": (-1, 1)},
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        heading_command=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1, 1), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
        ),
        heading_control_stiffness=0.5,
        curriculums_limit_ranges=(-2, 2),
        low_vel_env_lin_x_ranges=(-1, 1),
        rel_high_vel_envs=0.2,
        min_command_norm=0.2,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )

# observation compute step in lab: noise clip scale
# observation compute step in gym: clip scale noise
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # Critic 观测组 —— 继承 PolicyCfg 的全部 6 项 + 额外 3 项特权信息
    # Critic 在训练时拥有比 Actor 更多的信息（特权观测，即 privileged observations），
    # 从而学到更准确的价值估计，而 Actor 部署时无需这些特权信息，形成非对称训练。
    # Critic 观测 = Policy 观测（6 项） + 以下 3 项特权观测，构成 num_critic_obs 维输入
    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        # 特权项 1: 机身线速度 base_lin_vel —— 3 维 (vx, vy, vz)
        # Actor 没有这个信息（只能从历史观测中间接推断），Critic 用于精确评估速度跟踪表现
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=2.0,  # 缩放到 [-2, 2] 范围
            clip=(-100, 100),
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 加噪增强鲁棒性
        )
        # 特权项 2: 机身受到的外力 base_external_force —— 3 维 (fx, fy, fz)
        # 来自 mdp.base_external_force，读取 base 躯干上通过 permanent_wrench_composer 组合的累计外力
        # 让 Critic 知道机器人当前受到的外部扰动，更准确评估状态价值
        base_external_force = ObsTerm(
            func=mdp.base_external_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            clip=(-100, 100),
        )
        # 特权项 3: 地形高度扫描 height_scanner —— 160 维 (16×10 网格)
        # 来自 mdp.height_scan_clip，对 height_scanner 传感器的射线命中点做 clip(-1, 1)，减去 offset 0.5
        # 让 Critic 获得机器人周围 1.6m×1.0m 区域的地形高度信息
        height_scanner = ObsTerm(
            func=mdp.height_scan_clip,
            scale=5.0,  # 放大数值差异
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.0, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
        },
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    
    base_height_l2 = RewTerm(
        func=mdp.base_height, 
        weight=-1.0, 
        params={
            "target_height": 0.3,
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )

    # 摆动足高度惩罚：将足部位置旋转变换到机身坐标系，惩罚足部 Z 偏离 target_height，
    # 惩罚量 = 高度偏差² × 足部水平速率，仅摆动相生效（支撑相速率≈0，惩罚≈0）。
    # target_height=-0.2 表示期望足部在机身下方 0.2m 处抬腿，避免拖地或抬得过高。
    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=-0.01,       # 负权重，惩罚摆动足高度不当
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),  # 四足的所有 foot 刚体
            "target_height": -0.2,   # 机身坐标系下的目标足部 Z 坐标（机身下方 0.2m）
            "command_name": "base_velocity",
        }
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    smoothness = RewTerm(func=mdp.smoothness, weight=-0.01)
    # joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    
    # head_undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1,
    #     params={
    #         "threshold": 0.3,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
    #     },
    # )
    
    # other_undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.01,
    #     params={
    #         "threshold": 0.3,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
    #     },
    # )

    # is_terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)
    # joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-5.0)
    # applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-5.0)
    
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    
    # feet_stumble = RewTerm(
    #     func=mdp.feet_stumble,
    #     weight=-0.01,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # joint_pos = RewTerm(
    #     func=mdp.joint_position_penalty,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.3,
    #     },
    # )
    
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )


####################################################################################

    # # -- feet
    # air_time_variance = RewTerm(
    #     func=mdp.air_time_variance_penalty,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )
    # feet_gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=1.0,
    #     params={
    #         "period": 0.5,  
    #         "offset": [0.0, 0.5, 0.5, 0.0],  # （LF, RF, LH, RH）
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "threshold": 0.5,
    #         "command_name": "base_velocity",
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # simulation settings 
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # PhysX settings
        self.sim.physx.solver_type = 1  # TGS solver
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5
        
        self.sim.physx.gpu_max_rigid_patch_count = 2**23 
        self.sim.physx.gpu_max_rigid_contact_count = 2**23 
        # self.sim.physx.gpu_found_lost_pairs_capacity = 2**23

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.base_height_scanner.update_period = self.sim.dt * self.decimation

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1, 1), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0), 
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges=(1,1)
        
        # Disable randomization events for play mode
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_robot = None
