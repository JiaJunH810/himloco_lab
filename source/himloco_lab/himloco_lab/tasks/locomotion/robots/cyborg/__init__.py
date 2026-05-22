import gymnasium as gym

gym.register(
    id="Cyborg-Biped-Velocity",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:CyborgPPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himloco_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Cyborg-Biped-Velocity-Play",
    entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himloco_lab.tasks.locomotion.agents.himloco_rsl_rl_cfg:CyborgPPORunnerCfg",
    },
)
