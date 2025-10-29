"""Preset locomotion skills for the Unitree Go2 quadruped."""

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2BaseRoughEnvCfg, Go2RoughPPORunnerCfg
from .base import SkillSpec, register_skill


# Command templates -----------------------------------------------------------
_FORWARD_TROT_COMMANDS = {
    "lin_vel_x": (0.6, 1.5),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.4, 0.4),
    "resample": (6.0, 6.0),
}

_TURN_LEFT_COMMANDS = {
    "lin_vel_x": (-0.1, 0.4),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (0.5, 1.2),
    "resample": (5.0, 5.0),
}

_TURN_RIGHT_COMMANDS = {
    "lin_vel_x": (-0.1, 0.4),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-1.2, -0.5),
    "resample": (5.0, 5.0),
}

_CRAWL_COMMANDS = {
    "lin_vel_x": (0.1, 0.4),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (10.0, 10.0),
}

_DIAG_FORWARD_LEFT_COMMANDS = {
    "lin_vel_x": (0.5, 1.2),
    "lin_vel_y": (0.3, 0.8),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_DIAG_FORWARD_RIGHT_COMMANDS = {
    "lin_vel_x": (0.5, 1.2),
    "lin_vel_y": (-0.8, -0.3),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_DIAG_BACKWARD_LEFT_COMMANDS = {
    "lin_vel_x": (-1.2, -0.4),
    "lin_vel_y": (0.3, 0.8),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_DIAG_BACKWARD_RIGHT_COMMANDS = {
    "lin_vel_x": (-1.2, -0.4),
    "lin_vel_y": (-0.8, -0.3),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_STRAFE_LEFT_COMMANDS = {
    "lin_vel_x": (-0.1, 0.1),
    "lin_vel_y": (0.6, 1.1),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_STRAFE_RIGHT_COMMANDS = {
    "lin_vel_x": (-0.1, 0.1),
    "lin_vel_y": (-1.1, -0.6),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}

_BACKWARD_COMMANDS = {
    "lin_vel_x": (-1.2, -0.5),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (6.0, 6.0),
}


# Skill definitions -----------------------------------------------------------
@configclass
class Go2ForwardTrotEnvCfg(Go2BaseRoughEnvCfg):
    """Default trotting skill biased toward forward velocity tracking."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _FORWARD_TROT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _FORWARD_TROT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _FORWARD_TROT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _FORWARD_TROT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 0.3


@configclass
class Go2ForwardTrotRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_forward_trot"


FORWARD_TROT_SPEC = SkillSpec(
    skill_id="go2/velocity/trot_forward",
    robot="go2",
    gym_id="go2_skill_trot_forward",
    env_cfg_cls=Go2ForwardTrotEnvCfg,
    runner_cfg_cls=Go2ForwardTrotRunnerCfg,
    description="Unitree Go2 forward trot with moderate yaw authority.",
    tags=("velocity", "locomotion", "trot"),
    metadata={
        "command_ranges": {k: v for k, v in _FORWARD_TROT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _FORWARD_TROT_COMMANDS["resample"],
    },
)


@configclass
class Go2TurnLeftEnvCfg(Go2BaseRoughEnvCfg):
    """Left spinning skill for in-place yaw maneuvers."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _TURN_LEFT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _TURN_LEFT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _TURN_LEFT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _TURN_LEFT_COMMANDS["resample"]

        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_lin_vel_xy_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.01


@configclass
class Go2TurnLeftRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_turn_left"


TURN_LEFT_SPEC = SkillSpec(
    skill_id="go2/velocity/turn_left",
    robot="go2",
    gym_id="go2_skill_turn_left",
    env_cfg_cls=Go2TurnLeftEnvCfg,
    runner_cfg_cls=Go2TurnLeftRunnerCfg,
    description="Unitree Go2 left turn-in-place skill for agile yaw control.",
    tags=("velocity", "yaw", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _TURN_LEFT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _TURN_LEFT_COMMANDS["resample"],
    },
)


@configclass
class Go2TurnRightEnvCfg(Go2BaseRoughEnvCfg):
    """Right spinning skill mirroring the left-turn preset."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _TURN_RIGHT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _TURN_RIGHT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _TURN_RIGHT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _TURN_RIGHT_COMMANDS["resample"]

        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_lin_vel_xy_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.01


@configclass
class Go2TurnRightRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_turn_right"


TURN_RIGHT_SPEC = SkillSpec(
    skill_id="go2/velocity/turn_right",
    robot="go2",
    gym_id="go2_skill_turn_right",
    env_cfg_cls=Go2TurnRightEnvCfg,
    runner_cfg_cls=Go2TurnRightRunnerCfg,
    description="Unitree Go2 right turn-in-place counterpart to the left spin.",
    tags=("velocity", "yaw", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _TURN_RIGHT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _TURN_RIGHT_COMMANDS["resample"],
    },
)


@configclass
class Go2CrawlEnvCfg(Go2BaseRoughEnvCfg):
    """Low-profile crawl suitable for interacting with obstacles."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _CRAWL_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _CRAWL_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _CRAWL_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _CRAWL_COMMANDS["resample"]

        # Encourage crouched posture and precise foot placement.
        self.rewards.base_height.weight = -8.0
        self.rewards.base_height.params["target_height"] = 0.20
        self.rewards.hip_deviation.weight = -0.1
        self.rewards.joint_deviation.weight = -0.02
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.action_rate_l2.weight = -0.015


@configclass
class Go2CrawlRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_crawl_forward"
    num_steps_per_env = 40  # longer horizon helps crawling balance


CRAWL_SPEC = SkillSpec(
    skill_id="go2/manipulation/crawl_low",
    robot="go2",
    gym_id="go2_skill_crawl_low",
    env_cfg_cls=Go2CrawlEnvCfg,
    runner_cfg_cls=Go2CrawlRunnerCfg,
    description="Low-base crawl for close-proximity interactions with objects.",
    tags=("crawl", "posture", "interaction"),
    metadata={
        "command_ranges": {k: v for k, v in _CRAWL_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _CRAWL_COMMANDS["resample"],
        "target_height": 0.20,
    },
)

@configclass
class Go2ForwardLeftEnvCfg(Go2BaseRoughEnvCfg):
    """Diagonal gait combining forward motion with left strafe."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _DIAG_FORWARD_LEFT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _DIAG_FORWARD_LEFT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _DIAG_FORWARD_LEFT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _DIAG_FORWARD_LEFT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.015


@configclass
class Go2ForwardLeftRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_forward_left"


FORWARD_LEFT_SPEC = SkillSpec(
    skill_id="go2/velocity/forward_left",
    robot="go2",
    gym_id="go2_skill_forward_left",
    env_cfg_cls=Go2ForwardLeftEnvCfg,
    runner_cfg_cls=Go2ForwardLeftRunnerCfg,
    description="Forward-left diagonal trot for combining translation and sidestep.",
    tags=("velocity", "diagonal", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _DIAG_FORWARD_LEFT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _DIAG_FORWARD_LEFT_COMMANDS["resample"],
    },
)


@configclass
class Go2ForwardRightEnvCfg(Go2BaseRoughEnvCfg):
    """Diagonal gait combining forward motion with right strafe."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _DIAG_FORWARD_RIGHT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _DIAG_FORWARD_RIGHT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _DIAG_FORWARD_RIGHT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _DIAG_FORWARD_RIGHT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.2
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.015


@configclass
class Go2ForwardRightRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_forward_right"


FORWARD_RIGHT_SPEC = SkillSpec(
    skill_id="go2/velocity/forward_right",
    robot="go2",
    gym_id="go2_skill_forward_right",
    env_cfg_cls=Go2ForwardRightEnvCfg,
    runner_cfg_cls=Go2ForwardRightRunnerCfg,
    description="Forward-right diagonal trot for circling obstacles.",
    tags=("velocity", "diagonal", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _DIAG_FORWARD_RIGHT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _DIAG_FORWARD_RIGHT_COMMANDS["resample"],
    },
)


@configclass
class Go2BackwardEnvCfg(Go2BaseRoughEnvCfg):
    """Primary backward walking skill."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _BACKWARD_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _BACKWARD_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _BACKWARD_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _BACKWARD_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.02


@configclass
class Go2BackwardRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_backward_walk"


BACKWARD_SPEC = SkillSpec(
    skill_id="go2/velocity/walk_backward",
    robot="go2",
    gym_id="go2_skill_walk_backward",
    env_cfg_cls=Go2BackwardEnvCfg,
    runner_cfg_cls=Go2BackwardRunnerCfg,
    description="Reverse locomotion skill for retreating in cluttered spaces.",
    tags=("velocity", "backward", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _BACKWARD_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _BACKWARD_COMMANDS["resample"],
    },
)


@configclass
class Go2BackwardLeftEnvCfg(Go2BaseRoughEnvCfg):
    """Diagonal retreat toward the left."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _DIAG_BACKWARD_LEFT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _DIAG_BACKWARD_LEFT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _DIAG_BACKWARD_LEFT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _DIAG_BACKWARD_LEFT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.1
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.018


@configclass
class Go2BackwardLeftRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_backward_left"


BACKWARD_LEFT_SPEC = SkillSpec(
    skill_id="go2/velocity/backward_left",
    robot="go2",
    gym_id="go2_skill_backward_left",
    env_cfg_cls=Go2BackwardLeftEnvCfg,
    runner_cfg_cls=Go2BackwardLeftRunnerCfg,
    description="Diagonal backward-left gait for evasive maneuvers.",
    tags=("velocity", "diagonal", "backward"),
    metadata={
        "command_ranges": {k: v for k, v in _DIAG_BACKWARD_LEFT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _DIAG_BACKWARD_LEFT_COMMANDS["resample"],
    },
)


@configclass
class Go2BackwardRightEnvCfg(Go2BaseRoughEnvCfg):
    """Diagonal retreat toward the right."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _DIAG_BACKWARD_RIGHT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _DIAG_BACKWARD_RIGHT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _DIAG_BACKWARD_RIGHT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _DIAG_BACKWARD_RIGHT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.1
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.action_rate_l2.weight = -0.018


@configclass
class Go2BackwardRightRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_backward_right"


BACKWARD_RIGHT_SPEC = SkillSpec(
    skill_id="go2/velocity/backward_right",
    robot="go2",
    gym_id="go2_skill_backward_right",
    env_cfg_cls=Go2BackwardRightEnvCfg,
    runner_cfg_cls=Go2BackwardRightRunnerCfg,
    description="Diagonal backward-right gait for evasive maneuvers.",
    tags=("velocity", "diagonal", "backward"),
    metadata={
        "command_ranges": {k: v for k, v in _DIAG_BACKWARD_RIGHT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _DIAG_BACKWARD_RIGHT_COMMANDS["resample"],
    },
)


@configclass
class Go2StrafeLeftEnvCfg(Go2BaseRoughEnvCfg):
    """Pure lateral left shuffle."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _STRAFE_LEFT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _STRAFE_LEFT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _STRAFE_LEFT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _STRAFE_LEFT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.3
        self.rewards.track_ang_vel_z_exp.weight = 0.6
        self.rewards.action_rate_l2.weight = -0.018


@configclass
class Go2StrafeLeftRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_strafe_left"


STRAFE_LEFT_SPEC = SkillSpec(
    skill_id="go2/velocity/strafe_left",
    robot="go2",
    gym_id="go2_skill_strafe_left",
    env_cfg_cls=Go2StrafeLeftEnvCfg,
    runner_cfg_cls=Go2StrafeLeftRunnerCfg,
    description="Side-step left skill for aligning with targets.",
    tags=("velocity", "strafe", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _STRAFE_LEFT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _STRAFE_LEFT_COMMANDS["resample"],
    },
)


@configclass
class Go2StrafeRightEnvCfg(Go2BaseRoughEnvCfg):
    """Pure lateral right shuffle."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _STRAFE_RIGHT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _STRAFE_RIGHT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _STRAFE_RIGHT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _STRAFE_RIGHT_COMMANDS["resample"]

        self.rewards.track_lin_vel_xy_exp.weight = 2.3
        self.rewards.track_ang_vel_z_exp.weight = 0.6
        self.rewards.action_rate_l2.weight = -0.018


@configclass
class Go2StrafeRightRunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_skill_strafe_right"


STRAFE_RIGHT_SPEC = SkillSpec(
    skill_id="go2/velocity/strafe_right",
    robot="go2",
    gym_id="go2_skill_strafe_right",
    env_cfg_cls=Go2StrafeRightEnvCfg,
    runner_cfg_cls=Go2StrafeRightRunnerCfg,
    description="Side-step right skill for aligning with targets.",
    tags=("velocity", "strafe", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _STRAFE_RIGHT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _STRAFE_RIGHT_COMMANDS["resample"],
    },
)



register_skill(FORWARD_TROT_SPEC)
register_skill(TURN_LEFT_SPEC)
register_skill(TURN_RIGHT_SPEC)
register_skill(CRAWL_SPEC)
register_skill(FORWARD_LEFT_SPEC)
register_skill(FORWARD_RIGHT_SPEC)
register_skill(BACKWARD_SPEC)
register_skill(BACKWARD_LEFT_SPEC)
register_skill(BACKWARD_RIGHT_SPEC)
register_skill(STRAFE_LEFT_SPEC)
register_skill(STRAFE_RIGHT_SPEC)
