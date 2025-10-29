"""Preset velocity tracking skills for the H1 robot."""

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from omni.isaac.leggedloco.config.h1.h1_low_base_cfg import H1BaseRoughEnvCfg, H1RoughPPORunnerCfg
from .base import SkillSpec, register_skill


# ---------------------------------------------------------------------------
# Shared metadata helpers
# ---------------------------------------------------------------------------
FINE_TERRAIN_METADATA = {
    "terrain": "rough_base",
    "randomization": {
        "base_mass": False,
        "push_robot": False,
    },
}

_FORWARD_WALK_COMMANDS = {
    "lin_vel_x": (0.5, 1.2),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-0.3, 0.3),
    "resample": (8.0, 8.0),
}

_TURN_LEFT_COMMANDS = {
    "lin_vel_x": (0.0, 0.3),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (0.4, 1.2),
    "resample": (6.0, 6.0),
}

_TURN_RIGHT_COMMANDS = {
    "lin_vel_x": (0.0, 0.3),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-1.2, -0.4),
    "resample": (6.0, 6.0),
}

_STAND_COMMANDS = {
    "lin_vel_x": (-0.05, 0.05),
    "lin_vel_y": (-0.05, 0.05),
    "ang_vel_z": (-0.1, 0.1),
    "resample": (12.0, 12.0),
}


# ---------------------------------------------------------------------------
# Environment + runner definitions per skill
# ---------------------------------------------------------------------------
@configclass
class H1ForwardWalkSkillEnvCfg(H1BaseRoughEnvCfg):
    """Forward velocity tracking skill with mild heading corrections."""

    def __post_init__(self):
        super().__post_init__()
        # Focus on forward motion with small lateral deviations.
        self.commands.base_velocity.ranges.lin_vel_x = _FORWARD_WALK_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _FORWARD_WALK_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _FORWARD_WALK_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _FORWARD_WALK_COMMANDS["resample"]

        # Encourage energetic but stable forward steps.
        self.rewards.feet_air_time.weight = 0.4
        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.flat_orientation_l2.weight = -6.0


@configclass
class H1ForwardWalkSkillRunnerCfg(H1RoughPPORunnerCfg):
    experiment_name = "h1_skill_forward_walk"


FORWARD_WALK_SPEC = SkillSpec(
    skill_id="h1/velocity/walk_forward",
    robot="h1",
    gym_id="h1_skill_walk_forward",
    env_cfg_cls=H1ForwardWalkSkillEnvCfg,
    runner_cfg_cls=H1ForwardWalkSkillRunnerCfg,
    description="H1 gait that aggressively tracks forward base velocities.",
    tags=("velocity", "locomotion", "forward"),
    metadata={
        "command_ranges": {k: v for k, v in _FORWARD_WALK_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _FORWARD_WALK_COMMANDS["resample"],
    } | FINE_TERRAIN_METADATA,
)


@configclass
class H1TurnLeftSkillEnvCfg(H1BaseRoughEnvCfg):
    """Spot turning skill that emphasises positive yaw velocity."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _TURN_LEFT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _TURN_LEFT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _TURN_LEFT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _TURN_LEFT_COMMANDS["resample"]

        # Make heading tracking and yaw velocity accuracy dominant.
        if hasattr(self.rewards, "tracking_ang_vel_z"):
            self.rewards.tracking_ang_vel_z.weight = 5.0
        if hasattr(self.rewards, "tracking_heading"):
            self.rewards.tracking_heading.weight = 6.0
        self.rewards.action_rate_l2.weight = -0.002


@configclass
class H1TurnLeftSkillRunnerCfg(H1RoughPPORunnerCfg):
    experiment_name = "h1_skill_turn_left"


TURN_LEFT_SPEC = SkillSpec(
    skill_id="h1/velocity/turn_left",
    robot="h1",
    gym_id="h1_skill_turn_left",
    env_cfg_cls=H1TurnLeftSkillEnvCfg,
    runner_cfg_cls=H1TurnLeftSkillRunnerCfg,
    description="H1 skill for strong-left yaw spins with minimal drift.",
    tags=("velocity", "yaw", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _TURN_LEFT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _TURN_LEFT_COMMANDS["resample"],
    } | FINE_TERRAIN_METADATA,
)

@configclass
class H1TurnRightSkillEnvCfg(H1BaseRoughEnvCfg):
    """Spot turning skill that emphasises negative yaw velocity."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _TURN_RIGHT_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _TURN_RIGHT_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _TURN_RIGHT_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _TURN_RIGHT_COMMANDS["resample"]

        if hasattr(self.rewards, "tracking_ang_vel_z"):
            self.rewards.tracking_ang_vel_z.weight = 5.0
        if hasattr(self.rewards, "tracking_heading"):
            self.rewards.tracking_heading.weight = 6.0
        self.rewards.action_rate_l2.weight = -0.002


@configclass
class H1TurnRightSkillRunnerCfg(H1RoughPPORunnerCfg):
    experiment_name = "h1_skill_turn_right"


TURN_RIGHT_SPEC = SkillSpec(
    skill_id="h1/velocity/turn_right",
    robot="h1",
    gym_id="h1_skill_turn_right",
    env_cfg_cls=H1TurnRightSkillEnvCfg,
    runner_cfg_cls=H1TurnRightSkillRunnerCfg,
    description="H1 skill for rightward yaw spins matching the left-turn preset.",
    tags=("velocity", "yaw", "locomotion"),
    metadata={
        "command_ranges": {k: v for k, v in _TURN_RIGHT_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _TURN_RIGHT_COMMANDS["resample"],
    } | FINE_TERRAIN_METADATA,
)



@configclass
class H1StandSkillEnvCfg(H1BaseRoughEnvCfg):
    """Balance-in-place skill that learns quiet stance and posture control."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = _STAND_COMMANDS["lin_vel_x"]
        self.commands.base_velocity.ranges.lin_vel_y = _STAND_COMMANDS["lin_vel_y"]
        self.commands.base_velocity.ranges.ang_vel_z = _STAND_COMMANDS["ang_vel_z"]
        self.commands.base_velocity.resampling_time_range = _STAND_COMMANDS["resample"]

        # Prioritize stability over swing behaviour.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -8.0
        if hasattr(self.rewards, "dof_torques_l2"):
            self.rewards.dof_torques_l2.weight = -0.0005


@configclass
class H1StandSkillRunnerCfg(H1RoughPPORunnerCfg):
    experiment_name = "h1_skill_stand"


STAND_SPEC = SkillSpec(
    skill_id="h1/posture/stand",
    robot="h1",
    gym_id="h1_skill_stand",
    env_cfg_cls=H1StandSkillEnvCfg,
    runner_cfg_cls=H1StandSkillRunnerCfg,
    description="Quiet standing stabilization skill for low-speed balance.",
    tags=("posture", "balance"),
    metadata={
        "command_ranges": {k: v for k, v in _STAND_COMMANDS.items() if k != "resample"},
        "resampling_time_range": _STAND_COMMANDS["resample"],
    } | FINE_TERRAIN_METADATA,
)


# Register presets so they become discoverable and gym-compatible.



register_skill(FORWARD_WALK_SPEC)
register_skill(TURN_LEFT_SPEC)
register_skill(TURN_RIGHT_SPEC)
register_skill(STAND_SPEC)
