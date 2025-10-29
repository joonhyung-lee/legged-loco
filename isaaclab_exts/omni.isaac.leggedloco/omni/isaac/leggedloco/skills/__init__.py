"""Skill presets and registry accessors for legged-loco."""

from .base import SkillSpec, get_skill, list_skills, register_skill, skill_registry

# Import preset libraries so that they self-register with Gym.
from . import h1_velocity  # noqa: F401
from . import go2_velocity  # noqa: F401

__all__ = [
    "SkillSpec",
    "get_skill",
    "list_skills",
    "register_skill",
    "skill_registry",
]
