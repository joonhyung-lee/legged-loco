"""Skill registry utilities for modular skill training.

This module centralizes metadata about skill-specific environments so that
training scripts and high-level controllers can query and export skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, Type

import gymnasium as gym
from gymnasium.error import Error as GymRegistrationError

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


@dataclass(frozen=True)
class SkillSpec:
    """Metadata describing a trainable skill.

    Attributes:
        skill_id: Canonical hierarchical identifier (e.g. ``"h1/velocity/walk_forward"``).
        robot: Robot family the skill targets (e.g. ``"h1"``).
        gym_id: Gymnasium environment id that exposes the skill during training/inference.
        env_cfg_cls: Manager-based environment configuration class used to build the env.
        runner_cfg_cls: RSL-RL runner configuration class associated with the env.
        description: Human-readable description.
        tags: Optional tuple of tags for discoverability (e.g. ("velocity", "locomotion")).
        metadata: Arbitrary metadata describing configuration tweaks (command ranges, reward gains, etc.).
    """

    skill_id: str
    robot: str
    gym_id: str
    env_cfg_cls: Type[ManagerBasedRLEnvCfg]
    runner_cfg_cls: Type[RslRlOnPolicyRunnerCfg]
    description: str = ""
    tags: Tuple[str, ...] = tuple()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "skill_id": self.skill_id,
            "robot": self.robot,
            "gym_id": self.gym_id,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class SkillRegistry:
    """Singleton-style registry storing :class:`SkillSpec` definitions."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillSpec] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register(self, spec: SkillSpec, *, force: bool = False, register_gym: bool = True) -> SkillSpec:
        """Register a :class:`SkillSpec` and optionally expose it via Gymnasium.

        Args:
            spec: Skill specification to add.
            force: Overwrite an existing entry if one exists.
            register_gym: If ``True`` (default) the underlying Gym environment
                id is registered immediately.
        """
        if spec.skill_id in self._skills and not force:
            raise ValueError(f"Skill '{spec.skill_id}' is already registered.")

        self._skills[spec.skill_id] = spec

        if register_gym:
            self._ensure_gym_registration(spec, force=force)

        return spec

    def _ensure_gym_registration(self, spec: SkillSpec, *, force: bool) -> None:
        """Register the Gymnasium id backing *spec* if needed."""
        if force:
            try:
                gym.envs.registry.pop(spec.gym_id)
            except KeyError:
                pass

        try:
            gym.register(
                id=spec.gym_id,
                entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": spec.env_cfg_cls,
                    "rsl_rl_cfg_entry_point": spec.runner_cfg_cls,
                },
            )
        except GymRegistrationError as exc:
            if "exists" not in str(exc).lower():
                raise

    # ------------------------------------------------------------------
    # Query utilities
    # ------------------------------------------------------------------
    def get(self, skill_id: str) -> SkillSpec:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"Unknown skill '{skill_id}'. Known skills: {sorted(self._skills)}") from exc

    def find_by_gym_id(self, gym_id: str) -> SkillSpec:
        for spec in self._skills.values():
            if spec.gym_id == gym_id:
                return spec
        raise KeyError(f"Unknown gym id '{gym_id}'.")

    def iter(self, *, robot: Optional[str] = None) -> Iterator[SkillSpec]:
        for spec in self._skills.values():
            if robot is None or spec.robot == robot:
                yield spec

    def list(self, *, robot: Optional[str] = None) -> List[SkillSpec]:
        return list(self.iter(robot=robot))


# Module-level singleton for simplicity.
skill_registry = SkillRegistry()


def register_skill(spec: SkillSpec, *, force: bool = False, register_gym: bool = True) -> SkillSpec:
    """Convenience wrapper around :meth:`SkillRegistry.register`."""
    return skill_registry.register(spec, force=force, register_gym=register_gym)


def get_skill(skill_id: str) -> SkillSpec:
    """Lookup a skill specification by id."""
    return skill_registry.get(skill_id)


def list_skills(*, robot: Optional[str] = None) -> List[SkillSpec]:
    """Return a list of registered skills, optionally filtered by robot."""
    return skill_registry.list(robot=robot)
