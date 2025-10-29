"""Utilities for exporting trained skills into a reusable library."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

from omni.isaac.leggedloco.skills.base import SkillSpec


class SkillLibrary:
    """Filesystem-backed registry for trained skill checkpoints."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _skill_dir(self, spec: SkillSpec) -> Path:
        path = self.root_dir
        for part in spec.skill_id.split("/"):
            path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export(
        self,
        *,
        spec: SkillSpec,
        runner,
        agent_cfg,
        env_cfg,
        log_dir: str | Path,
        policy_filename: str = "policy.pt",
        extras: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the policy checkpoint and configs for *spec*.

        Args:
            spec: Skill specification describing the skill.
            runner: Active :class:`OnPolicyRunner` used for training.
            agent_cfg: PPO runner configuration dataclass.
            env_cfg: Environment configuration dataclass.
            log_dir: Training log directory.
            policy_filename: Optional custom filename for the exported policy.
            extras: Additional JSON-serializable payload to merge in metadata.

        Returns:
            Path to the exported policy file.
        """
        skill_dir = self._skill_dir(spec)
        policy_path = skill_dir / policy_filename
        runner.save(str(policy_path))

        # Dump configs for reproducibility.
        dump_yaml(skill_dir / "agent.yaml", agent_cfg)
        dump_yaml(skill_dir / "env.yaml", env_cfg)
        dump_pickle(skill_dir / "agent.pkl", agent_cfg)
        dump_pickle(skill_dir / "env.pkl", env_cfg)

        metadata = spec.to_metadata()
        metadata.update(
            {
                "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "policy": policy_path.name,
                "log_dir": str(log_dir),
                "iterations": getattr(runner, "current_learning_iteration", None),
            }
        )
        if extras:
            metadata.update(extras)

        with (skill_dir / "metadata.json").open("w", encoding="utf-8") as outfile:
            json.dump(metadata, outfile, indent=2, sort_keys=True)

        return policy_path


__all__ = ["SkillLibrary"]
