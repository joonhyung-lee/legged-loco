"""Train one or more low-level skills and export them as reusable policies."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train modular skills with RSL-RL and export them.")
    parser.add_argument(
        "--skill",
        dest="skills",
        action="append",
        help="Skill id to train (e.g. 'h1/velocity/walk_forward'). Repeat for multiple skills.",
    )
    parser.add_argument("--robot", type=str, default=None, help="Optional robot filter when listing skills.")
    parser.add_argument("--list_skills", action="store_true", help="Print available skills and exit.")
    parser.add_argument("--export_dir", type=str, default="skill_library", help="Directory where trained skills are stored.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO max iterations.")
    parser.add_argument("--history_length", type=int, default=0, help="Length of observation history buffer.")
    parser.add_argument("--use_cnn", action="store_true", default=None, help="Train with depth CNN policy variant.")
    parser.add_argument("--use_rnn", action="store_true", default=False, help="Train recurrent actor-critic.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def list_available_skills(robot: str | None) -> None:
    from omni.isaac.leggedloco.skills import list_skills

    specs = list_skills(robot=robot)
    if not specs:
        print("No skills registered." if robot is None else f"No skills registered for robot '{robot}'.")
        return

    print("Registered skills:")
    for spec in specs:
        tags = ", ".join(spec.tags)
        print(f"  - {spec.skill_id} (gym_id={spec.gym_id}) :: {spec.description or 'no description'}")
        if tags:
            print(f"      tags: {tags}")


def main() -> None:
    parser = build_arg_parser()
    args_cli = parser.parse_args()

    if args_cli.list_skills:
        list_available_skills(args_cli.robot)
        return

    if not args_cli.skills:
        parser.error("Provide at least one --skill (use --list_skills to inspect options).")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Deferred imports until simulator is initialized.
    import gymnasium as gym
    import torch

    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    from rsl_rl.runners import OnPolicyRunner

    from omni.isaac.leggedloco.skills import get_skill
    from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper
    from omni.isaac.leggedloco.utils.skill_library import SkillLibrary

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    skill_library = SkillLibrary(args_cli.export_dir)

    try:
        for skill_id in args_cli.skills:
            spec = get_skill(skill_id)
            print(f"\n[INFO] === Training skill: {spec.skill_id} (gym: {spec.gym_id}) ===")

            env_cfg = parse_env_cfg(spec.gym_id, num_envs=args_cli.num_envs)
            agent_cfg = cli_args.parse_rsl_rl_cfg(spec.gym_id, args_cli)

            if args_cli.max_iterations:
                agent_cfg.max_iterations = args_cli.max_iterations

            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, log_dir)

            env = gym.make(spec.gym_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            try:
                if args_cli.video:
                    video_kwargs = {
                        "video_folder": os.path.join(log_dir, "videos"),
                        "step_trigger": lambda step: step % args_cli.video_interval == 0,
                        "video_length": args_cli.video_length,
                        "disable_logger": True,
                    }
                    print("[INFO] Recording videos during training.")
                    print_dict(video_kwargs, nesting=4)
                    env = gym.wrappers.RecordVideo(env, **video_kwargs)

                if args_cli.history_length > 0:
                    env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
                else:
                    env = RslRlVecEnvWrapper(env)

                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
                runner.add_git_repo_to_log(__file__)

                if agent_cfg.resume:
                    resume_path = get_checkpoint_path(
                        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name),
                        agent_cfg.load_run,
                        agent_cfg.load_checkpoint,
                    )
                    print(f"[INFO] Resuming checkpoint: {resume_path}")
                    runner.load(resume_path)

                env.seed(agent_cfg.seed)

                dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
                dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
                dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

                policy_path = skill_library.export(
                    spec=spec,
                    runner=runner,
                    agent_cfg=agent_cfg,
                    env_cfg=env_cfg,
                    log_dir=log_dir,
                )
                print(f"[INFO] Skill '{spec.skill_id}' exported to {policy_path}")
            finally:
                env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
