"""
Sequentially train multiple skills into a single policy checkpoint.
python scripts/train_multiskill.py --list_skills --headless
python scripts/train_multiskill.py --skill=go2/velocity/turn_right --skill=go2/manipulation/crawl_low --headless

python scripts/train_multiskill.py \
  --skill=go2/velocity/trot_forward \
  --skill=go2/velocity/turn_left \
  --skill=go2/velocity/turn_right \
  --skill=go2/manipulation/crawl_low \
  --skill=go2/velocity/forward_left \
  --skill=go2/velocity/forward_right \
  --skill=go2/velocity/walk_backward \
  --skill=go2/velocity/backward_left \
  --skill=go2/velocity/backward_right \
  --skill=go2/velocity/strafe_left \
  --skill=go2/velocity/strafe_right \
  --iterations_per_skill=3000 \
  --max_iterations=10000 \
  --num_envs=4096 \
  --headless
  --use_wandb \
  --wandb_project="legged-loco" \
  --wandb_entity="rebel-joonhyung-lee-rebellions" \
  --wandb_run_name="multiskill-experiment-3" \

# With Wandb Logging
python scripts/train_multiskill.py \
  --skill=go2/velocity/trot_forward \
  --skill=go2/velocity/turn_left \
  --skill=go2/velocity/turn_right \
  --use_wandb \
  --wandb_project="legged-loco" \
  --wandb_entity="rebel-joonhyung-lee-rebellions" \
  --wandb_run_name="multiskill-experiment-1" \
  --num_envs=256 \
  --headless

python scripts/train_multiskill.py \
  --skill=go2/velocity/trot_forward \
  --skill=go2/velocity/turn_left \
  --skill=go2/velocity/turn_right \
  --use_wandb \
  --wandb_project="legged-loco" \
  --wandb_entity="rebel-joonhyung-lee-rebellions" \
  --wandb_run_name="multiskill-experiment-2" \
  --num_envs=512 \
  --headless
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multiple skills sequentially into a single policy.")
    parser.add_argument(
        "--skill",
        dest="skills",
        action="append",
        help="Skill id to include (repeat for multiple). Order defines training schedule.",
    )
    parser.add_argument("--list_skills", action="store_true", help="List registered skills and exit.")
    parser.add_argument(
        "--combined_skill_id",
        type=str,
        default=None,
        help="Skill id used when exporting the final merged policy (default: <robot>/multi/<timestamp>).",
    )
    parser.add_argument(
        "--iterations_per_skill",
        type=int,
        default=None,
        help="Override number of PPO iterations per skill (applied after CLI overrides).",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="skill_library",
        help="Directory to store the merged skill checkpoint.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Video capture interval (steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO max iterations globally.")
    parser.add_argument("--history_length", type=int, default=0, help="Observation history length.")
    parser.add_argument("--use_cnn", action="store_true", default=None, help="Use depth CNN policy variant.")
    parser.add_argument("--use_rnn", action="store_true", default=False, help="Use recurrent actor-critic.")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="legged-loco-multiskill", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def list_registered_skills() -> None:
    from omni.isaac.leggedloco.skills import list_skills

    specs = list_skills()
    if not specs:
        print("No skills registered.")
        return

    print("Registered skills:")
    for spec in specs:
        tags = ", ".join(spec.tags)
        print(f"  - {spec.skill_id} (robot={spec.robot}, gym_id={spec.gym_id})")
        if tags:
            print(f"      tags: {tags}")


def main() -> None:
    parser = build_arg_parser()
    args_cli = parser.parse_args()

    # AppLauncher를 먼저 시작 (Isaac Sim 런타임 로드를 위해)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    if args_cli.list_skills:
        list_registered_skills()
        simulation_app.close()
        return

    if not args_cli.skills or len(args_cli.skills) < 2:
        parser.error("Provide at least two --skill entries to train a merged policy.")

    # Deferred heavy imports until simulator is live.
    import gymnasium as gym
    import torch

    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    from rsl_rl.runners import OnPolicyRunner

    from omni.isaac.leggedloco.skills import SkillSpec, get_skill
    from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper
    from omni.isaac.leggedloco.utils.skill_library import SkillLibrary

    # Initialize wandb if requested
    if args_cli.use_wandb:
        import wandb
        wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_run_name or f"multiskill_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args_cli),
            tags=["multiskill", "legged-loco"] + [skill.split('/')[0] for skill in args_cli.skills]
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    skill_specs: List[SkillSpec] = [get_skill(skill_id) for skill_id in args_cli.skills]

    robots = {spec.robot for spec in skill_specs}
    if len(robots) != 1:
        raise ValueError(f"All skills must target the same robot. Received robots: {sorted(robots)}")
    robot_name = robots.pop()

    final_runner: OnPolicyRunner | None = None
    final_env_cfg = None
    final_agent_cfg = None
    final_log_dir = None
    last_checkpoint = None

    skill_library = SkillLibrary(args_cli.export_dir)

    try:
        for idx, spec in enumerate(skill_specs):
            print(f"\n[INFO] === Training stage {idx + 1}/{len(skill_specs)}: {spec.skill_id} ===")
            
            # Log current skill to wandb
            if args_cli.use_wandb:
                wandb.log({"current_skill": spec.skill_id, "skill_stage": idx + 1, "total_skills": len(skill_specs)})

            env_cfg = parse_env_cfg(spec.gym_id, num_envs=args_cli.num_envs)
            agent_cfg = cli_args.parse_rsl_rl_cfg(spec.gym_id, args_cli)

            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            if args_cli.iterations_per_skill is not None:
                agent_cfg.max_iterations = args_cli.iterations_per_skill

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
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

                if last_checkpoint:
                    print(f"[INFO] Loading previous stage checkpoint: {last_checkpoint}")
                    runner.load(last_checkpoint)

                env.seed(agent_cfg.seed)

                dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
                dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
                dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

                # Custom training loop with wandb logging
                if args_cli.use_wandb:
                    # Override the learn method to add wandb logging
                    original_learn = runner.learn
                    
                    def learn_with_wandb(num_learning_iterations, init_at_random_ep_len=True):
                        for it in range(num_learning_iterations):
                            # Run one iteration
                            runner.collect_rollout()
                            runner.update()
                            
                            # Log metrics to wandb
                            if it % 10 == 0:  # Log every 10 iterations
                                metrics = {
                                    f"skill_{idx+1}_{spec.skill_id}/iteration": it,
                                    f"skill_{idx+1}_{spec.skill_id}/value_loss": runner.value_loss,
                                    f"skill_{idx+1}_{spec.skill_id}/surrogate_loss": runner.surrogate_loss,
                                    f"skill_{idx+1}_{spec.skill_id}/mean_reward": runner.mean_reward,
                                    f"skill_{idx+1}_{spec.skill_id}/mean_episode_length": runner.mean_episode_length,
                                    f"skill_{idx+1}_{spec.skill_id}/total_timesteps": runner.total_timesteps,
                                }
                                
                                # Add reward components if available
                                if hasattr(runner, 'reward_components'):
                                    for key, value in runner.reward_components.items():
                                        metrics[f"skill_{idx+1}_{spec.skill_id}/reward_{key}"] = value
                                
                                # Add curriculum metrics if available
                                if hasattr(runner, 'curriculum_metrics'):
                                    for key, value in runner.curriculum_metrics.items():
                                        metrics[f"skill_{idx+1}_{spec.skill_id}/curriculum_{key}"] = value
                                
                                wandb.log(metrics)
                    
                    runner.learn = learn_with_wandb
                
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

                last_checkpoint = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
                print(f"[INFO] Stage '{spec.skill_id}' checkpoint saved to {last_checkpoint}")

                final_runner = runner
                final_env_cfg = env_cfg
                final_agent_cfg = agent_cfg
                final_log_dir = log_dir

            finally:
                env.close()

    finally:
        simulation_app.close()

    if final_runner is None or final_env_cfg is None or final_agent_cfg is None or final_log_dir is None:
        raise RuntimeError("Training did not produce a final runner.")

    combined_skill_id = args_cli.combined_skill_id
    if combined_skill_id is None:
        combined_skill_id = f"{robot_name}/multi/{datetime.now().strftime('%Y%m%d%H%M%S')}"

    combined_spec = SkillSpec(
        skill_id=combined_skill_id,
        robot=robot_name,
        gym_id=skill_specs[-1].gym_id,
        env_cfg_cls=skill_specs[-1].env_cfg_cls,
        runner_cfg_cls=skill_specs[-1].runner_cfg_cls,
        description=f"Merged policy trained sequentially on: {', '.join(args_cli.skills)}",
        tags=("multi", "sequential"),
        metadata={"skills": args_cli.skills},
    )

    policy_path = skill_library.export(
        spec=combined_spec,
        runner=final_runner,
        agent_cfg=final_agent_cfg,
        env_cfg=final_env_cfg,
        log_dir=final_log_dir,
    )

    print("\n[INFO] Combined skill exported to:", policy_path)
    print("[INFO] Metadata includes the training order for reproducibility.")
    
    # Finish wandb run
    if args_cli.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()