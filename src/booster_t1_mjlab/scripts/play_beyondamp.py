"""Play a beyondAMP policy on the Booster T1 (mjlab backend).

Usage:
    uv run booster_t1_play_beyondamp Mjlab-BeyondAMP-Velocity-Flat-Booster-T1 \\
        --checkpoint-file logs/rsl_rl/t1_amp_locomotion/<run>/model_500.pt

    # Zero-action sanity check (no checkpoint needed):
    uv run booster_t1_play_beyondamp Mjlab-BeyondAMP-Velocity-Flat-Booster-T1 \\
        --agent zero
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    no_terminations: bool = False
    viewer: Literal["auto", "native", "viser"] = "auto"
    fixed_ball_offset_m: float | None = None
    """If set, spawn ball this many metres directly in front of robot every reset (kick tasks only)."""


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AMPRunnerCfg), (
        f"Task '{task_id}' is not an AMP task — got {type(agent_cfg).__name__}."
    )

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")
    if cfg.fixed_ball_offset_m is not None and hasattr(env_cfg, "events") and "reset_base" in env_cfg.events:
        from mjlab.managers.event_manager import EventTermCfg
        from booster_t1_mjlab.tasks.kick.mdp.events import reset_robot_and_ball_fixed_offset
        env_cfg.events["reset_base"] = EventTermCfg(
            func=reset_robot_and_ball_fixed_offset,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-0.15, 0.15)},
                "velocity_range": {},
                "ball_name": "ball",
                "ball_offset_m": cfg.fixed_ball_offset_m,
                "ball_radius": 0.11,
            },
        )
        print(f"[INFO] Fixed ball spawn: {cfg.fixed_ball_offset_m:.2f} m in front of robot")

    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

        if cfg.agent == "zero":
            def policy(obs: torch.Tensor) -> torch.Tensor:
                return torch.zeros(action_shape, device=device)
        else:
            def policy(obs: torch.Tensor) -> torch.Tensor:
                return 2 * torch.rand(action_shape, device=device) - 1
    else:
        if cfg.checkpoint_file is None:
            raise ValueError("--checkpoint-file is required for --agent trained")
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[INFO] Loading checkpoint: {resume_path.name}")
        runner = AMPOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
        runner.load(str(resume_path), load_optimizer=False)
        policy = runner.get_inference_policy(device=device)

    # Wrap policy to print ball position for kick tasks
    base_env = env.unwrapped
    if "ball" in base_env.scene.entities:
        _step = [0]
        _inner_policy = policy
        def policy(obs: torch.Tensor) -> torch.Tensor:
            action = _inner_policy(obs)
            ball_pos = base_env.scene["ball"].data.root_link_pos_w[0]
            robot_pos = base_env.scene["robot"].data.root_link_pos_w[0]
            rel_w = (ball_pos - robot_pos)[:2]
            # Rotate world-frame delta into robot body frame using robot yaw
            q = base_env.scene["robot"].data.root_link_quat_w[0]
            yaw = torch.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
            c, s = torch.cos(yaw), torch.sin(yaw)
            rx = c * rel_w[0] + s * rel_w[1]   # forward
            ry = -s * rel_w[0] + c * rel_w[1]  # left
            print(f"[step {_step[0]:5d}] ball world=({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f})  "
                  f"rel_robot=({rx:.2f}, {ry:.2f})  dist={rel_w.norm():.2f}m")
            _step[0] += 1
            return action

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {resolved_viewer}")

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import booster_t1_mjlab.tasks  # noqa: F401
    import amp_tasks_mjlab  # noqa: F401

    import mjlab

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]
    if not amp_tasks:
        raise RuntimeError("No AMP tasks registered.")

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(amp_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
