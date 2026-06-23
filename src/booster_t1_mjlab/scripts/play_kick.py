"""Interactive play script for the directed kick task.

Keyboard controls (in the MuJoCo viewer window):
  [   / ]   — rotate commanded kick angle  ±15 degrees
  UP  / DOWN — change target speed         ±0.5 m/s
  ENTER      — reset episode
  SPACE      — pause / resume

Usage:
    uv run booster_t1_play_kick --checkpoint-file logs/rsl_rl/t1_amp_kick/<run>/model_X.pt

The commanded angle and speed are printed to the terminal whenever they change.
The orange arrow in the viewer always shows the current command.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import (
    KEY_DOWN,
    KEY_LEFT_BRACKET,
    KEY_RIGHT_BRACKET,
    KEY_UP,
)

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner

TASK_ID = "Mjlab-AmpKick-Booster-T1-21Dof"

ANGLE_STEP_DEG = 15.0
SPEED_STEP = 0.5
SPEED_MIN = 1.0
SPEED_MAX = 8.0
ANGLE_RANGE_DEG = 90.0  # clamp to ±90° matching training


@dataclass(frozen=True)
class PlayKickConfig:
    checkpoint_file: str | None = None
    agent: Literal["zero", "trained"] = "trained"
    num_envs: int = 1
    device: str | None = None
    no_terminations: bool = False
    viewer: Literal["auto", "native", "viser"] = "auto"
    kick_angle_deg: float = 0.0
    """Initial kick direction in world-frame degrees (0 = east/+x, 90 = north/+y)."""
    kick_speed: float = 3.0
    """Initial target ball speed (m/s)."""


class KickController:
    """Shared mutable state updated by viewer key callback, read by step loop."""

    def __init__(self, angle_rad: float, speed: float):
        self.angle_rad = angle_rad
        self.speed = speed
        self._dirty = True  # print on first step

    def rotate(self, delta_deg: float) -> None:
        new = self.angle_rad + math.radians(delta_deg)
        new = max(min(new, math.radians(ANGLE_RANGE_DEG)), math.radians(-ANGLE_RANGE_DEG))
        self.angle_rad = new
        self._dirty = True

    def change_speed(self, delta: float) -> None:
        self.speed = max(min(self.speed + delta, SPEED_MAX), SPEED_MIN)
        self._dirty = True

    def maybe_print(self) -> None:
        if self._dirty:
            print(
                f"[kick] angle={math.degrees(self.angle_rad):+.1f}°  "
                f"speed={self.speed:.1f} m/s"
            )
            self._dirty = False


def run_play(cfg: PlayKickConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import booster_t1_mjlab.tasks  # noqa: F401 — register tasks
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, AMPRunnerCfg), (
        f"Expected AMPRunnerCfg for {TASK_ID}, got {type(agent_cfg).__name__}"
    )

    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled")

    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    if cfg.agent == "zero":
        action_shape = env.unwrapped.action_space.shape

        def policy(obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(action_shape, device=device)
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

    controller = KickController(
        angle_rad=math.radians(cfg.kick_angle_deg),
        speed=cfg.kick_speed,
    )
    N = env.unwrapped.num_envs

    # --- force kick state onto the env every step (beats reset_kick_state) ---
    _orig_step = env.step

    def _patched_step(actions: torch.Tensor):
        result = _orig_step(actions)
        env.unwrapped._kick_world_shot_angle = torch.full(
            (N,), controller.angle_rad, device=device
        )
        env.unwrapped._kick_target_speed = torch.full(
            (N,), controller.speed, device=device
        )
        controller.maybe_print()
        return result

    env.step = _patched_step  # type: ignore[method-assign]

    # --- key callback (runs on viewer thread — only writes Python scalars) ---
    def key_callback(key: int) -> None:
        if key == KEY_LEFT_BRACKET:
            controller.rotate(-ANGLE_STEP_DEG)
        elif key == KEY_RIGHT_BRACKET:
            controller.rotate(+ANGLE_STEP_DEG)
        elif key == KEY_UP:
            controller.change_speed(+SPEED_STEP)
        elif key == KEY_DOWN:
            controller.change_speed(-SPEED_STEP)

    print(
        "[kick] Controls: [ / ] = angle ±15°  |  ↑ / ↓ = speed ±0.5 m/s"
        "  |  ENTER = reset  |  SPACE = pause"
    )
    controller.maybe_print()

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy, key_callback=key_callback).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {resolved_viewer}")

    env.close()


def main() -> None:
    import mjlab

    args = tyro.cli(PlayKickConfig, config=mjlab.TYRO_FLAGS)
    run_play(args)


if __name__ == "__main__":
    main()
