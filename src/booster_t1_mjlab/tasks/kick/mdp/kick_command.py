"""Visualization-only command term for kick direction + target speed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class KickDirectionCommand(CommandTerm):
    """Visualization-only command term for kick direction and target speed.

    The actual kick state (_kick_world_shot_angle, _kick_target_speed,
    _kick_timer) is managed by the reset_kick_state and kick_cycle_step
    events — this term only reads that state and draws arrows in the viewer.
    The command tensor is empty (zero columns) so it never contributes to
    any observation group.
    """

    cfg: "KickDirectionCommandCfg"

    def __init__(self, cfg: "KickDirectionCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self._ball_name: str = cfg.ball_name

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # state owned by reset_kick_state / kick_cycle_step events

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        env = self._env
        if not hasattr(env, "_kick_world_shot_angle"):
            return

        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        shot_angles = env._kick_world_shot_angle.cpu().numpy()
        target_speeds = env._kick_target_speed.cpu().numpy()

        ball_pos_ws = None
        if self._ball_name and self._ball_name in env.scene.entities:
            ball_pos_ws = env.scene[self._ball_name].data.root_link_pos_w.cpu().numpy()

        Z_HEIGHT = 0.05   # just above ground
        SCALE = 0.25      # metres per m/s of target speed
        APPROACH_DIST = 0.35  # must match approach_kick_position reward param

        for i in env_indices:
            pos = base_pos_ws[i]
            angle = shot_angles[i]
            speed = target_speeds[i]

            if np.linalg.norm(pos) < 1e-6:
                continue

            kick_dir_w = np.array([np.cos(angle), np.sin(angle), 0.0])
            kick_perp_w = np.array([-kick_dir_w[1], kick_dir_w[0], 0.0])
            root = np.array([pos[0], pos[1], Z_HEIGHT])
            tip = root + kick_dir_w * speed * SCALE

            # Orange arrow: kick direction, length = target speed
            visualizer.add_arrow(root, tip, color=(1.0, 0.5, 0.0, 0.9), width=0.02)
            visualizer.add_sphere(root, radius=0.03, color=(1.0, 1.0, 1.0, 0.8))

            if ball_pos_ws is not None:
                bpos = ball_pos_ws[i]
                kd = kick_dir_w[:2]
                kp = kick_perp_w[:2]
                ball_xy = bpos[:2]
                robot_xy = pos[:2]

                # Green sphere: kick position marker (ball - kick_dir * approach_dist)
                approach_target = np.array([
                    bpos[0] - kd[0] * APPROACH_DIST,
                    bpos[1] - kd[1] * APPROACH_DIST,
                    Z_HEIGHT,
                ])
                visualizer.add_sphere(approach_target, radius=0.06, color=(0.0, 1.0, 0.2, 0.85))

                # Cyan axis: approach target → ball
                ball_ground = np.array([bpos[0], bpos[1], Z_HEIGHT])
                visualizer.add_arrow(approach_target, ball_ground, color=(0.0, 0.8, 1.0, 0.7), width=0.015)

                # Red sphere: phase-2 drive-through target (ball + kick_dir * approach_dist)
                drive_through = np.array([
                    bpos[0] + kd[0] * APPROACH_DIST,
                    bpos[1] + kd[1] * APPROACH_DIST,
                    Z_HEIGHT,
                ])
                visualizer.add_sphere(drive_through, radius=0.04, color=(1.0, 0.2, 0.2, 0.75))
                visualizer.add_arrow(ball_ground, drive_through, color=(0.0, 0.8, 1.0, 0.5), width=0.012)

                # Yellow sphere + arrow: live phase-1 arc target (shifts as robot moves)
                rel = robot_xy - ball_xy
                along = np.dot(rel, -kd)
                perp  = np.dot(rel, kp)
                lateral = float(np.clip(perp, -0.5, 0.5)) * 0.4
                if along < 0.0:
                    lateral += np.sign(perp) * 0.25
                p1_target = np.array([
                    approach_target[0] + kp[0] * lateral,
                    approach_target[1] + kp[1] * lateral,
                    Z_HEIGHT,
                ])
                visualizer.add_sphere(p1_target, radius=0.04, color=(1.0, 1.0, 0.0, 0.85))
                visualizer.add_arrow(root, p1_target, color=(1.0, 1.0, 0.0, 0.4), width=0.01)


@dataclass(kw_only=True)
class KickDirectionCommandCfg(CommandTermCfg):
    """Config for the kick direction visualizer (no-op command, vis only)."""

    debug_vis: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # never resamples
    ball_name: str = "ball"

    def build(self, env: "ManagerBasedRlEnv") -> KickDirectionCommand:
        return KickDirectionCommand(self, env)
