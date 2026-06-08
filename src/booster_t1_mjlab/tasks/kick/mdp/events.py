from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Shared kick-state helpers
# ---------------------------------------------------------------------------

def _ensure_kick_state(env: "ManagerBasedRlEnv") -> None:
    if not hasattr(env, "_kick_timer"):
        N = env.num_envs
        env._kick_timer = torch.zeros(N, dtype=torch.long, device=env.device)
        env._kick_world_shot_angle = torch.zeros(N, device=env.device)
        env._kick_target_speed = torch.zeros(N, device=env.device)
        env._kick_ball_vel_at_kick = torch.zeros(N, 3, device=env.device)


def _robot_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _place_ball_ahead_of_robot(
    env: "ManagerBasedRlEnv",
    ball: Entity,
    env_ids: torch.Tensor,
    robot_pos_w: torch.Tensor,   # [N, 3] already-computed world position
    yaw: torch.Tensor,            # [N] already-computed yaw
    distance_range: tuple[float, float],
    y_range: tuple[float, float],
    ball_radius: float,
) -> None:
    """Place ball in front of given robot pose (world frame)."""
    n = len(env_ids)
    dist  = sample_uniform(*distance_range, (n,), env.device)
    y_off = sample_uniform(*y_range, (n,), env.device)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    ball_pos = torch.stack([
        robot_pos_w[:, 0] + dist * cos_yaw - y_off * sin_yaw,
        robot_pos_w[:, 1] + dist * sin_yaw + y_off * cos_yaw,
        env.scene.env_origins[env_ids, 2] + ball_radius,
    ], dim=-1)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball_vel = torch.zeros((n, 6), device=env.device)
    ball.write_root_state_to_sim(
        torch.cat([ball_pos, ball_quat, ball_vel], dim=-1),
        env_ids=env_ids,
    )


# ---------------------------------------------------------------------------
# Combined robot + ball reset (replaces reset_base for kick task)
# ---------------------------------------------------------------------------

def reset_robot_and_ball(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    ball_name: str,
    ball_distance_range: tuple[float, float] = (0.5, 1.5),
    ball_y_range: tuple[float, float] = (-0.3, 0.3),
    ball_radius: float = 0.11,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot pose (same as reset_root_state_uniform) and place ball in
    front of the NEW spawn position in one shot — no stale data issue."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity  = env.scene[ball_name]

    # --- sample robot pose (mirrors reset_root_state_uniform) ---
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), env.device)

    default_state = robot.data.default_root_state[env_ids].clone()
    positions = default_state[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    orient_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = quat_mul(default_state[:, 3:7], orient_delta)

    if velocity_range is None:
        velocity_range = {}
    vel_range_list = [velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=env.device)
    vel_samples = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), env.device)
    velocities = default_state[:, 7:13] + vel_samples

    robot.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

    # --- place ball using the pose we just computed (no stale data) ---
    # extract yaw from the sampled orientation delta (yaw = pose_samples[:, 5])
    spawn_yaw = pose_samples[:, 5]
    _place_ball_ahead_of_robot(
        env, ball, env_ids,
        robot_pos_w=positions,
        yaw=spawn_yaw,
        distance_range=ball_distance_range,
        y_range=ball_y_range,
        ball_radius=ball_radius,
    )


# ---------------------------------------------------------------------------
# Mid-episode kick-cycle manager
# ---------------------------------------------------------------------------

def reset_play_kick_timer(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """Clear the play-mode kick timer on episode reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if hasattr(env, "_play_kick_timer"):
        env._play_kick_timer[env_ids] = 0


def reset_kick_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    shot_angle_offset_range: tuple[float, float] = (-math.pi / 3, math.pi / 3),
    target_speed_range: tuple[float, float] = (2.0, 8.0),
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    _ensure_kick_state(env)
    n = len(env_ids)
    offset = sample_uniform(*shot_angle_offset_range, (n,), env.device)
    env._kick_world_shot_angle[env_ids] = _robot_yaw(env)[env_ids] + offset
    env._kick_target_speed[env_ids] = sample_uniform(*target_speed_range, (n,), env.device)
    env._kick_timer[env_ids] = 0
    env._kick_ball_vel_at_kick[env_ids] = 0.0


def kick_cycle_step(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    ball_name: str = "ball",
    speed_threshold: float = 1.5,
    reset_delay_steps: int = 10,
    ball_reset_prob: float = 0.9,
    distance_range: tuple[float, float] = (0.5, 1.5),
    y_range: tuple[float, float] = (-0.3, 0.3),
    ball_radius: float = 0.11,
    shot_angle_offset_range: tuple[float, float] = (-math.pi / 3, math.pi / 3),
    target_speed_range: tuple[float, float] = (2.0, 8.0),
) -> None:
    """Per-step kick-cycle manager.

    Detects kick contact, waits 200ms, then resets ball in front of robot
    (90%) or leaves it for the robot to chase (10%).
    """
    _ensure_kick_state(env)
    ball: Entity = env.scene[ball_name]

    ball_lin_vel = ball.data.root_link_lin_vel_w
    ball_speed = torch.norm(ball_lin_vel, dim=-1)

    just_kicked = (ball_speed > speed_threshold) & (env._kick_timer == 0)
    if just_kicked.any():
        env._kick_ball_vel_at_kick[just_kicked] = ball_lin_vel[just_kicked].clone()
        env._kick_timer[just_kicked] = 1

    env._kick_timer[env._kick_timer > 0] += 1

    expired_mask = env._kick_timer > reset_delay_steps
    if not expired_mask.any():
        return

    expired_ids = expired_mask.nonzero(as_tuple=False).squeeze(-1)
    n_exp = len(expired_ids)

    do_full = torch.rand(n_exp, device=env.device) < ball_reset_prob
    full_ids = expired_ids[do_full]

    if len(full_ids) > 0:
        robot_pos = env.scene["robot"].data.root_link_pos_w[full_ids]
        yaw = _robot_yaw(env)[full_ids]
        _place_ball_ahead_of_robot(
            env, ball, full_ids, robot_pos, yaw, distance_range, y_range, ball_radius
        )

    offset = sample_uniform(*shot_angle_offset_range, (n_exp,), env.device)
    env._kick_world_shot_angle[expired_ids] = _robot_yaw(env)[expired_ids] + offset
    env._kick_target_speed[expired_ids] = sample_uniform(*target_speed_range, (n_exp,), env.device)
    env._kick_timer[expired_ids] = 0
    env._kick_ball_vel_at_kick[expired_ids] = 0.0
