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
        env._kick_count = torch.zeros(N, dtype=torch.long, device=env.device)


def _robot_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _place_ball_around_robot(
    env: "ManagerBasedRlEnv",
    ball: Entity,
    env_ids: torch.Tensor,
    robot_pos_w: torch.Tensor,
    distance_range: tuple[float, float],
    ball_radius: float,
    ball_vel_max: float = 0.0,
) -> None:
    """Polar ball spawn: front hemisphere (±90°) around world +x axis.

    Robot always respawns facing near world +x (yaw ±0.15 rad), so world-frame
    ±π/2 reliably places ball in the robot's front hemisphere without needing
    the (potentially stale) robot yaw from entity data.
    """
    n = len(env_ids)
    dist  = sample_uniform(*distance_range, (n,), env.device)
    angle = sample_uniform(-math.pi / 2, math.pi / 2, (n,), env.device)

    ball_pos = torch.stack([
        robot_pos_w[:, 0] + dist * torch.cos(angle),
        robot_pos_w[:, 1] + dist * torch.sin(angle),
        env.scene.env_origins[env_ids, 2] + ball_radius,
    ], dim=-1)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball_vel = torch.zeros((n, 6), device=env.device)
    if ball_vel_max > 0.0:
        a = torch.rand(n, device=env.device) * (2.0 * math.pi)
        s = torch.rand(n, device=env.device) * ball_vel_max
        ball_vel[:, 0] = s * torch.cos(a)
        ball_vel[:, 1] = s * torch.sin(a)
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
    ball_radius: float = 0.11,
    ball_vel_max: float = 0.0,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot pose and place ball at a random angle/distance around robot."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity  = env.scene[ball_name]

    # --- sample robot pose ---
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

    # --- polar ball spawn: anywhere around robot ---
    _place_ball_around_robot(
        env, ball, env_ids,
        robot_pos_w=positions,
        distance_range=ball_distance_range,
        ball_radius=ball_radius,
        ball_vel_max=ball_vel_max,
    )


# ---------------------------------------------------------------------------
# Fixed ball spawn for testing (play mode with --fixed-ball-offset)
# ---------------------------------------------------------------------------

def reset_robot_and_ball_fixed_offset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    ball_name: str,
    ball_offset_m: float = 0.4,
    ball_radius: float = 0.11,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot and place ball at a fixed distance directly in front.

    Used in play mode with --fixed-ball-offset to test specific distances.
    Ball is placed along the robot's spawn yaw direction (not world +x).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity  = env.scene[ball_name]

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

    # Place ball directly in front of robot using its spawn yaw
    q = orientations  # (N, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    n = len(env_ids)
    ball_xy = positions[:, :2] + torch.stack([
        torch.cos(yaw) * ball_offset_m,
        torch.sin(yaw) * ball_offset_m,
    ], dim=-1)
    ball_pos = torch.stack([
        ball_xy[:, 0],
        ball_xy[:, 1],
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


def set_fixed_kick_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    world_angle_rad: float = 0.0,
    target_speed: float = 3.0,
) -> None:
    """Set a fixed world-frame kick angle and speed — no robot-yaw dependency.

    Use this in play mode so the arrow stays constant across episodes.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    _ensure_kick_state(env)
    env._kick_world_shot_angle[env_ids] = world_angle_rad
    env._kick_target_speed[env_ids] = target_speed
    env._kick_timer[env_ids] = 0
    env._kick_ball_vel_at_kick[env_ids] = 0.0
    env._kick_count[env_ids] = 0
    env._play_mode = True


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
    env._kick_count[env_ids] = 0


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
    ball_vel_max: float = 0.0,
    shot_angle_offset_range: tuple[float, float] = (-math.pi / 3, math.pi / 3),
    target_speed_range: tuple[float, float] = (2.0, 8.0),
    angle_resample_prob: float = 0.0,
    min_episode_steps: int = 50,
) -> None:
    """Per-step kick-cycle manager.

    Detects kick contact, waits reset_delay_steps, then resets ball in front
    of robot (ball_reset_prob) or leaves it for the robot to chase.

    angle_resample_prob: per-step probability of resampling kick direction
    outside of a kick window. Teaches robot to reposition mid-episode.

    min_episode_steps: ignore ball speed in the first N steps after reset.
    Joint velocity randomization causes limbs to flail through the ball
    immediately after reset, producing false kick detections.
    """
    _ensure_kick_state(env)
    ball: Entity = env.scene[ball_name]

    ball_lin_vel = ball.data.root_link_lin_vel_w
    ball_speed = torch.norm(ball_lin_vel, dim=-1)

    # Gate: don't count kicks in the first min_episode_steps after reset.
    # Reset joint-velocity randomization causes limbs to briefly flail through
    # the ball before the robot stabilises, giving false high ball speeds.
    episode_step = (env.episode_length_buf).long()
    past_warmup = episode_step >= min_episode_steps

    # Play mode print must run BEFORE just_kicked sets the timer to 1.
    # If placed after, _kick_timer[0] is already 1 for any real kick so the
    # condition == 0 fails and hard kicks never print.
    if getattr(env, "_play_mode", False) and ball_speed.shape[0] > 0:
        s = ball_speed[0].item()
        if s > 0.3 and env._kick_timer[0].item() == 0:
            t = env._kick_target_speed[0].item()
            tag = "[KICK]" if s > speed_threshold else "[soft]"
            print(f"{tag} ball_speed={s:.2f} m/s  target={t:.2f} m/s  error={s-t:+.2f}")

    just_kicked = (ball_speed > speed_threshold) & (env._kick_timer == 0) & past_warmup
    if just_kicked.any():
        env._kick_ball_vel_at_kick[just_kicked] = ball_lin_vel[just_kicked].clone()
        env._kick_timer[just_kicked] = 1
        env._kick_count[just_kicked] += 1
        env.extras["log"]["Metrics/ball_speed_at_kick"] = ball_speed[just_kicked].mean()

        # Direction error: angle between snapshot ball velocity and commanded kick direction
        ball_vel_xy = ball_lin_vel[just_kicked, :2]
        ball_spd_xy = torch.norm(ball_vel_xy, dim=-1).clamp(min=1e-6)
        ball_dir = ball_vel_xy / ball_spd_xy.unsqueeze(-1)
        kick_dir_xy = torch.stack([
            torch.cos(env._kick_world_shot_angle[just_kicked]),
            torch.sin(env._kick_world_shot_angle[just_kicked]),
        ], dim=-1)
        cos_sim = torch.sum(ball_dir * kick_dir_xy, dim=-1).clamp(-1.0, 1.0)
        env.extras["log"]["Metrics/kick_direction_error_deg"] = (
            torch.acos(cos_sim) * (180.0 / math.pi)
        ).mean()

        # Speed error: how far off from commanded target speed
        env.extras["log"]["Metrics/kick_speed_error"] = (
            (ball_speed[just_kicked] - env._kick_target_speed[just_kicked]).abs().mean()
        )

    env._kick_timer[env._kick_timer > 0] += 1

    # Mid-episode angle resample: small probability each step outside kick window
    if angle_resample_prob > 0.0:
        can_resample = env._kick_timer == 0
        do_resample = can_resample & (torch.rand(env.num_envs, device=env.device) < angle_resample_prob)
        if do_resample.any():
            resample_ids = do_resample.nonzero(as_tuple=False).squeeze(-1)
            offset = sample_uniform(*shot_angle_offset_range, (len(resample_ids),), env.device)
            env._kick_world_shot_angle[resample_ids] = _robot_yaw(env)[resample_ids] + offset

    expired_mask = env._kick_timer > reset_delay_steps
    if not expired_mask.any():
        return

    expired_ids = expired_mask.nonzero(as_tuple=False).squeeze(-1)
    n_exp = len(expired_ids)

    do_full = torch.rand(n_exp, device=env.device) < ball_reset_prob
    full_ids = expired_ids[do_full]

    if len(full_ids) > 0:
        robot_pos = env.scene["robot"].data.root_link_pos_w[full_ids]
        _place_ball_around_robot(
            env, ball, full_ids, robot_pos, distance_range, ball_radius, ball_vel_max
        )

    # In play mode, keep the fixed angle set by set_fixed_kick_state.
    # Without this guard, angle = _robot_yaw() + 0 which tracks robot heading.
    if not getattr(env, "_play_mode", False):
        offset = sample_uniform(*shot_angle_offset_range, (n_exp,), env.device)
        env._kick_world_shot_angle[expired_ids] = _robot_yaw(env)[expired_ids] + offset
    env._kick_target_speed[expired_ids] = sample_uniform(*target_speed_range, (n_exp,), env.device)
    env._kick_timer[expired_ids] = 0
    env._kick_ball_vel_at_kick[expired_ids] = 0.0
