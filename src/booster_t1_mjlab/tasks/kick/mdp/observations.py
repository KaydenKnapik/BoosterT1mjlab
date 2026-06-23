from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def ball_pos_xy_robot_frame(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Ball XY offset relative to foot midpoint, expressed in robot body frame.

    Origin: midpoint between left_foot_link and right_foot_link.
    Orientation: robot base yaw frame (x=forward, y=left).
    Returns [x_forward, y_left] — no height component.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    foot_pos = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :2]  # [B, 2, 2]
    feet_mid_xy = foot_pos.mean(dim=1)  # [B, 2]

    ball_xy = ball.data.root_link_pos_w[:, :2]  # [B, 2]

    rel_w = torch.zeros(feet_mid_xy.shape[0], 3, device=feet_mid_xy.device)
    rel_w[:, :2] = ball_xy - feet_mid_xy

    rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_w)  # [B, 3]
    return rel_b[:, :2]  # [B, 2]: [x_forward, y_left]


def ball_pos_xy_robot_frame_delayed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    min_delay_steps: int = 1,
    max_delay_steps: int = 4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Ball XY in robot frame with randomised per-episode observation delay.

    Computes ball_pos_xy_robot_frame normally then buffers the result.
    Each environment gets a delay sampled uniformly from
    [min_delay_steps, max_delay_steps] at episode reset, so the policy
    learns to handle the full range (at 50 Hz: 1 step=20ms, 4 steps=80ms).

    Critic should use ball_pos_xy_robot_frame (no delay) for privileged training.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    B = env.num_envs

    # --- compute current relative ball pos (same as undelayed version) ---
    foot_pos = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :2]
    feet_mid_xy = foot_pos.mean(dim=1)
    ball_xy = ball.data.root_link_pos_w[:, :2]
    rel_w = torch.zeros(B, 3, device=env.device)
    rel_w[:, :2] = ball_xy - feet_mid_xy
    rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_w)
    current_obs = rel_b[:, :2]  # [B, 2]

    # --- initialise buffer and per-env delay on first call ---
    if not hasattr(env, "_ball_obs_buf"):
        env._ball_obs_buf = current_obs.unsqueeze(1).expand(-1, max_delay_steps, -1).clone()
        env._ball_obs_delay = torch.randint(
            min_delay_steps, max_delay_steps + 1, (B,), device=env.device
        )

    # --- shift buffer, insert current obs at front ---
    env._ball_obs_buf = torch.roll(env._ball_obs_buf, 1, dims=1)
    env._ball_obs_buf[:, 0, :] = current_obs

    # --- gather per-env delayed entry ---
    idx = env._ball_obs_delay.clamp(0, max_delay_steps - 1)  # [B]
    delayed = env._ball_obs_buf[torch.arange(B, device=env.device), idx]  # [B, 2]
    return delayed


def kick_shot_angle_obs(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Commanded kick direction as (cos, sin) in robot frame. Shape [N, 2].

    Using cos/sin avoids the ±π discontinuity of a raw angle representation.
    Frozen to (0, 0) once the kick is in flight so the policy stops steering.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    q = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    delta = env._kick_world_shot_angle - robot_yaw
    cos_sin = torch.stack([torch.cos(delta), torch.sin(delta)], dim=-1)
    cos_sin[env._kick_timer > 0] = 0.0
    return cos_sin


def kick_target_speed_obs(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Commanded ball speed after kick [m/s]. Shape [N, 1]."""
    if not hasattr(env, "_kick_target_speed"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env._kick_target_speed.unsqueeze(-1)
