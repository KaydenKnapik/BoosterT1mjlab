from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def posture(
    env: "ManagerBasedRlEnv",
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward for staying near the default joint pose."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_sq = torch.square(joint_pos - default_pos)
    return torch.exp(-torch.mean(error_sq / std ** 2, dim=1))


def approach_ball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    target_x: float = 0.25,
    x_std: float = 0.15,
    y_std: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward for lining up with the ball: ball should be target_x ahead (X)
    and centered (Y≈0) in robot frame relative to foot midpoint.

    Separating X and Y prevents the robot parking with the ball beside its foot
    (0.3m Euclidean but entirely sideways) — the robot must actually face and
    approach the ball from behind.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    foot_pos_w = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :2]
    feet_mid_xy = foot_pos_w.mean(dim=1)                    # [N, 2]
    ball_xy = ball.data.root_link_pos_w[:, :2]              # [N, 2]

    # ball relative to feet midpoint in world frame → rotate to robot frame
    rel_w = torch.zeros(env.num_envs, 3, device=env.device)
    rel_w[:, :2] = ball_xy - feet_mid_xy
    rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_w)  # [N, 3]
    bx, by = rel_b[:, 0], rel_b[:, 1]

    dist = torch.norm(rel_w[:, :2], dim=-1)
    env.extras["log"]["Metrics/ball_distance_mean"] = dist.mean()

    # X: reward for ball being ~target_x ahead; Y: reward for ball being centred
    reward_x = torch.exp(-((bx - target_x) ** 2) / x_std ** 2)
    reward_y = torch.exp(-(by ** 2) / y_std ** 2)
    return reward_x * reward_y


def ball_movement(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Dense reward for any ball movement — gives gradient signal before hard kicks."""
    ball: Entity = env.scene[ball_name]
    speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)
    return (speed / max_speed).clamp(max=1.0)


def face_shot_direction(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cosine reward for robot forward axis aligned with commanded shot direction.

    +1 when the robot is perfectly aligned to kick in the commanded direction,
    -1 when facing opposite. Encourages the robot to set up correctly before
    kicking rather than just facing the ball.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    forward_w = quat_apply(robot.data.root_link_quat_w, robot.data.forward_vec_b)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp(min=1e-6)

    target = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)  # [N, 2]

    return torch.sum(forward_xy * target, dim=-1)


def kick_direction(
    env: "ManagerBasedRlEnv",
    speed_threshold: float = 1.5,
) -> torch.Tensor:
    """Cosine similarity between ball velocity at kick and commanded direction.

    Only active during the 200 ms reward window after a kick is detected
    (while ``_kick_ball_vel_at_kick`` is non-zero). Gives dense reward each
    step of the window, then resets to 0 until the next kick.
    """
    if not hasattr(env, "_kick_ball_vel_at_kick"):
        return torch.zeros(env.num_envs, device=env.device)

    ball_vel = env._kick_ball_vel_at_kick  # [N, 3], velocity at kick moment
    ball_speed = torch.norm(ball_vel, dim=-1)
    kick_active = ball_speed > speed_threshold  # mask: kick window open

    kick_dir_xy = ball_vel[:, :2] / ball_speed.unsqueeze(-1).clamp(min=1e-6)
    target = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    cos_sim = torch.sum(kick_dir_xy * target, dim=-1)
    return cos_sim * kick_active.float()


def kick_speed(
    env: "ManagerBasedRlEnv",
    speed_threshold: float = 1.5,
    max_speed: float = 10.0,
) -> torch.Tensor:
    """Reward ball speed at kick moment, gated on kick being detected.

    Normalised to [0, 1] by max_speed so reward scale stays stable regardless
    of how hard the robot kicks.
    """
    if not hasattr(env, "_kick_ball_vel_at_kick"):
        return torch.zeros(env.num_envs, device=env.device)

    ball_speed = torch.norm(env._kick_ball_vel_at_kick, dim=-1)
    kick_active = ball_speed > speed_threshold
    return (ball_speed / max_speed).clamp(max=1.0) * kick_active.float()


# ---------------------------------------------------------------------------
# Legacy (kept for old task compatibility)
# ---------------------------------------------------------------------------

def face_ball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cosine similarity between robot forward axis and direction to ball."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    to_ball = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    to_ball = to_ball / torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)

    forward_w = quat_apply(robot.data.root_link_quat_w, robot.data.forward_vec_b)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp(min=1e-6)

    return torch.sum(forward_xy * to_ball, dim=-1)
