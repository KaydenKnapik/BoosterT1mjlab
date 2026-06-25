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
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward robot for moving toward the ball."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    to_ball = ball_xy - robot_xy
    dist = torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
    env.extras["log"]["Metrics/ball_distance_mean"] = dist.squeeze(-1).mean()

    to_ball_hat = to_ball / dist
    vel_xy = robot.data.root_link_lin_vel_w[:, :2]
    vel_toward_ball = torch.sum(vel_xy * to_ball_hat, dim=-1)
    return (vel_toward_ball / max_speed).clamp(min=0.0, max=1.0)


def approach_kick_position(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    approach_dist: float = 0.35,
    contact_dist: float = 0.15,
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Route robot around ball to kick position via arc waypoint, velocity-based.

    Back to velocity-based reward (what originally produced good kicks) combined
    with the stable arc-routing target:
      - Arc waypoint uses a FIXED lateral offset (sign-based, not proportional to
        the robot's current perpendicular position) — unlike the old proportional
        shift, this target doesn't continuously chase the robot, so it can't be
        farmed by oscillating side to side.
      - Far from base: target ≈ arc waypoint (routes robot around ball).
      - Close to base: target smoothly blends to base_target (directly behind ball).

    Arc side fix:
      - When robot is on kick axis (perp ≈ 0) AND wrong side (180° kick case),
        old code gave lateral push of 0 (0 × 0.25 = 0), routing straight through ball.
      - Now defaults to left arc (+1) in that singularity case.

    Phase 1 -> Phase 2 is a SMOOTH velocity-reward blend (same blend weight used
    for the target position) instead of a hard cutoff. A hard cutoff created a
    reward cliff: p1 (then position-based) gave ~0.6 passive reward just outside
    the trigger, while p2 (velocity-based) started at 0 just inside it if the robot
    wasn't already moving — so the policy learned to hover just outside forever
    rather than cross into the kick-through phase. Blending removes that cliff.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)
    kick_perp = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)

    rel = robot_xy - ball_xy
    along = torch.sum(rel * (-kick_dir), dim=-1)  # + = behind ball (correct side)
    perp  = torch.sum(rel * kick_perp,  dim=-1)   # + = left of kick axis

    env.extras["log"]["Metrics/ball_distance_mean"] = torch.norm(rel, dim=-1).mean()

    base_target = ball_xy - kick_dir * approach_dist

    # Arc side: which side of kick axis to route around.
    # Fix 180° singularity: when on axis (perp ≈ 0) default to left, so robot
    # always gets a non-zero lateral push instead of driving straight through ball.
    wrong_side = (along < 0.0).float()
    arc_side = torch.where(perp.abs() < 0.1, torch.ones_like(perp), perp.sign())

    # Arc distance from kick axis.
    # Correct side: 0.3m nudge prevents robot cutting straight at ball from the side.
    # Wrong side: 0.65m forces robot to swing wide enough to clear the ball.
    arc_dist = 0.3 + wrong_side * 0.35
    arc_target = base_target + kick_perp * (arc_side * arc_dist).unsqueeze(-1)

    # Linear blend (NOT sigmoid): far → arc waypoint, close → base_target.
    # Anchored so blend == 0 exactly at the phase-2 trigger distance below.
    # A sigmoid has a nonzero floor everywhere, which creates a stable equilibrium
    # just outside the phase-2 trigger zone (proven for wrong-side arcs: settles at
    # ~0.29m from base_target, trapping the robot in phase-1 forever since phase-2
    # never turns on). Clamped linear guarantees no fixed point can exist short of
    # the trigger distance.
    # BLEND_RANGE narrowed from 0.5 -> 0.2: with the old 0.5m range, blend started
    # mixing in p2 (drive-at-ball) from as far as 0.75m out, so the robot could earn
    # partial "charge the ball" reward well before properly settling into the
    # correct behind-ball stance -- robot was observed going for the ball instead
    # of the kick pose. Now pure phase-1 (must reach the correct position) holds
    # until 0.45m out, with the transition compressed into the final 0.2m.
    PHASE2_TRIGGER_DIST = 0.25  # distance to base_target where blend reaches 0 (pure phase 2)
    BLEND_RANGE = 0.2
    dist_to_kick_pos = torch.norm(robot_xy - base_target, dim=-1)
    blend = ((dist_to_kick_pos - PHASE2_TRIGGER_DIST) / BLEND_RANGE).clamp(0.0, 1.0).unsqueeze(-1)
    target_p1 = blend * arc_target + (1.0 - blend) * base_target

    vel_xy = robot.data.root_link_lin_vel_w[:, :2]

    # Phase 1: velocity toward arc-blended target (routes around ball).
    to_p1 = target_p1 - robot_xy
    to_p1_hat = to_p1 / torch.norm(to_p1, dim=-1, keepdim=True).clamp(min=1e-6)
    p1_reward = (torch.sum(vel_xy * to_p1_hat, dim=-1) / max_speed).clamp(0.0, 1.0)

    # Phase 2: velocity toward kick-through point (35cm past ball) to drive contact.
    kick_target = ball_xy + kick_dir * approach_dist
    to_p2 = kick_target - robot_xy
    to_p2_hat = to_p2 / torch.norm(to_p2, dim=-1, keepdim=True).clamp(min=1e-6)
    p2_reward = (torch.sum(vel_xy * to_p2_hat, dim=-1) / max_speed).clamp(0.0, 1.0)

    blend_1d = blend.squeeze(-1)
    velocity_reward = blend_1d * p1_reward + (1.0 - blend_1d) * p2_reward

    # Small always-on position pull toward base_target (NOT target_p1/arc_target).
    # Pure velocity reward is exactly 0 whenever the robot is stationary, regardless
    # of position — no gradient to resume walking the last stretch. Observed: robot
    # closes most of the distance then freezes completely (e.g. stops at 0.85m when
    # base_target is 0.35m away), since standing still costs nothing once moving stops
    # paying off. base_target depends ONLY on ball position + kick_dir, never on the
    # robot's own position, so this can't be chased/oscillated like target_p1 could.
    # Kept deliberately small (max 0.2) so it can't recreate the old "free reward for
    # camping near the target" exploit — it's a nudge, not a replacement for the
    # velocity-driven approach.
    proximity_pull = torch.exp(-dist_to_kick_pos / 0.6) * 0.2

    # Zero out entirely during the post-kick snapshot window (_kick_timer > 0).
    # This reward never turned off after a kick, so "keep moving toward the ball"
    # was always the dominant incentive -- there was no reward-level reason to ever
    # stop pursuing. Gating it off here means standing still and stable (favored by
    # upright/action_rate_l2) becomes the best available option right after a kick,
    # so "relax, you did the job" emerges as a learned behavior instead of being
    # forced by a termination (which doesn't teach anything, just cuts the episode).
    not_in_kick_window = (env._kick_timer == 0).float()
    return (velocity_reward + proximity_pull) * not_in_kick_window


def ball_proximity_behind(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    optimal_dist: float = 0.15,
    sigma: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward peaking at optimal kick stance distance behind ball.

    Unlike a linear sink toward ball center, this peaks at optimal_dist so the
    robot is rewarded for the pre-kick stance (not for walking on top of ball).
    Still provides nonzero gradient at zero velocity.

    Reward = exp(-(dist - optimal_dist)^2 / sigma^2) when behind ball.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    rel = robot_xy - ball_xy
    along = torch.sum(rel * (-kick_dir), dim=-1)  # + = behind ball
    behind_ball = (along > 0.0).float()

    dist_to_ball = torch.norm(rel, dim=-1)
    stance = torch.exp(-((dist_to_ball - optimal_dist) ** 2) / (sigma ** 2))

    return behind_ball * stance


def ball_avoidance(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    sigma: float = 0.15,
    feet_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot approaching ball when robot is NOT behind it on kick axis.

    Prevents stepping over/through ball while walking around it. Only fires when
    robot is on the wrong side — allows foot near ball once in kick position.
    Use with negative weight in config.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene[feet_cfg.name]
    body: Entity = env.scene[asset_cfg.name]

    ball_pos = ball.data.root_link_pos_w[:, :3]
    foot_pos = robot.data.body_link_pos_w[:, feet_cfg.body_ids, :3]

    # Closest foot to ball
    dist_foot = torch.norm(ball_pos.unsqueeze(1) - foot_pos, dim=-1).min(dim=-1).values

    # Gaussian proximity: high when foot is close to ball
    proximity = torch.exp(-dist_foot ** 2 / sigma ** 2)

    # Gate: only penalise when NOT behind the ball on kick axis
    robot_xy = body.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]
    to_ball = ball_xy - robot_xy
    to_ball_hat = to_ball / torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)
    cos_sim = torch.sum(to_ball_hat * kick_dir, dim=-1)

    # cos_sim > 0.7 means robot is behind ball (allow foot near ball → kick)
    wrong_side = (cos_sim < 0.7).float()
    return proximity * wrong_side


def ball_movement(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    min_speed: float = 0.5,
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Reward ball movement above a minimum threshold.

    min_speed filters out slow dribbling — robot must hit the ball with
    real force to earn anything. Scales linearly from 0 at min_speed,
    uncapped above max_speed so harder kicks always earn more reward.
    """
    ball: Entity = env.scene[ball_name]
    speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)
    return ((speed - min_speed) / (max_speed - min_speed)).clamp(min=0.0)


def kick_approach_angle(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    max_dist: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward robot for being behind the ball relative to the kick direction.

    Returns cosine similarity between kick_dir and (ball - robot).
    +1 when robot is directly behind the ball, -1 when it's in front.
    Only active within max_dist metres so it doesn't fight long-range approach.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    to_ball = ball_xy - robot_xy
    dist = torch.norm(to_ball, dim=-1)
    to_ball_hat = to_ball / dist.unsqueeze(-1).clamp(min=1e-6)

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    cos_sim = torch.sum(to_ball_hat * kick_dir, dim=-1)
    return cos_sim * (dist < max_dist).float()


def face_ball_during_approach(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    max_dist: float = 0.8,
    sigma: float = 1.4,
    negative_scale: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward for robot forward axis pointing toward the literal ball.

    Active whenever farther than max_dist from the ball (travel/navigation phase).
    Once within max_dist, face_shot_direction takes over (if also behind ball) to
    orient for the kick.

    Gated by DISTANCE, not by kick-axis side (old behavior). The old gate was
    "active only when NOT behind ball on kick axis" — but "behind ball" is a
    property of robot position vs. the KICK DIRECTION, totally independent of
    which way the robot is actually facing. If the kick direction geometry
    happened to put the robot already on the "correct side" purely by spawn
    coincidence, that gate (and face_shot_direction's dist<max_dist gate) could
    BOTH be off simultaneously while still far away — nothing then forced a
    turn-around, so the robot just walked backward to the position-only target.
    Distance-based gating closes that coverage gap: this reward is always active
    while far, regardless of kick-axis side.

    sigma=1.4 (wider than the original 0.8) so it doesn't fight the arc-routing
    in approach_kick_position as hard — zero-crossing to negative reward is
    around 67° of heading deviation now (was ~46°), giving enough room to lean
    into a wide arc while still strongly penalising near-180° backward facing.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy  = ball.data.root_link_pos_w[:, :2]

    to_ball = ball_xy - robot_xy
    dist = torch.norm(to_ball, dim=-1)
    to_ball_hat = to_ball / dist.unsqueeze(-1).clamp(min=1e-6)

    forward_w = quat_apply(robot.data.root_link_quat_w, robot.data.forward_vec_b)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp(min=1e-6)

    cos_sim = torch.sum(forward_xy * to_ball_hat, dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_sim)
    raw = (torch.exp(-theta ** 2 / sigma ** 2) - 0.5) * 2.0
    reward = torch.where(raw >= 0, raw, raw * negative_scale)

    # Active when far away OR not yet behind ball on kick axis -- NOT just distance.
    # Pure distance gating (old version) left a dead zone for ball spawns under
    # max_dist (now possible down to 0.3m): if the robot also wasn't yet behind_ball,
    # face_shot_direction (needs behind_ball AND dist<max_dist) was ALSO off, so
    # nothing forced a turn-around for close, wrong-side spawns -- robot just walked
    # backward to the position-only target. This OR closes that gap while keeping
    # the far-away fix: exactly one of {this, face_shot_direction} is active at a time.
    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)
    rel = robot_xy - ball_xy
    along = torch.sum(rel * (-kick_dir), dim=-1)
    behind_ball = along > 0.0

    active = (dist >= max_dist) | (~behind_ball)
    return reward * active.float()


def face_shot_direction(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    max_dist: float = 0.8,
    sigma: float = 1.0,
    negative_scale: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward for robot forward axis aligned with commanded shot direction.

    Gated to only fire when the robot is already BEHIND the ball on the kick axis.
    Prevents early facing that conflicts with the approach walk — while navigating
    around the ball the robot should face where it's going, not the shot direction.
    Once behind the ball (correct side), it should face the shot direction to kick.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy  = ball.data.root_link_pos_w[:, :2]

    dist = torch.norm(ball_xy - robot_xy, dim=-1)

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    # Gate: robot must be on the correct side (behind ball on kick axis)
    # along > 0 means robot is in the -kick_dir half-space (behind ball)
    rel = robot_xy - ball_xy
    along = torch.sum(rel * (-kick_dir), dim=-1)
    behind_ball = (along > 0.0).float()

    forward_w = quat_apply(robot.data.root_link_quat_w, robot.data.forward_vec_b)
    forward_xy = forward_w[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp(min=1e-6)

    cos_sim = torch.sum(forward_xy * kick_dir, dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_sim)
    raw = (torch.exp(-theta ** 2 / sigma ** 2) - 0.5) * 2.0
    reward = torch.where(raw >= 0, raw, raw * negative_scale)
    return reward * (dist < max_dist).float() * behind_ball


def kick_symmetry(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    feet_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward kicking with the foot on the same side as the ball (robot frame).

    Ball to robot's left  → reward left foot (index 0) velocity in kick direction.
    Ball to robot's right → reward right foot (index 1) velocity in kick direction.
    Discourages always kicking with the same foot regardless of ball position.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[feet_cfg.name]
    ball: Entity = env.scene[ball_name]

    robot_pos = robot.data.root_link_pos_w[:, :2]
    ball_pos = ball.data.root_link_pos_w[:, :2]

    q = robot.data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos_yaw = 1.0 - 2.0 * (y ** 2 + z ** 2)
    sin_yaw = 2.0 * (w * z + x * y)

    rel = ball_pos - robot_pos
    ball_y_robot = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]  # + = ball is to robot's left

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    foot_vel = robot.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2]
    foot_vel_kick = torch.sum(foot_vel * kick_dir.unsqueeze(1), dim=-1)  # (N, 2)

    ball_left = (ball_y_robot > 0).float()
    ball_right = 1.0 - ball_left

    return (ball_left * foot_vel_kick[:, 0] + ball_right * foot_vel_kick[:, 1]).clamp(min=0.0)


def kick_impulse(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw impulse kick reward: delta_ball_vel · kick_dir at moment of hard contact.

    Unbounded — harder kicks always earn more reward. Gated on:
    - Robot BEHIND ball on kick axis (along > 0)
    - kick_timer == 0: only the FIRST contact per kick cycle earns reward.
      During the post-kick snapshot window (timer > 0) the reward is blocked,
      preventing the double-kick exploit (right-foot tap → left-foot swing).
    """
    if not hasattr(env, "_kick_world_shot_angle") or not hasattr(env, "_kick_timer"):
        return torch.zeros(env.num_envs, device=env.device)

    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene[asset_cfg.name]
    current_vel = ball.data.root_link_lin_vel_w[:, :2]

    if not hasattr(env, "_prev_ball_vel_impulse"):
        env._prev_ball_vel_impulse = current_vel.clone()

    delta = current_vel - env._prev_ball_vel_impulse
    env._prev_ball_vel_impulse = current_vel.clone()

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]
    rel = robot_xy - ball_xy
    along = torch.sum(rel * (-kick_dir), dim=-1)  # + = robot behind ball
    behind_ball = (along > 0.0).float()

    # Block during post-kick window so second-foot contacts earn nothing
    not_in_window = (env._kick_timer == 0).float()

    # Pose quality gate: reward scales with how close robot is to kick position.
    # σ=0.5m — robot must be near ball - kick_dir*0.35 to earn meaningful reward.
    base_target = ball_xy - kick_dir * 0.35
    dist_to_kick_pos = torch.norm(robot_xy - base_target, dim=-1)
    pose_quality = torch.exp(-dist_to_kick_pos ** 2 / (0.5 ** 2))

    return torch.sum(delta * kick_dir, dim=-1).clamp(min=0.0) * behind_ball * not_in_window * pose_quality


def kick_velocity(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
) -> torch.Tensor:
    """Philip-style kick reward: ball_speed * cos(angle_to_target), clamped >= 0.

    Dot product of ball velocity with commanded kick direction.
    Scales with both kick speed AND direction accuracy simultaneously.
    A hard kick in the right direction earns far more than a soft/misdirected one.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    ball: Entity = env.scene[ball_name]
    ball_vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)
    return torch.sum(ball_vel_xy * kick_dir, dim=-1).clamp(min=0.0)


def kick_direction(
    env: "ManagerBasedRlEnv",
    speed_threshold: float = 1.0,
    sigma: float = 1.0,
    negative_scale: float = 0.1,
) -> torch.Tensor:
    """Philip-style snapshot direction reward: cosine similarity of snapshot ball
    velocity vs commanded kick direction.

    Uses _kick_ball_vel_at_kick (same snapshot as kick_speed), NOT live ball velocity.
    Live velocity causes dribbling: robot earns continuous reward by slowly pushing
    the ball in the right direction. Snapshot only fires after genuine kick detection.

    Formula: (exp(-θ²/σ²) - 0.5) * 2 → range [-1, +1].
    Perfect direction = +1, 90° off = ~0, opposite = ~-1 (scaled by negative_scale).
    Large σ tolerates inaccuracy early in training — reduce as kicks become directed.
    """
    if not hasattr(env, "_kick_ball_vel_at_kick") or not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    ball_vel = env._kick_ball_vel_at_kick[:, :2]
    ball_speed = torch.norm(ball_vel, dim=-1)
    kick_active = (ball_speed > speed_threshold).float()

    kick_dir = torch.stack([
        torch.cos(env._kick_world_shot_angle),
        torch.sin(env._kick_world_shot_angle),
    ], dim=-1)

    ball_vel_norm = ball_vel / ball_speed.unsqueeze(-1).clamp(min=1e-6)
    cos_sim = torch.sum(ball_vel_norm * kick_dir, dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_sim)
    raw = (torch.exp(-theta ** 2 / sigma ** 2) - 0.5) * 2.0
    reward = torch.where(raw >= 0, raw, raw * negative_scale)
    pose_quality = getattr(env, "_kick_pose_quality", torch.ones(env.num_envs, device=env.device))
    return reward * kick_active * pose_quality


def kick_speed(
    env: "ManagerBasedRlEnv",
    speed_threshold: float = 1.0,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Gaussian reward for matching the commanded target kick speed.

    Uses the snapshot ball velocity at kick detection (_kick_ball_vel_at_kick),
    NOT live ball velocity. Fires for reset_delay_steps after each kick then goes
    silent — so dribbling (ball never exceeds speed_threshold) earns zero reward.

    Philip's approach: large sigma early in training so the gradient is broad,
    reduce to 1.0–1.5 once the robot reliably generates kicks near target range.
    """
    if not hasattr(env, "_kick_ball_vel_at_kick") or not hasattr(env, "_kick_target_speed"):
        return torch.zeros(env.num_envs, device=env.device)

    ball_speed = torch.norm(env._kick_ball_vel_at_kick, dim=-1)
    kick_active = (ball_speed > speed_threshold).float()
    error = ball_speed - env._kick_target_speed
    pose_quality = getattr(env, "_kick_pose_quality", torch.ones(env.num_envs, device=env.device))
    return torch.exp(-error ** 2 / (sigma ** 2)) * kick_active * pose_quality


def _kick_foot_geometry(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    approach_dist: float,
    feet_cfg: SceneEntityCfg,
):
    """Shared geometry for foot-at-kick-position rewards.

    Returns (along, perp, kick_dir, foot_vel, foot_facing) where:
      along/perp  — foot displacement relative to kick_pos (green circle).
                    along > 0 = foot has passed kick_pos toward ball.
      foot_facing — cos(angle between foot local-x and kick_dir), clamped [0,1].
                    1.0 when foot points exactly in kick direction.
    """
    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene[feet_cfg.name]

    ball_xy  = ball.data.root_link_pos_w[:, :2]
    foot_pos = robot.data.body_link_pos_w[:, feet_cfg.body_ids, :2]
    foot_vel = robot.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2]

    kick_dir  = torch.stack([torch.cos(env._kick_world_shot_angle),
                              torch.sin(env._kick_world_shot_angle)], dim=-1)
    kick_perp = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)

    kick_pos = ball_xy - kick_dir * approach_dist              # (N, 2) green circle
    rel      = foot_pos - kick_pos.unsqueeze(1)                # (N, 2, 2)
    along    = torch.sum(rel * kick_dir.unsqueeze(1),  dim=-1) # (N, 2)
    perp     = torch.sum(rel * kick_perp.unsqueeze(1), dim=-1) # (N, 2)

    # Foot orientation: extract local-x axis from foot quaternion [w,x,y,z]
    q = robot.data.body_link_quat_w[:, feet_cfg.body_ids, :]   # (N, 2, 4)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fwd = torch.stack([
        1.0 - 2.0 * (y**2 + z**2),   # world-x component of foot local-x
        2.0 * (x*y + w*z),            # world-y component of foot local-x
    ], dim=-1)                                                   # (N, 2, 2)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    foot_facing = torch.sum(fwd * kick_dir.unsqueeze(1), dim=-1).clamp(0.0, 1.0)  # (N, 2)

    return along, perp, kick_dir, foot_vel, foot_facing


def foot_at_kick_position(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    approach_dist: float = 0.35,
    sigma_along: float = 0.10,
    sigma_perp: float = 0.08,
    feet_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward one foot being at the kick position marker, pointing in kick direction.

    No velocity requirement — teaches the robot to load a foot at the green
    circle AND have that foot face the ball. The foot orientation factor
    (foot_facing) is 1 when the foot's local-x axis aligns with kick_dir.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    along, perp, _, _, foot_facing = _kick_foot_geometry(env, ball_name, approach_dist, feet_cfg)
    pos_reward = torch.exp(-along ** 2 / sigma_along ** 2 - perp ** 2 / sigma_perp ** 2)
    per_foot = pos_reward * foot_facing                         # (N, 2)
    # max - min: rewards one foot at kick_pos while the other is NOT there.
    # Zero when both feet are at kick_pos (standing over it), max when one foot
    # is there and the other is back in support position.
    return (per_foot.max(dim=-1).values - per_foot.min(dim=-1).values).clamp(0.0, 1.0)


def foot_swing_toward_ball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    approach_dist: float = 0.35,
    sigma_along: float = 0.15,
    sigma_perp: float = 0.08,
    min_speed: float = 0.4,
    max_speed: float = 4.0,
    feet_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward foot swinging from kick position toward ball, foot facing kick direction.

    Gaussian peaks when foot is at the kick position marker (green circle) and
    moving in kick direction. Foot orientation factor ensures foot faces ball.
    Together with foot_at_kick_position this teaches:
      1. Load a foot at the green circle, pointing toward ball
      2. Swing that foot forward through the ball
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, device=env.device)

    along, perp, kick_dir, foot_vel, foot_facing = _kick_foot_geometry(
        env, ball_name, approach_dist, feet_cfg
    )

    vel_kick   = torch.sum(foot_vel * kick_dir.unsqueeze(1), dim=-1)
    vel_reward = ((vel_kick - min_speed) / (max_speed - min_speed)).clamp(0.0, 1.0)

    pos_reward = torch.exp(-along ** 2 / sigma_along ** 2 - perp ** 2 / sigma_perp ** 2)
    return (vel_reward * pos_reward * foot_facing).max(dim=-1).values


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
