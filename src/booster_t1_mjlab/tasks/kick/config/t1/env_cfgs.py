"""Booster T1 kick environment configurations."""

import math
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from booster_t1_mjlab.tasks.kick import mdp as kick_mdp
from booster_t1_mjlab.robots import get_t1_headless_robot_cfg, T1_ACTION_SCALE_HEADLESS
from booster_t1_mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg

BALL_XML: Path = Path(__file__).parents[4] / "robots" / "boostert1" / "xmls" / "ball.xml"
assert BALL_XML.exists(), f"ball.xml not found at {BALL_XML}"

BALL_NAME = "ball"

_ALL_JOINTS_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_FEET_ASSET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def _make_ball_entity_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(BALL_XML)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(1.0, 0.0, 0.11),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def booster_t1_kick_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """T1 ball-approach task: robot spawns facing the ball and walks to it.

  The only ball-specific observation is the 2-D offset [x_forward, y_left]
  in the robot body frame — no height component. The only ball-specific reward
  is approach_ball (exponential in XY distance).
  """
  cfg = booster_t1_flat_env_cfg(play=play)

  # -------------------------------------------------------------------------
  # Scene: add the ball entity
  # -------------------------------------------------------------------------
  cfg.scene.entities[BALL_NAME] = _make_ball_entity_cfg()

  # -------------------------------------------------------------------------
  # Commands: remove the velocity twist command
  # -------------------------------------------------------------------------
  cfg.commands.clear()

  # -------------------------------------------------------------------------
  # Curriculum: remove command-dependent curriculum
  # -------------------------------------------------------------------------
  cfg.curriculum.clear()

  # -------------------------------------------------------------------------
  # Observations: replace "command" with ball XY in robot body frame
  # -------------------------------------------------------------------------
  ball_pos_noisy = ObservationTermCfg(
    func=kick_mdp.ball_pos_xy_robot_frame,
    params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_ASSET_CFG},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  ball_pos_clean = ObservationTermCfg(
    func=kick_mdp.ball_pos_xy_robot_frame,
    params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_ASSET_CFG},
  )

  del cfg.observations["actor"].terms["command"]
  cfg.observations["actor"].terms["ball_pos"] = ball_pos_noisy

  del cfg.observations["critic"].terms["command"]
  cfg.observations["critic"].terms["ball_pos"] = ball_pos_clean

  # -------------------------------------------------------------------------
  # Rewards
  # -------------------------------------------------------------------------
  del cfg.rewards["track_linear_velocity"]
  del cfg.rewards["track_angular_velocity"]

  cfg.rewards["action_rate_l2"].weight = -0.5

  del cfg.rewards["foot_slip"]
  del cfg.rewards["foot_swing_height"]

  del cfg.rewards["pose"]
  cfg.rewards["pose"] = RewardTermCfg(
    func=kick_mdp.posture,
    weight=1.0,
    params={"std": 0.25, "asset_cfg": _ALL_JOINTS_CFG},
  )

  cfg.rewards["air_time"].params.pop("command_name", None)
  cfg.rewards["air_time"].params.pop("command_threshold", None)
  cfg.rewards["foot_clearance"].params.pop("command_name", None)
  cfg.rewards["foot_clearance"].params.pop("command_threshold", None)
  cfg.rewards["soft_landing"].params.pop("command_name", None)
  cfg.rewards["soft_landing"].params.pop("command_threshold", None)

  cfg.rewards["approach_ball"] = RewardTermCfg(
    func=kick_mdp.approach_ball,
    weight=2.0,
    params={"ball_name": BALL_NAME, "max_speed": 1.5},
  )
  cfg.rewards["face_ball"] = RewardTermCfg(
    func=kick_mdp.face_ball,
    weight=1.0,
    params={"ball_name": BALL_NAME},
  )

  # -------------------------------------------------------------------------
  # Events
  # -------------------------------------------------------------------------
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.15, 0.15)

  # -------------------------------------------------------------------------
  # Play-mode overrides
  # -------------------------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def booster_t1_kick_headless_flat_env_cfg(play: bool = False):
    """Kick task with head joints fixed — 21-DOF policy (no head in obs/action)."""
    cfg = booster_t1_kick_flat_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_t1_headless_robot_cfg()
    cfg.actions["joint_pos"].scale = T1_ACTION_SCALE_HEADLESS
    return cfg


# ---------------------------------------------------------------------------
# V2: directed kick with shot-angle + target-speed commands
# ---------------------------------------------------------------------------

def booster_t1_kick_v2_headless_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Ball-kick task — 21-DOF headless T1, with directional kick command.

    Actor obs (74-dim):
        base_ang_vel (3) | projected_gravity (3) | joint_pos_rel (21) |
        joint_vel (21)   | actions (21)          | ball_pos_xy (2)    |
        kick_shot_angle cos/sin (2) | kick_target_speed (1)

    Training loop:
        - Episode terminates only on fell_over or time_out (15 s).
        - Every kick (ball_speed > 0.8 m/s) starts a 200 ms window:
            * 90 %: ball reset in front of robot.
            * 10 %: ball stays where it rolled (robot must chase and re-kick).
        - kick_shot_angle sampled uniformly from ±45° at each episode/kick.
        - target_speed sampled from (1.0, 8.0) m/s.
    """
    cfg = booster_t1_kick_headless_flat_env_cfg(play=play)

    # --- episode length ---
    cfg.episode_length_s = 10.0

    # --- remove unused rewards ---
    for key in ["track_linear_velocity", "track_angular_velocity",
                "foot_slip", "foot_swing_height", "air_time",
                "foot_clearance", "soft_landing", "face_ball", "approach_ball"]:
        cfg.rewards.pop(key, None)

    # AMP discriminator already provides style signal for all joints (including arms),
    # so posture reward is redundant and fights the arm motion in reference motions.
    cfg.rewards.pop("pose", None)
    cfg.rewards["action_rate_l2"].weight = -0.3

    # --- reference kick position: walk to approach_dist behind ball on kick axis ---
    # Phase 1: target = ball - kick_dir * 0.25m (robot arcs naturally around ball)
    # Phase 2: once within 0.15m of that point, target shifts to ball itself → drives contact
    cfg.rewards["approach_kick_position"] = RewardTermCfg(
        func=kick_mdp.approach_kick_position,
        weight=5.0,
        params={"ball_name": BALL_NAME, "approach_dist": 0.35, "contact_dist": 0.15, "max_speed": 1.5},
    )

    # --- face shot direction: face kick direction once behind ball ---
    cfg.rewards["face_shot_direction"] = RewardTermCfg(
        func=kick_mdp.face_shot_direction,
        weight=1.5,
        params={"ball_name": BALL_NAME, "max_dist": 0.8},
    )

    # --- face ball during approach: keep ball in FOV while arcing to kick position ---
    cfg.rewards["face_ball_during_approach"] = RewardTermCfg(
        func=kick_mdp.face_ball_during_approach,
        weight=1.0,
        params={"ball_name": BALL_NAME},
    )

    # --- commented out foot rewards ---
    # cfg.rewards["foot_at_kick_position"] = RewardTermCfg(...)
    # cfg.rewards["foot_swing_toward_ball"] = RewardTermCfg(...)

    # --- impulse kick reward: delta_ball_vel · kick_dir, fires at moment of hard contact ---
    # Per-timestep ball speed rewards sustained pushing over kicking (push 100 steps > kick 1 step).
    # Impulse reward fires only when ball velocity changes suddenly — zero during sustained contact.
    cfg.rewards["kick_impulse"] = RewardTermCfg(
        func=kick_mdp.kick_impulse,
        weight=30.0,
        params={"ball_name": BALL_NAME},
    )
    # cfg.rewards["kick_symmetry"] — removed, caused robot to run backward in kick direction

    # --- commanded speed matching: gaussian centered on _kick_target_speed ---
    # kick_impulse teaches HOW to generate hard kicks; kick_speed teaches WHAT power to use.
    cfg.rewards["kick_speed"] = RewardTermCfg(
        func=kick_mdp.kick_speed,
        weight=20.0,
        params={"speed_threshold": 1.0, "sigma": 3.0},
    )

    # --- Philip's direction reward: snapshot cos_sim(ball_vel, kick_dir) → [-1, +1] ---
    # Missing piece: kick_speed only rewards speed magnitude, not ball direction.
    # This fires from the same snapshot window as kick_speed so dribbling earns nothing.
    # negative_scale=0.1 means wrong direction is lightly penalised (not harshly).
    cfg.rewards["kick_direction"] = RewardTermCfg(
        func=kick_mdp.kick_direction,
        weight=8.0,
        params={"speed_threshold": 1.0, "sigma": 1.0, "negative_scale": 0.1},
    )

    # --- commented out: per-timestep speed rewards favour pushing over kicking ---
    # cfg.rewards["ball_movement"] = RewardTermCfg(weight=60.0, ...)
    # cfg.rewards["kick_direction"] = RewardTermCfg(weight=10.0, ...)
    # cfg.rewards["kick_velocity"] = RewardTermCfg(weight=20.0, ...)

    # --- observations ---
    _FEET_OBS = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
    # Actor: delayed ball pos to match real camera latency
    ball_pos_noisy = ObservationTermCfg(
        func=kick_mdp.ball_pos_xy_robot_frame_delayed,
        params={
            "ball_name": BALL_NAME,
            "feet_asset_cfg": _FEET_OBS,
            "min_delay_steps": 1,
            "max_delay_steps": 4,
        },
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    # Critic: clean ground-truth ball pos (privileged info)
    ball_pos_clean = ObservationTermCfg(
        func=kick_mdp.ball_pos_xy_robot_frame,
        params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_OBS},
    )
    kick_angle_term_noisy = ObservationTermCfg(
        func=kick_mdp.kick_shot_angle_obs, params={},
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    kick_angle_term_clean = ObservationTermCfg(func=kick_mdp.kick_shot_angle_obs, params={})
    kick_speed_term = ObservationTermCfg(func=kick_mdp.kick_target_speed_obs, params={})

    cfg.observations["actor"].terms["ball_pos"] = ball_pos_noisy
    cfg.observations["actor"].terms["kick_shot_angle"] = kick_angle_term_noisy
    cfg.observations["actor"].terms["kick_target_speed"] = kick_speed_term

    cfg.observations["critic"].terms["ball_pos"] = ball_pos_clean
    cfg.observations["critic"].terms["kick_shot_angle"] = kick_angle_term_clean
    cfg.observations["critic"].terms["kick_target_speed"] = kick_speed_term

    # --- wire kick state reset so each episode gets a fresh command ---
    cfg.events["reset_kick_state"] = EventTermCfg(
        func=kick_mdp.reset_kick_state,
        mode="reset",
        params={
            "shot_angle_offset_range": (-math.pi, math.pi),
            "target_speed_range": (1.0, 15.0),
        },
    )

    # --- replace reset_base with combined robot+ball reset ---
    cfg.events["reset_base"] = EventTermCfg(
        func=kick_mdp.reset_robot_and_ball,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "ball_name": BALL_NAME,
            "ball_distance_range": (0.5, 1.5),
            "ball_radius": 0.11,
            "ball_vel_max": 0.1,
        },
    )
    cfg.events.pop("reset_ball", None)

    cfg.events["kick_cycle_step"] = EventTermCfg(
        func=kick_mdp.kick_cycle_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ball_name": BALL_NAME,
            "speed_threshold": 1.0,
            "reset_delay_steps": 20,
            "ball_reset_prob": 0.9,
            "distance_range": (0.5, 1.5),
            "ball_radius": 0.11,
            "ball_vel_max": 0.1,
            "shot_angle_offset_range": (-math.pi, math.pi),
            "target_speed_range": (1.0, 15.0),
            "angle_resample_prob": 0.003,
            "min_episode_steps": 50,
        },
    )

    # --- kick direction visualization (orange arrow) ---
    cfg.commands["kick_dir_viz"] = kick_mdp.KickDirectionCommandCfg()

    # --- terminations: only fell_over + time_out ---
    cfg.terminations.pop("ball_kicked", None)

    if play:
        # ---- Edit these values before running play_beyondamp.py ----
        PLAY_KICK_ANGLE_DEG: float = 45.0   # world-frame degrees (0=+x, 90=+y)
        PLAY_KICK_TARGET_SPEED: float = 4.0  # m/s (1.0 – 15.0)
        # ------------------------------------------------------------
        cfg.events["reset_kick_state"] = EventTermCfg(
            func=kick_mdp.set_fixed_kick_state,
            mode="reset",
            params={
                "world_angle_rad": math.radians(PLAY_KICK_ANGLE_DEG),
                "target_speed": PLAY_KICK_TARGET_SPEED,
            },
        )

        cfg.episode_length_s = 5.0
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # Keep kick_cycle_step for kick detection (updates _kick_ball_vel_at_kick so
        # kick_speed reward is visible), but disable ball resets and angle resampling.
        cfg.events["kick_cycle_step"] = EventTermCfg(
            func=kick_mdp.kick_cycle_step,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "ball_name": BALL_NAME,
                "speed_threshold": 1.5,
                "reset_delay_steps": 5,
                "ball_reset_prob": 0.0,
                "distance_range": (0.5, 1.5),
                "ball_radius": 0.11,
                "ball_vel_max": 0.0,
                "shot_angle_offset_range": (0.0, 0.0),
                "target_speed_range": (PLAY_KICK_TARGET_SPEED, PLAY_KICK_TARGET_SPEED),
                "angle_resample_prob": 0.0,
                "min_episode_steps": 0,
            },
        )
        cfg.events["reset_play_kick_timer"] = EventTermCfg(
            func=kick_mdp.reset_play_kick_timer,
            mode="reset",
            params={},
        )
        cfg.terminations["after_kick"] = TerminationTermCfg(
            func=kick_mdp.after_kick,
            time_out=False,
            params={"ball_name": BALL_NAME, "delay_steps": 50, "speed_threshold": 0.8},
        )

    return cfg
