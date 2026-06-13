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
    """Ball-kick task — 21-DOF headless T1, no directional command.

    Obs (69-dim):
        base_ang_vel (3) | projected_gravity (3) | joint_pos_rel (21) |
        joint_vel (21)   | actions (21)          | ball_pos_xy (2)

    Training loop:
        - Episode terminates only on fell_over or time_out (15 s).
        - Every kick (ball_speed > 1.5 m/s) starts a 200 ms window:
            * 90 %: ball reset in front of robot.
            * 10 %: ball stays where it rolled (robot must chase and re-kick).
    """
    cfg = booster_t1_kick_headless_flat_env_cfg(play=play)

    # --- episode length ---
    cfg.episode_length_s = 15.0

    # --- remove unused rewards ---
    for key in ["track_linear_velocity", "track_angular_velocity",
                "foot_slip", "foot_swing_height", "air_time",
                "foot_clearance", "soft_landing", "face_ball", "approach_ball"]:
        cfg.rewards.pop(key, None)

    # --- pose / regularisation ---
    _ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))
    cfg.rewards["pose"] = RewardTermCfg(
        func=kick_mdp.posture,
        weight=1.0,
        params={"std": 0.25, "asset_cfg": _ALL_JOINTS},
    )
    cfg.rewards["action_rate_l2"].weight = -0.3

    # --- approach: foot midpoint within 0.3 m of ball ---
    _FEET = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
    cfg.rewards["approach_ball"] = RewardTermCfg(
        func=kick_mdp.approach_ball,
        weight=2.0,
        params={
            "ball_name": BALL_NAME,
            "max_speed": 1.5,
        },
    )

    # --- ball movement: reward ball speed above dribble threshold ---
    cfg.rewards["ball_movement"] = RewardTermCfg(
        func=kick_mdp.ball_movement,
        weight=8.0,
        params={"ball_name": BALL_NAME, "min_speed": 0.5, "max_speed": 5.0},
    )

    # --- kick speed: maximise ball speed at contact ---
    cfg.rewards["kick_speed"] = RewardTermCfg(
        func=kick_mdp.kick_speed,
        weight=8.0,
        params={"speed_threshold": 1.5, "max_speed": 10.0},
    )

    # --- observations: ball pos only, no command terms ---
    _FEET_OBS = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
    ball_pos_noisy = ObservationTermCfg(
        func=kick_mdp.ball_pos_xy_robot_frame,
        params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_OBS},
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    ball_pos_clean = ObservationTermCfg(
        func=kick_mdp.ball_pos_xy_robot_frame,
        params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_OBS},
    )
    cfg.observations["actor"].terms["ball_pos"] = ball_pos_noisy
    cfg.observations["critic"].terms["ball_pos"] = ball_pos_clean

    # --- replace reset_base with combined robot+ball reset (no teleport) ---
    cfg.events["reset_base"] = EventTermCfg(
        func=kick_mdp.reset_robot_and_ball,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "ball_name": BALL_NAME,
            "ball_distance_range": (0.5, 1.1),
            "ball_y_range": (-0.3, 0.3),
            "ball_radius": 0.11,
            "ball_vel_max": 1.0,
        },
    )
    cfg.events.pop("reset_ball", None)

    cfg.events["kick_cycle_step"] = EventTermCfg(
        func=kick_mdp.kick_cycle_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ball_name": BALL_NAME,
            "speed_threshold": 1.5,
            "reset_delay_steps": 10,
            "ball_reset_prob": 0.9,
            "distance_range": (0.5, 1.1),
            "y_range": (-0.3, 0.3),
            "ball_radius": 0.11,
            "ball_vel_max": 1.0,
        },
    )

    # --- terminations: only fell_over + time_out ---
    cfg.terminations.pop("ball_kicked", None)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events.pop("kick_cycle_step", None)  # no mid-episode ball respawn
        # Clear the kick timer on each episode reset
        cfg.events["reset_play_kick_timer"] = EventTermCfg(
            func=kick_mdp.reset_play_kick_timer,
            mode="reset",
            params={},
        )
        # Full episode reset 2s after kick — clean first-kick demo loop
        cfg.terminations["after_kick"] = TerminationTermCfg(
            func=kick_mdp.after_kick,
            time_out=False,
            params={"ball_name": BALL_NAME, "delay_steps": 50, "speed_threshold": 1.5},
        )

    return cfg
