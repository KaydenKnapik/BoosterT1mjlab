"""Booster T1 flat tracking environment configurations."""

from booster_t1_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def booster_t1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "Trunk"
  motion_cmd.body_names = (
    "Trunk",
    "Hip_Roll_Left",
    "Shank_Left",
    "left_foot_link",
    "Hip_Roll_Right",
    "Shank_Right",
    "right_foot_link",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR2",
    "AR3",
    "right_hand_link",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
  )

  cfg.viewer.body_name = "Trunk"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
