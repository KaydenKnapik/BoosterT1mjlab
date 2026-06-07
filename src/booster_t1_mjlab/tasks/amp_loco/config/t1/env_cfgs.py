"""Booster T1 AMP locomotion environment configurations (21-DOF headless)."""

from booster_t1_mjlab.robots import (
    T1_ACTION_SCALE_HEADLESS,
    get_t1_headless_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import posture as posture_reward
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from booster_t1_mjlab.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from booster_t1_mjlab.tasks.amp_loco.mdp.rewards import self_collision_cost


def t1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 headless rough terrain AMP configuration."""
    cfg = make_amp_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 128
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = 48

    cfg.scene.entities = {"robot": get_t1_headless_robot_cfg()}

    # Raycast from Trunk (T1 root body).
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "Trunk"

    # T1 has 4 foot collision geoms per foot.
    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
    )

    # Body names for AMP observations and rewards.
    # Trunk is the anchor; Waist plays the role of pelvis (below Trunk, above hips).
    body_names = (
        "Waist",
        "Hip_Pitch_Left",
        "Shank_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Shank_Right",
        "right_foot_link",
        "AL1",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR3",
        "right_hand_link",
    )
    anchor_name = "Trunk"
    root_name = "Trunk"

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = T1_ACTION_SCALE_HEADLESS

    cfg.viewer.body_name = "Trunk"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )
    cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

    cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

    cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

    cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

    cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

    cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

    cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

    # T1 home height ~0.665m; terminate if root drops below 0.3m (allows deep crouch for recovery).
    cfg.terminations["bad_base_height"].params["minimum_height"] = 0.3

    # Keep arms near HOME_KEYFRAME (shoulder_roll ±1.4, elbow_yaw ±0.4).
    # At early training when the AMP discriminator is untrained, this prevents arm
    # drift to T-pose (joints at zero). Mirrors the pose reward in the velocity task.
    cfg.rewards["arm_pose"] = RewardTermCfg(
        func=posture_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            )),
            "std": {".*": 0.5},
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

    return cfg


def t1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 headless flat terrain AMP configuration."""
    cfg = t1_amp_rough_env_cfg(play=play)

    cfg.sim.njmax = 640
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor (flat plane has no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.5, 0.5)
        twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
        twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    return cfg
