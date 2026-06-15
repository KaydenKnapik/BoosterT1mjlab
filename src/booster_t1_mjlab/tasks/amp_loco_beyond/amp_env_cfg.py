"""Booster T1 velocity AMP environment configurations using beyondAMP.

Reuses :mod:`booster_t1_mjlab.tasks.velocity.config.t1` flat headless factory
and attaches a basic (joint_pos + joint_vel) AMP observation group.

The NPZ files store joint_pos relative to HOME_KEYFRAME so they match what
joint_pos_rel returns at runtime — consistent with how G1 works (G1's HOME
is all-zeros so the distinction doesn't arise there).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from booster_t1_mjlab.tasks.velocity.config.t1.env_cfgs import (
    booster_t1_flat_headless_env_cfg,
)
from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

T1_ANCHOR_NAME: str = "Trunk"

T1_KEY_BODY_NAMES: list[str] = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
    "Waist",
]


def t1_flat_headless_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = booster_t1_flat_headless_env_cfg(play=play)
    cfg.observations["amp"] = amp_obs_basic_group()
    # AMP discriminator already provides style signal for all joints including arms.
    # Posture reward fights the reference arm motion (G1-retargeted walk has elbows
    # bent ~1.2 rad but posture reward pulls toward 0), so drop it here.
    cfg.rewards.pop("pose", None)
    return cfg
