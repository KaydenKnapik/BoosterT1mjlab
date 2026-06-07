"""RL configuration for Booster T1 AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

# Motion data directory — AMPLoader walks this tree for all .npz files.
_MOTION_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir, os.pardir,
    "assets", "motions", "t1", "amp",
))


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config extended with AMP-specific parameters."""
    amp_reward_coef: float = 0.1
    amp_motion_files: str = ""
    amp_num_preload_transitions: int = 200000
    amp_task_reward_lerp: float = 0.75
    amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * 21)
    amp_body_names: tuple = ()
    amp_anchor_name: str = ""


def t1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create RL runner configuration for Booster T1 AMP locomotion task."""
    return RslRlAmpRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            class_name="AMPPPO",
        ),
        experiment_name="t1_amp_locomotion",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100_001,
        # AMP parameters
        amp_reward_coef=0.1,
        amp_motion_files=os.path.join(_MOTION_DATA_DIR, "WalkandRun"),
        amp_num_preload_transitions=200000,
        amp_task_reward_lerp=0.75,
        amp_discr_hidden_dims=[1024, 512, 256],
        # 21-DOF headless T1 (no head joints).
        min_normalized_std=[0.05] * 21,
        # 13 bodies tracked relative to Trunk anchor — matches env_cfgs.py body_names.
        amp_body_names=(
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
        ),
        amp_anchor_name="Trunk",
    )
