"""AMPRunnerCfg for the T1 headless kick task (beyondAMP)."""

from __future__ import annotations

import os

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import (
    AMPPPOAlgorithmCfg,
    AMPRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

from ..amp_env_cfg import KICK_ANCHOR_NAME, KICK_KEY_BODY_NAMES

_KICK_MOTIONS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "motions")
)
_WALK_MOTIONS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "motions", "t1", "amp", "WalkandRun")
)

def _npz_files(directory: str, filter_prefix: list[str] | None = None) -> list[str]:
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".npz"))
    if filter_prefix:
        files = [f for f in files if any(f.startswith(p) for p in filter_prefix)]
    return [os.path.join(directory, f) for f in files]

_FAR_KICKS_DIR = os.path.join(_KICK_MOTIONS_DIR, "old_far_kicks")

_MOTION_FILES: list[str] = (
    _npz_files(_KICK_MOTIONS_DIR)
    + [f for f in _npz_files(_FAR_KICKS_DIR) if "_close" not in os.path.basename(f)]
    + _npz_files(_WALK_MOTIONS_DIR, filter_prefix=["walk_forward", "walk_sideway"])
    + _npz_files(_WALK_MOTIONS_DIR, filter_prefix=["idle_turn_270", "idle_turn_360"])
)


def t1_amp_kick_runner_cfg() -> AMPRunnerCfg:
    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=100_001,
        save_interval=100,
        experiment_name="t1_amp_kick",
        run_name="amp",
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=AMPPPOAlgorithmCfg(
            class_name="AMPPPO",
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
        ),
        amp_data=MotionDatasetCfg(
            motion_files=_MOTION_FILES,
            body_names=KICK_KEY_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name=KICK_ANCHOR_NAME,
        ),
        amp_discr_hidden_dims=[512, 256, 128],
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.7,
        amp_min_normalized_std=0.05,
    )
