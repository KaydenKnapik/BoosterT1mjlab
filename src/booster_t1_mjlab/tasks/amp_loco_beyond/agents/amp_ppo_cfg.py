"""RL config for Booster T1 headless velocity AMP task (beyondAMP/mjlab)."""

from __future__ import annotations

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import (
    AMPPPOAlgorithmCfg,
    AMPRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from beyondAMP.motion.motion_dataset import MotionDatasetCfg

from ..amp_env_cfg import T1_ANCHOR_NAME, T1_KEY_BODY_NAMES

# Default motion-clip path — override at run-time via
#   --agent.amp-data.motion-files=[path/to/motion.npz]
# Points to the retargeted WalkandRun clips produced by retarget_g1_to_t1.py.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
T1_DEFAULT_MOTION_DIR: str = _os.path.normpath(
    _os.path.join(_HERE, "..", "..", "..", "assets", "motions", "t1", "amp", "WalkandRun")
)
T1_DEFAULT_MOTION_FILES: list[str] = sorted(
    _os.path.join(T1_DEFAULT_MOTION_DIR, f)
    for f in _os.listdir(T1_DEFAULT_MOTION_DIR)
    if f.endswith(".npz")
) if _os.path.isdir(T1_DEFAULT_MOTION_DIR) else []


def t1_amp_runner_cfg() -> AMPRunnerCfg:
    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=100_001,
        save_interval=100,
        experiment_name="t1_amp_loco",
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
            motion_files=T1_DEFAULT_MOTION_FILES,
            body_names=T1_KEY_BODY_NAMES,
            amp_obs_terms=AMPObsBaiscTerms,
            anchor_name=T1_ANCHOR_NAME,
        ),
        amp_discr_hidden_dims=[512, 256, 128],
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.7,
        amp_min_normalized_std=0.05,
    )
