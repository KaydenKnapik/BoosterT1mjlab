"""Register Booster T1 beyondAMP velocity tasks (mjlab backend).

Tasks use the beyondAMP runner (AMPEnvWrapper + AMPOnPolicyRunner) instead of
mjlab's stock wrapper. Train with:

    uv run python beyondAMP/scripts/factoryMjlab/train.py \\
        Mjlab-BeyondAMP-Velocity-Flat-Booster-T1 \\
        --agent.amp-data.motion-files=<path/to/motion.npz>
"""

from mjlab.tasks.registry import register_mjlab_task

from .agents.amp_ppo_cfg import t1_amp_runner_cfg
from .amp_env_cfg import t1_flat_headless_amp_env_cfg

register_mjlab_task(
    task_id="Mjlab-BeyondAMP-Velocity-Flat-Booster-T1",
    env_cfg=t1_flat_headless_amp_env_cfg(),
    play_env_cfg=t1_flat_headless_amp_env_cfg(play=True),
    rl_cfg=t1_amp_runner_cfg(),
    runner_cls=None,
)
