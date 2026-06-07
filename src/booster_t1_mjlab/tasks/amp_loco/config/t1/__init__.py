from booster_t1_mjlab.tasks.registry import register_mjlab_task
from booster_t1_mjlab.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
    t1_amp_flat_env_cfg,
    t1_amp_rough_env_cfg,
)
from .rl_cfg import t1_amp_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-AMP-Rough-Booster-T1",
    env_cfg=t1_amp_rough_env_cfg(),
    play_env_cfg=t1_amp_rough_env_cfg(play=True),
    rl_cfg=t1_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-AMP-Flat-Booster-T1",
    env_cfg=t1_amp_flat_env_cfg(),
    play_env_cfg=t1_amp_flat_env_cfg(play=True),
    rl_cfg=t1_amp_ppo_runner_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
