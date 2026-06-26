from booster_t1_mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  booster_t1_flat_env_cfg,
  booster_t1_flat_headless_env_cfg,
  booster_t1_rough_env_cfg,
  booster_t1_rough_headless_env_cfg,
)
from .rl_cfg import booster_t1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-T1",
  env_cfg=booster_t1_rough_env_cfg(),
  play_env_cfg=booster_t1_rough_env_cfg(play=True),
  rl_cfg=booster_t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-T1",
  env_cfg=booster_t1_flat_env_cfg(),
  play_env_cfg=booster_t1_flat_env_cfg(play=True),
  rl_cfg=booster_t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

_headless_rl_cfg = booster_t1_ppo_runner_cfg()
_headless_rl_cfg.experiment_name = "t1_velocity_flat_headless"

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-T1-Headless",
  env_cfg=booster_t1_flat_headless_env_cfg(),
  play_env_cfg=booster_t1_flat_headless_env_cfg(play=True),
  rl_cfg=_headless_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

_rough_headless_rl_cfg = booster_t1_ppo_runner_cfg()
_rough_headless_rl_cfg.experiment_name = "t1_velocity_rough_headless"

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-T1-Headless",
  env_cfg=booster_t1_rough_headless_env_cfg(),
  play_env_cfg=booster_t1_rough_headless_env_cfg(play=True),
  rl_cfg=_rough_headless_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

