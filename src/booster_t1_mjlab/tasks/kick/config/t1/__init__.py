from booster_t1_mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_t1_kick_v2_headless_flat_env_cfg
from .rl_cfg import booster_t1_kick_v2_ppo_runner_cfg

# NOTE: Mjlab-AmpKick-Booster-T1-21Dof is registered as a beyondAMP task in
# tasks/kick/agents/__init__.py — do not register it here with the PPO runner
# or booster_t1_train will silently train it as plain PPO with no motion imitation.
