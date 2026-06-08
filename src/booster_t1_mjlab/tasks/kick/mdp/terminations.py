from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_kicked(
  env: ManagerBasedRlEnv,
  ball_name: str,
  speed_threshold: float = 0.5,
) -> torch.Tensor:
  """Terminate when the ball exceeds speed_threshold m/s."""
  ball: Entity = env.scene[ball_name]
  ball_speed = torch.norm(ball.data.root_link_lin_vel_w, dim=-1)  # [B]
  return ball_speed > speed_threshold


def after_kick(
  env: ManagerBasedRlEnv,
  ball_name: str,
  delay_steps: int = 100,
  speed_threshold: float = 1.5,
) -> torch.Tensor:
  """Terminate delay_steps after the first kick (ball speed > threshold).

  Designed for play mode: robot kicks, ball rolls for ~2s, episode resets.
  The _play_kick_timer attribute is cleared by reset_play_kick_timer on reset.
  """
  if not hasattr(env, "_play_kick_timer"):
    env._play_kick_timer = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  ball: Entity = env.scene[ball_name]
  ball_speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)

  just_kicked = (ball_speed > speed_threshold) & (env._play_kick_timer == 0)
  env._play_kick_timer[just_kicked] = 1
  env._play_kick_timer[env._play_kick_timer > 0] += 1

  return env._play_kick_timer > delay_steps
