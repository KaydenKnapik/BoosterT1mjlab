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


def after_n_kicks(
  env: ManagerBasedRlEnv,
  n: int = 3,
) -> torch.Tensor:
  """Terminate episode after the robot has kicked n times.

  Prevents the robot from drifting far from spawn across many kick cycles.
  Each fresh episode starts with a clean robot and ball position.
  """
  if not hasattr(env, "_kick_count"):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env._kick_count >= n


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
