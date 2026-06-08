"""MDP components for the kick task."""

from booster_t1_mjlab.tasks.kick.mdp.events import (
    kick_cycle_step,
    reset_kick_state,
    reset_play_kick_timer,
    reset_robot_and_ball,
)
from booster_t1_mjlab.tasks.kick.mdp.terminations import (
    after_kick,
    ball_kicked,
)
from booster_t1_mjlab.tasks.kick.mdp.observations import (
    ball_pos_xy_robot_frame,
    kick_shot_angle_obs,
    kick_target_speed_obs,
)
from booster_t1_mjlab.tasks.kick.mdp.rewards import (
    approach_ball,
    ball_movement,
    face_ball,
    face_shot_direction,
    kick_direction,
    kick_speed,
    posture,
)
