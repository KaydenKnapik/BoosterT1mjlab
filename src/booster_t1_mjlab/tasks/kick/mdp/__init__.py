"""MDP components for the kick task."""

from booster_t1_mjlab.tasks.kick.mdp.events import (
    kick_cycle_step,
    reset_kick_state,
    reset_play_kick_timer,
    reset_robot_and_ball,
    set_fixed_kick_state,
)
from booster_t1_mjlab.tasks.kick.mdp.terminations import (
    after_kick,
    after_n_kicks,
    ball_kicked,
)
from booster_t1_mjlab.tasks.kick.mdp.observations import (
    ball_pos_xy_robot_frame,
    ball_pos_xy_robot_frame_delayed,
    kick_shot_angle_obs,
    kick_target_speed_obs,
)
from booster_t1_mjlab.tasks.kick.mdp.kick_command import (
    KickDirectionCommand,
    KickDirectionCommandCfg,
)
from booster_t1_mjlab.tasks.kick.mdp.rewards import (
    approach_ball,
    approach_kick_position,
    ball_avoidance,
    ball_movement,
    ball_proximity_behind,
    face_ball,
    face_ball_during_approach,
    face_shot_direction,
    foot_at_kick_position,
    foot_swing_toward_ball,
    kick_approach_angle,
    kick_direction,
    kick_impulse,
    kick_speed,
    kick_symmetry,
    kick_velocity,
    posture,
)
