"""Convert a G1 motion CSV to a Booster T1 NPZ via joint retargeting.

Reads a Unitree G1 29-DOF retargeted CSV, remaps joints to the T1 skeleton,
runs forward kinematics through the T1 MuJoCo model, and saves an NPZ ready
for use with Mjlab-Tracking-Flat-Booster-T1.
"""

from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

from booster_t1_mjlab.tasks.tracking.config.t1.env_cfgs import (
  booster_t1_flat_tracking_env_cfg,
)

# G1 DOF column order (columns 7-35 of a G1 CSV).
_G1_JOINTS = [
  "left_hip_pitch_joint",       # 0
  "left_hip_roll_joint",        # 1
  "left_hip_yaw_joint",         # 2
  "left_knee_joint",            # 3
  "left_ankle_pitch_joint",     # 4
  "left_ankle_roll_joint",      # 5
  "right_hip_pitch_joint",      # 6
  "right_hip_roll_joint",       # 7
  "right_hip_yaw_joint",        # 8
  "right_knee_joint",           # 9
  "right_ankle_pitch_joint",    # 10
  "right_ankle_roll_joint",     # 11
  "waist_yaw_joint",            # 12
  "waist_roll_joint",           # 13  (no T1 equivalent)
  "waist_pitch_joint",          # 14  (no T1 equivalent)
  "left_shoulder_pitch_joint",  # 15
  "left_shoulder_roll_joint",   # 16
  "left_shoulder_yaw_joint",    # 17  (no T1 equivalent)
  "left_elbow_joint",           # 18
  "left_wrist_roll_joint",      # 19  (no T1 equivalent)
  "left_wrist_pitch_joint",     # 20  (no T1 equivalent)
  "left_wrist_yaw_joint",       # 21
  "right_shoulder_pitch_joint", # 22
  "right_shoulder_roll_joint",  # 23
  "right_shoulder_yaw_joint",   # 24  (no T1 equivalent)
  "right_elbow_joint",          # 25
  "right_wrist_roll_joint",     # 26  (no T1 equivalent)
  "right_wrist_pitch_joint",    # 27  (no T1 equivalent)
  "right_wrist_yaw_joint",      # 28
]

# T1 DOF order (depth-first traversal of T1 XML).
_T1_JOINTS = [
  "AAHead_yaw",           # 0  — no G1 equivalent, stays at default
  "Head_pitch",           # 1  — no G1 equivalent, stays at default
  "Left_Shoulder_Pitch",  # 2  ← G1[15]
  "Left_Shoulder_Roll",   # 3  ← G1[16]
  "Left_Elbow_Pitch",     # 4  ← G1[18]
  "Left_Elbow_Yaw",       # 5  ← G1[21]  wrist_yaw ≈ elbow_yaw
  "Right_Shoulder_Pitch", # 6  ← G1[22]
  "Right_Shoulder_Roll",  # 7  ← G1[23]
  "Right_Elbow_Pitch",    # 8  ← G1[25]
  "Right_Elbow_Yaw",      # 9  ← G1[28]  wrist_yaw ≈ elbow_yaw
  "Waist",                # 10 ← G1[12]  waist_yaw (only T1 waist DOF)
  "Left_Hip_Pitch",       # 11 ← G1[0]
  "Left_Hip_Roll",        # 12 ← G1[1]
  "Left_Hip_Yaw",         # 13 ← G1[2]
  "Left_Knee_Pitch",      # 14 ← G1[3]
  "Left_Ankle_Pitch",     # 15 ← G1[4]
  "Left_Ankle_Roll",      # 16 ← G1[5]
  "Right_Hip_Pitch",      # 17 ← G1[6]
  "Right_Hip_Roll",       # 18 ← G1[7]
  "Right_Hip_Yaw",        # 19 ← G1[8]
  "Right_Knee_Pitch",     # 20 ← G1[9]
  "Right_Ankle_Pitch",    # 21 ← G1[10]
  "Right_Ankle_Roll",     # 22 ← G1[11]
]

# T1 joint index → G1 joint index (unmapped joints keep robot default).
_T1_TO_G1: dict[int, int] = {
  2: 15,  3: 16,  4: 18,  5: 21,   # left arm
  6: 22,  7: 23,  8: 25,  9: 28,   # right arm
  10: 12,                            # waist
  11: 0,  12: 1,  13: 2,  14: 3,  15: 4,  16: 5,   # left leg
  17: 6,  18: 7,  19: 8,  20: 9,  21: 10, 22: 11,  # right leg
}

_G1_TO_T1: dict[int, int] = {g: t for t, g in _T1_TO_G1.items()}

# G1 motion data is captured relative to G1's anatomical T-pose (all joints = 0).
# T1's natural resting pose (HOME_KEYFRAME) differs, so add these offsets so that
# when G1 is at T-pose, T1 is at its home pose instead.
_T1_JOINT_OFFSETS: dict[str, float] = {
  "Left_Shoulder_Roll":  -1.4,  # home: arms at sides, not horizontal
  "Right_Shoulder_Roll":  1.4,
  "Left_Elbow_Yaw":      -0.4,  # home: slight inward rotation
  "Right_Elbow_Yaw":      0.4,
}


class G1ToT1MotionLoader:
  """Loads a G1 CSV and remaps DOF positions to T1 joint order."""

  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / input_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load(motion_file)
    self._interpolate()
    self._compute_velocities()

  def _load(self, path: str) -> None:
    if self.line_range is None:
      raw = torch.from_numpy(np.loadtxt(path, delimiter=","))
    else:
      raw = torch.from_numpy(
        np.loadtxt(
          path,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    raw = raw.to(torch.float32).to(self.device)

    self.motion_base_poss_input = raw[:, :3]
    quat_xyzw = raw[:, 3:7]
    self.motion_base_rots_input = quat_xyzw[:, [3, 0, 1, 2]]  # → wxyz

    g1_dof = raw[:, 7:]  # (T, 29)
    n_frames = g1_dof.shape[0]
    t1_dof = torch.zeros(n_frames, len(_T1_JOINTS), device=self.device)
    for t1_idx, g1_idx in _T1_TO_G1.items():
      t1_dof[:, t1_idx] = g1_dof[:, g1_idx]
    for joint_name, offset in _T1_JOINT_OFFSETS.items():
      t1_dof[:, _T1_JOINTS.index(joint_name)] += offset
    self.motion_dof_poss_input = t1_dof

    self.input_frames = n_frames
    self.duration = (n_frames - 1) * self.input_dt

  def _interpolate(self) -> None:
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    idx0, idx1, blend = self._frame_blend(times)

    def lerp(a, b):
      return a * (1 - blend.unsqueeze(1)) + b * blend.unsqueeze(1)

    self.motion_base_poss = lerp(
      self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1]
    )
    self.motion_dof_poss = lerp(
      self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1]
    )
    slerped = torch.zeros(self.output_frames, 4, device=self.device)
    for i in range(self.output_frames):
      slerped[i] = quat_slerp(
        self.motion_base_rots_input[idx0[i]],
        self.motion_base_rots_input[idx1[i]],
        float(blend[i]),
      )
    self.motion_base_rots = slerped
    print(
      f"Motion loaded: {self.input_frames} input frames @ {self.input_fps} fps → "
      f"{self.output_frames} output frames @ {self.output_fps} fps"
    )

  def _frame_blend(self, times):
    phase = times / self.duration
    idx0 = (phase * (self.input_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - idx0
    return idx0, idx1, blend

  def _compute_velocities(self) -> None:
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]

    q_prev = self.motion_base_rots[:-2]
    q_next = self.motion_base_rots[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
    self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(self):
    i = self.current_idx
    state = (
      self.motion_base_poss[i : i + 1],
      self.motion_base_rots[i : i + 1],
      self.motion_base_lin_vels[i : i + 1],
      self.motion_base_ang_vels[i : i + 1],
      self.motion_dof_poss[i : i + 1],
      self.motion_dof_vels[i : i + 1],
    )
    self.current_idx += 1
    reset = self.current_idx >= self.output_frames
    if reset:
      self.current_idx = 0
    return state, reset


def run_sim(sim: Simulation, scene: Scene, input_file: str, input_fps: int,
            output_fps: int, output_file: str) -> None:
  motion = G1ToT1MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(_T1_JOINTS, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False
  scene.reset()

  # Auto z-correction: run FK on frame 0 to find how high the feet are,
  # then shift all base positions down so the lowest foot touches z=0.
  foot_body_idxs = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
  probe_state, _ = motion.get_next_state()
  motion.current_idx = 0  # reset so the main loop starts from frame 0
  root_probe = robot.data.default_root_state.clone()
  root_probe[:, :3] = probe_state[0]
  root_probe[:, :2] += scene.env_origins[:, :2]
  root_probe[:, 3:7] = probe_state[1]
  robot.write_root_state_to_sim(root_probe)
  jp = robot.data.default_joint_pos.clone()
  jp[:, robot_joint_indexes] = probe_state[4]
  robot.write_joint_state_to_sim(jp, robot.data.default_joint_vel.clone())
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)
  foot_z = robot.data.body_link_pos_w[0, foot_body_idxs, 2]
  z_correction = -float(foot_z.min())
  print(f"[INFO]: Auto z-correction: {z_correction:+.4f}m (foot heights: {foot_z.tolist()})")
  motion.motion_base_poss[:, 2] += z_correction
  motion.motion_base_lin_vels[:, 2] = torch.gradient(
    motion.motion_base_poss[:, 2], spacing=motion.output_dt, dim=0
  )[0]
  scene.reset()

  print(f"\nRunning T1 forward kinematics for {motion.output_frames} frames...")
  pbar = tqdm(total=motion.output_frames, unit="frame", ncols=100)

  while not file_saved:
    (
      motion_base_pos,
      motion_base_rot,
      motion_base_lin_vel,
      motion_base_ang_vel,
      motion_dof_pos,
      motion_dof_vel,
    ), reset_flag = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    pbar.update(1)

    if reset_flag:
      file_saved = True
      pbar.close()
      print("\nStacking arrays...")
      for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
      print(f"Saving to {output_file}...")
      np.savez(output_file, **log)
      print(f"[INFO]: Motion saved to {output_file}")


def main(
  input_file: str,
  output_file: str = "/tmp/t1_motion.npz",
  input_fps: float = 60.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  """Retarget a G1 motion CSV to Booster T1 and save as NPZ.

  Args:
    input_file: Path to the G1 retargeted CSV file.
    output_file: Where to save the output NPZ.
    input_fps: Frame rate of the input CSV.
    output_fps: Output frame rate (should match T1 tracking env, default 50).
    device: Torch/MuJoCo device.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA not available, falling back to CPU.")
    device = "cpu"

  env_cfg = booster_t1_flat_tracking_env_cfg()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  run_sim(
    sim=sim,
    scene=scene,
    input_file=input_file,
    input_fps=int(input_fps),
    output_fps=int(output_fps),
    output_file=output_file,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
