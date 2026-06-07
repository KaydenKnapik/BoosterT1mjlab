"""Convert a Booster T1 motion PKL (already T1 23-DOF format) to NPZ for Mjlab tracking.

Unlike pkl_to_npz_t1.py (which expects G1 29-DOF data), this script handles PKLs
that are already retargeted to the T1 skeleton (e.g. booster_t1_motion.pkl).

Expected PKL keys:
  fps            float
  root_pos       (T, 3)   root position, xyzw convention
  root_rot       (T, 4)   root quaternion, xyzw convention
  dof_pos        (T, 23)  T1 joint positions in T1 joint order
"""

from typing import Any
import pickle

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
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

_T1_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


class T1MotionLoader:
    def __init__(self, pkl_path: str, output_fps: int, device: torch.device | str):
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        self._load(pkl_path)
        self._interpolate()
        self._compute_velocities()

    def _load(self, pkl_path: str) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.input_fps = data["fps"]
        self.input_dt = 1.0 / self.input_fps

        root_pos = torch.from_numpy(np.array(data["root_pos"], dtype=np.float32)).to(self.device)
        # root_rot is xyzw — convert to wxyz for mjlab
        root_rot_xyzw = torch.from_numpy(np.array(data["root_rot"], dtype=np.float32)).to(self.device)
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

        dof_pos = torch.from_numpy(np.array(data["dof_pos"], dtype=np.float32)).to(self.device)
        self.input_frames = dof_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

        self.motion_base_poss_input = root_pos
        self.motion_base_rots_input = root_rot_wxyz
        self.motion_dof_poss_input = dof_pos

        print(f"Loaded: {self.input_frames} frames @ {self.input_fps:.2f} fps ({self.duration:.2f}s)")

    def _interpolate(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._frame_blend(times)

        def lerp(a, b):
            return a * (1 - blend.unsqueeze(1)) + b * blend.unsqueeze(1)

        self.motion_base_poss = lerp(self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1])
        self.motion_dof_poss = lerp(self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1])

        slerped = torch.zeros(self.output_frames, 4, device=self.device)
        for i in range(self.output_frames):
            slerped[i] = quat_slerp(
                self.motion_base_rots_input[idx0[i]],
                self.motion_base_rots_input[idx1[i]],
                float(blend[i]),
            )
        self.motion_base_rots = slerped
        print(f"Interpolated: {self.input_frames} frames @ {self.input_fps:.1f} fps → {self.output_frames} frames @ {self.output_fps} fps")

    def _frame_blend(self, times):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        q_prev = self.motion_base_rots[:-2]
        q_next = self.motion_base_rots[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        i = self.current_idx
        state = (
            self.motion_base_poss[i: i + 1],
            self.motion_base_rots[i: i + 1],
            self.motion_base_lin_vels[i: i + 1],
            self.motion_base_ang_vels[i: i + 1],
            self.motion_dof_poss[i: i + 1],
            self.motion_dof_vels[i: i + 1],
        )
        self.current_idx += 1
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, reset


def run_sim(sim: Simulation, scene: Scene, motion: T1MotionLoader, output_fps: int, output_file: str) -> None:
    robot = scene["robot"]
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
    scene.reset()

    foot_body_idxs = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]

    print("Pass 1/2: scanning all frames for global minimum foot height...")
    min_foot_z = float("inf")
    for i in range(motion.output_frames):
        root_probe = robot.data.default_root_state.clone()
        root_probe[:, :3] = motion.motion_base_poss[i: i + 1]
        root_probe[:, :2] += scene.env_origins[:, :2]
        root_probe[:, 3:7] = motion.motion_base_rots[i: i + 1]
        robot.write_root_state_to_sim(root_probe)
        jp = robot.data.default_joint_pos.clone()
        jp[:, robot_joint_indexes] = motion.motion_dof_poss[i: i + 1]
        robot.write_joint_state_to_sim(jp, robot.data.default_joint_vel.clone())
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        foot_z = robot.data.body_link_pos_w[0, foot_body_idxs, 2]
        min_foot_z = min(min_foot_z, float(foot_z.min()))

    z_correction = -min_foot_z
    print(f"[INFO]: Auto z-correction: {z_correction:+.4f}m (global min foot height: {min_foot_z:.4f}m)")
    motion.motion_base_poss[:, 2] += z_correction
    motion.motion_base_lin_vels[:, 2] = torch.gradient(
        motion.motion_base_poss[:, 2], spacing=motion.output_dt, dim=0
    )[0]
    scene.reset()

    print(f"\nPass 2/2: Running T1 forward kinematics for {motion.output_frames} frames...")
    pbar = tqdm(total=motion.output_frames, unit="frame", ncols=100)
    file_saved = False

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
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)
            print(f"Saving to {output_file}...")
            np.savez(output_file, **log)
            print(f"[INFO]: Motion saved to {output_file}")


def main(
    input_file: str,
    output_file: str = "t1_motion.npz",
    output_fps: float = 50.0,
    device: str = "cuda:0",
):
    """Convert a Booster T1 motion PKL (already T1 23-DOF) to NPZ for Mjlab tracking.

    Args:
        input_file: Path to the T1 pkl file.
        output_file: Where to save the output NPZ.
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

    motion = T1MotionLoader(pkl_path=input_file, output_fps=int(output_fps), device=device)

    run_sim(sim=sim, scene=scene, motion=motion, output_fps=int(output_fps), output_file=output_file)


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
