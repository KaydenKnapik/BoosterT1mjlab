"""Retarget G1 AMP motion NPZ files to Booster T1 headless (21-DOF).

Usage:
    uv run python src/booster_t1_mjlab/scripts/retarget_g1_to_t1.py \
        --src-dir AMP_mjlab/src/assets/motions/g1/amp/WalkandRun \
        --dst-dir src/booster_t1_mjlab/assets/motions/t1/amp/WalkandRun

Approach:
    1. Map G1 joint angles → T1 joint angles.
       T1_angle = T1_default + G1_angle  (offsets the retargeted angles to T1's
       natural resting pose, matching HOME_KEYFRAME in t1_constants.py).
    2. Preserve root pose from the G1 clip (both are bipeds).
    3. Recompute all T1 body poses via MuJoCo forward kinematics.
    4. Compute body velocities via finite differences.
    5. Body quaternions stored as (w, x, y, z) — MuJoCo qpos and xquat both
       use (w, x, y, z) convention, which matches the G1 NPZ format directly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# T1 HOME_KEYFRAME default joint positions (from t1_constants.py).
# These are the absolute qpos values for T1's natural standing pose.
# G1 stands with joints near zero, T1 does not — we offset every retargeted
# joint by T1's default so the motion is expressed in T1's posture space.
# ---------------------------------------------------------------------------
_T1_HOME_JOINT_POS: dict[str, float] = {
    "Left_Shoulder_Roll":  -1.4,
    "Left_Elbow_Yaw":      -0.4,
    "Right_Shoulder_Roll":  1.4,
    "Right_Elbow_Yaw":      0.4,
    "Left_Hip_Pitch":      -0.2,
    "Right_Hip_Pitch":     -0.2,
    "Left_Knee_Pitch":      0.4,
    "Right_Knee_Pitch":     0.4,
    "Left_Ankle_Pitch":    -0.2,
    "Right_Ankle_Pitch":   -0.2,
}


def _build_t1_default_qpos(t1_joint_names: list[str]) -> np.ndarray:
    """Return default joint positions array in T1 joint ordering."""
    arr = np.zeros(len(t1_joint_names), dtype=np.float32)
    for i, name in enumerate(t1_joint_names):
        arr[i] = _T1_HOME_JOINT_POS.get(name, 0.0)
    return arr


# ---------------------------------------------------------------------------
# Joint mapping: G1 (29 actuated DOF) → T1 headless (21 actuated DOF)
# Each entry: (g1_joint_name, t1_joint_name | None)
# None means no T1 equivalent — that T1 DOF stays at its default (zero).
# ---------------------------------------------------------------------------
_G1_TO_T1: list[tuple[str, str | None]] = [
    # --- Legs (direct structural equivalents) ---
    ("left_hip_pitch_joint",     "Left_Hip_Pitch"),
    ("left_hip_roll_joint",      "Left_Hip_Roll"),
    ("left_hip_yaw_joint",       "Left_Hip_Yaw"),
    ("left_knee_joint",          "Left_Knee_Pitch"),
    ("left_ankle_pitch_joint",   "Left_Ankle_Pitch"),
    ("left_ankle_roll_joint",    "Left_Ankle_Roll"),
    ("right_hip_pitch_joint",    "Right_Hip_Pitch"),
    ("right_hip_roll_joint",     "Right_Hip_Roll"),
    ("right_hip_yaw_joint",      "Right_Hip_Yaw"),
    ("right_knee_joint",         "Right_Knee_Pitch"),
    ("right_ankle_pitch_joint",  "Right_Ankle_Pitch"),
    ("right_ankle_roll_joint",   "Right_Ankle_Roll"),
    # --- Waist (G1 has 3 DOF, T1 has 1 yaw DOF) ---
    ("waist_yaw_joint",          "Waist"),
    ("waist_roll_joint",         None),   # T1 has no waist roll
    ("waist_pitch_joint",        None),   # T1 has no waist pitch
    # --- Left arm ---
    ("left_shoulder_pitch_joint", "Left_Shoulder_Pitch"),
    ("left_shoulder_roll_joint",  "Left_Shoulder_Roll"),
    ("left_shoulder_yaw_joint",   "Left_Elbow_Yaw"),    # upper-arm rotation → forearm yaw
    ("left_elbow_joint",          "Left_Elbow_Pitch"),
    ("left_wrist_roll_joint",     None),
    ("left_wrist_pitch_joint",    None),
    ("left_wrist_yaw_joint",      None),
    # --- Right arm ---
    ("right_shoulder_pitch_joint", "Right_Shoulder_Pitch"),
    ("right_shoulder_roll_joint",  "Right_Shoulder_Roll"),
    ("right_shoulder_yaw_joint",   "Right_Elbow_Yaw"),
    ("right_elbow_joint",          "Right_Elbow_Pitch"),
    ("right_wrist_roll_joint",     None),
    ("right_wrist_pitch_joint",    None),
    ("right_wrist_yaw_joint",      None),
]


def _build_index_map(
    g1_joints: list[str],
    t1_joints: list[str],
) -> list[tuple[int, int]]:
    """Return (g1_idx, t1_idx) pairs for joints that have a T1 equivalent."""
    t1_idx = {name: i for i, name in enumerate(t1_joints)}
    g1_idx = {name: i for i, name in enumerate(g1_joints)}
    pairs = []
    for g1_name, t1_name in _G1_TO_T1:
        if t1_name is not None and g1_name in g1_idx and t1_name in t1_idx:
            pairs.append((g1_idx[g1_name], t1_idx[t1_name]))
    return pairs


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w,x,y,z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _canonicalize_root(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate trajectory so frame-0 faces world +X (yaw = 0).

    Returns corrected (root_pos, root_quat) arrays.  XY is centred at the
    frame-0 position; Z is left unchanged (height correction is done later).
    """
    w, x, y, z = root_quat[0]
    yaw0 = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    # Quaternion that undoes the initial yaw: Rz(-yaw0).
    half = -yaw0 / 2.0
    corr = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    c, s = np.cos(-yaw0), np.sin(-yaw0)
    origin = root_pos[0].copy()

    new_pos = root_pos.copy()
    dx = root_pos[:, 0] - origin[0]
    dy = root_pos[:, 1] - origin[1]
    new_pos[:, 0] = c * dx - s * dy   # centred at origin, facing +X
    new_pos[:, 1] = s * dx + c * dy
    new_pos[:, 2] = root_pos[:, 2]

    new_quat = np.array([_quat_mul_wxyz(corr, q) for q in root_quat])
    return new_pos, new_quat




def retarget_clip(
    src_path: Path,
    t1_model: mujoco.MjModel,
    t1_data: mujoco.MjData,
    g1_joint_names: list[str],
    t1_joint_names: list[str],
    t1_body_names: list[str],
    index_map: list[tuple[int, int]],
    t1_default_qpos: np.ndarray,
) -> dict:
    """Retarget a single G1 NPZ clip to T1 and return the result dict."""
    src = np.load(src_path)
    fps = float(np.asarray(src["fps"]).flat[0])
    dt = 1.0 / fps

    g1_joint_pos = src["joint_pos"]   # (T, 29)
    g1_joint_vel = src["joint_vel"]   # (T, 29)
    g1_body_pos  = src["body_pos_w"]  # (T, 30, 3)
    g1_body_quat = src["body_quat_w"] # (T, 30, 4) — stored as (w, x, y, z)

    T = g1_joint_pos.shape[0]
    n_t1_joints = len(t1_joint_names)
    n_t1_bodies = len(t1_body_names)  # excludes world body

    # Initialise at T1's default pose so unmapped joints land at their resting angle.
    t1_joint_pos  = np.tile(t1_default_qpos, (T, 1)).astype(np.float32)
    t1_joint_vel  = np.zeros((T, n_t1_joints), dtype=np.float32)
    t1_body_pos   = np.zeros((T, n_t1_bodies, 3), dtype=np.float32)
    t1_body_quat  = np.zeros((T, n_t1_bodies, 4), dtype=np.float32)

    # --- Map joint angles from G1 → T1: T1_angle = T1_default + G1_angle ---
    for g1_i, t1_i in index_map:
        t1_joint_pos[:, t1_i] = t1_default_qpos[t1_i] + g1_joint_pos[:, g1_i]
        t1_joint_vel[:, t1_i] = g1_joint_vel[:, g1_i]

    # G1 root is body index 0 in NPZ (pelvis), stored as (w, x, y, z).
    # Canonicalize: rotate so frame-0 faces world +X (yaw = 0), centred at origin.
    g1_root_pos, g1_root_quat_wxyz = _canonicalize_root(
        g1_body_pos[:, 0, :].copy(),
        g1_body_quat[:, 0, :].copy(),
    )

    # Foot body indices (0-based in NPZ = body_id - 1 in MuJoCo).
    foot_npz_indices = [
        i for i, name in enumerate(t1_body_names) if "foot_link" in name.lower()
    ]

    # --- Run FK on T1 for each frame ---
    for t in range(T):
        # Root orientation: copy directly — both G1 NPZ and MuJoCo qpos use (w, x, y, z).
        root_quat_wxyz = g1_root_quat_wxyz[t]

        # Set qpos: [root_pos(3), root_quat_wxyz(4), joints(21)]
        t1_data.qpos[:3]   = g1_root_pos[t]
        t1_data.qpos[3:7]  = root_quat_wxyz
        t1_data.qpos[7:]   = t1_joint_pos[t]

        mujoco.mj_kinematics(t1_model, t1_data)

        # Read body poses — skip body 0 (world), store bodies 1..n_t1_bodies.
        for b in range(n_t1_bodies):
            body_id = b + 1  # skip world
            t1_body_pos[t, b]  = t1_data.xpos[body_id]
            # MuJoCo xquat returns (w, x, y, z) — same as G1 NPZ convention.
            t1_body_quat[t, b] = t1_data.xquat[body_id]

        # Height correction: shift all bodies so the lowest foot_link body sits at
        # its natural standing height (~0.03 m above ground), matching T1 HOME pose.
        # G1 pelvis ≠ T1 Trunk so without this correction the robot floats.
        _FOOT_GROUND_CLEARANCE = 0.03  # matches left/right_foot_link z at HOME_KEYFRAME
        if foot_npz_indices:
            min_foot_z = min(t1_body_pos[t, fi, 2] for fi in foot_npz_indices)
            t1_body_pos[t, :, 2] -= (min_foot_z - _FOOT_GROUND_CLEARANCE)

    # --- Compute body linear/angular velocities via finite differences ---
    t1_body_lin_vel  = np.zeros_like(t1_body_pos)
    t1_body_ang_vel  = np.zeros_like(t1_body_pos)

    # Central differences for interior frames, forward/backward at ends.
    pos = t1_body_pos
    t1_body_lin_vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    t1_body_lin_vel[0]    = (pos[1]  - pos[0])   / dt
    t1_body_lin_vel[-1]   = (pos[-1] - pos[-2])  / dt

    # Angular velocity from quaternion differences (approximate).
    for t in range(T):
        if t == 0:
            t_prev, t_next = 0, min(1, T - 1)
            scale = 1.0 / dt
        elif t == T - 1:
            t_prev, t_next = max(0, T - 2), T - 1
            scale = 1.0 / dt
        else:
            t_prev, t_next = t - 1, t + 1
            scale = 1.0 / (2.0 * dt)

        for b in range(n_t1_bodies):
            q0 = t1_body_quat[t_prev, b]  # (w, x, y, z)
            q1 = t1_body_quat[t_next, b]

            # q_rel = q1 * q0_inv  (rotation from t_prev to t_next)
            w0, x0, y0, z0 = q0
            w1, x1, y1, z1 = q1
            # q0_inv = (w0, -x0, -y0, -z0) (unit quat conjugate)
            # q_rel = q1 * q0_inv
            q_rel_w = w1*w0 + x1*x0 + y1*y0 + z1*z0
            q_rel_x = x1*w0 - w1*x0 - y1*z0 + z1*y0  # noqa: wrong sign fixed below
            q_rel_y = y1*w0 + x1*z0 - w1*y0 - z1*x0
            q_rel_z = z1*w0 - x1*y0 + y1*x0 - w1*z0

            # Re-derive correctly: q1 * conj(q0)
            # conj(q0) = (w0, -x0, -y0, -z0)
            cw, cx, cy, cz = w0, -x0, -y0, -z0
            rw = w1*cw - x1*cx - y1*cy - z1*cz
            rx = w1*cx + x1*cw + y1*cz - z1*cy
            ry = w1*cy - x1*cz + y1*cw + z1*cx
            rz = w1*cz + x1*cy - y1*cx + z1*cw

            # Angular velocity ≈ 2 * (rx, ry, rz) / dt (small angle approx)
            # Clamp rw to [-1, 1] to avoid NaN in acos.
            rw = float(np.clip(rw, -1.0, 1.0))
            norm_xyz = float(np.sqrt(rx*rx + ry*ry + rz*rz)) + 1e-10
            angle = 2.0 * np.arccos(abs(rw)) * scale
            if rw < 0:
                angle = -angle
            axis = np.array([rx, ry, rz]) / norm_xyz
            t1_body_ang_vel[t, b] = axis * angle

    return {
        "fps":           np.array([fps], dtype=np.float32),
        "joint_pos":     t1_joint_pos.astype(np.float32),
        "joint_vel":     t1_joint_vel.astype(np.float32),
        "body_pos_w":    t1_body_pos.astype(np.float32),
        "body_quat_w":   t1_body_quat.astype(np.float32),
        "body_lin_vel_w": t1_body_lin_vel.astype(np.float32),
        "body_ang_vel_w": t1_body_ang_vel.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget G1 AMP NPZ clips to Booster T1.")
    parser.add_argument("--src-dir", required=True, help="Directory of G1 .npz files")
    parser.add_argument("--dst-dir", required=True, help="Output directory for T1 .npz files")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # --- Load T1 headless model ---
    _here = Path(__file__).parent
    t1_xml = _here.parent / "robots" / "boostert1" / "xmls" / "t1_headless.xml"
    assert t1_xml.exists(), f"T1 headless XML not found: {t1_xml}"
    t1_model = mujoco.MjModel.from_xml_path(str(t1_xml))
    t1_data  = mujoco.MjData(t1_model)

    # Joint names (skip joint 0 = free root joint).
    t1_joint_names = [t1_model.joint(i).name for i in range(1, t1_model.njnt)]
    # Body names (skip body 0 = world).
    t1_body_names  = [t1_model.body(i).name for i in range(1, t1_model.nbody)]

    print(f"T1 joints ({len(t1_joint_names)}): {t1_joint_names}")
    print(f"T1 bodies ({len(t1_body_names)})")

    # G1 joint names (skip free joint at index 0).
    g1_xml_path = src_dir.parents[4] / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
    if not g1_xml_path.exists():
        # Fallback: derive from the known G1 ordering.
        g1_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    else:
        g1_model = mujoco.MjModel.from_xml_path(str(g1_xml_path))
        g1_joint_names = [g1_model.joint(i).name for i in range(1, g1_model.njnt)]

    index_map = _build_index_map(g1_joint_names, t1_joint_names)
    t1_default_qpos = _build_t1_default_qpos(t1_joint_names)
    print(f"Mapped {len(index_map)} joints from G1 → T1")
    print(f"T1 default offsets: {dict(zip(t1_joint_names, t1_default_qpos))}")

    npz_files = sorted(src_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {src_dir}")
        return

    for src_path in npz_files:
        print(f"  Retargeting {src_path.name} ...", end=" ", flush=True)
        result = retarget_clip(
            src_path, t1_model, t1_data,
            g1_joint_names, t1_joint_names, t1_body_names,
            index_map, t1_default_qpos,
        )
        dst_path = dst_dir / src_path.name
        np.savez(dst_path, **result)
        frames = result["joint_pos"].shape[0]
        print(f"{frames} frames → {dst_path.name}")

    print(f"\nDone. {len(npz_files)} clip(s) written to {dst_dir}")


if __name__ == "__main__":
    main()
