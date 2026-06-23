"""Preview a retargeted T1 motion NPZ file using MuJoCo passive viewer.

Usage:
    uv run python src/booster_t1_mjlab/scripts/preview_motion.py \
        src/booster_t1_mjlab/assets/motions/t1/amp/WalkandRun/walk_forward_loop_002__A022.npz

Controls: the viewer plays back in real time. Close the window to exit.
"""

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# HOME_KEYFRAME joint positions (matches t1_constants.py HOME_KEYFRAME).
_HOME_EXACT: dict[str, float] = {
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
}
_HOME_REGEX: list[tuple[str, float]] = [
    (r".*_Hip_Pitch", -0.2),
    (r".*_Knee_Pitch", 0.4),
    (r".*_Ankle_Pitch", -0.2),
]


def _build_home_offset(model: mujoco.MjModel) -> np.ndarray:
    """Return a (nq-7,) array of HOME_KEYFRAME joint offsets in MuJoCo qpos order."""
    n = model.nq - 7
    home = np.zeros(n)
    for ji in range(1, model.njnt):  # skip freejoint at index 0
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
        idx = int(model.jnt_qposadr[ji]) - 7
        if 0 <= idx < n:
            val = _HOME_EXACT.get(name, 0.0)
            if val == 0.0:
                for pattern, v in _HOME_REGEX:
                    if re.fullmatch(pattern, name):
                        val = v
                        break
            home[idx] = val
    return home


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to retargeted T1 .npz file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop playback")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End frame (exclusive, -1 = all)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path)

    fps = float(np.asarray(data["fps"]).flat[0])
    joint_pos = data["joint_pos"]   # (T, 21)
    body_pos  = data["body_pos_w"]  # (T, N_bodies, 3)
    body_quat = data["body_quat_w"] # (T, N_bodies, 4) — (w, x, y, z)
    T = joint_pos.shape[0]

    start = max(0, args.start)
    end = T if args.end < 0 else min(args.end, T)
    joint_pos = joint_pos[start:end]
    body_pos  = body_pos[start:end]
    body_quat = body_quat[start:end]
    T = joint_pos.shape[0]

    n_dofs = joint_pos.shape[1]
    is_absolute = int(np.asarray(data.get("joint_pos_absolute", [0])).flat[0]) == 1

    _xmls = Path(__file__).parent.parent / "robots" / "boostert1" / "xmls"
    if n_dofs == 21:
        xml_path = _xmls / "T1_21dof_deploy.xml"
    elif n_dofs == 23:
        xml_path = _xmls / "T1_23dof_deploy.xml"
    else:
        raise ValueError(f"Unexpected DOF count in NPZ: {n_dofs} (expected 21 or 23)")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mjdata = mujoco.MjData(model)
    # G1-retargeted NPZs store joint_pos relative to HOME → add HOME back.
    # T1-native kick NPZs flag joint_pos_absolute=1 → no offset needed.
    home_offset = np.zeros(model.nq - 7) if is_absolute else _build_home_offset(model)

    dt = 1.0 / (fps * args.speed)
    print(f"Playing {npz_path.name}: {T} frames @ {fps:.0f} fps, duration {T/fps:.1f}s")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        frame = 0
        while viewer.is_running():
            t_start = time.perf_counter()

            # Root position and orientation.
            # Both NPZ (w,x,y,z) and MuJoCo qpos (w,x,y,z) use the same convention.
            mjdata.qpos[:3]  = body_pos[frame, 0, :]
            mjdata.qpos[3:7] = body_quat[frame, 0, :]
            mjdata.qpos[7:]  = joint_pos[frame] + home_offset
            mjdata.qvel[:]   = 0.0

            mujoco.mj_forward(model, mjdata)
            viewer.sync()

            frame += 1
            if frame >= T:
                if args.loop:
                    frame = 0
                else:
                    break

            elapsed = time.perf_counter() - t_start
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
