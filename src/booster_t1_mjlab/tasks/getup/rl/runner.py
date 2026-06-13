"""Getup task runner with automatic ONNX export on every checkpoint save."""

import torch
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.entity import Entity


def _get_getup_metadata(env, run_path: str) -> dict:
    """Build ONNX metadata for the getup task.

    Same fields as get_base_metadata but works with RelativeJointPositionAction
    which does not inherit from JointPositionAction.
    """
    robot: Entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")

    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id

    ctrl_ids_natural = [
        joint_name_to_ctrl_id[jname]
        for jname in robot.joint_names
        if jname in joint_name_to_ctrl_id
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

    scale = joint_action._scale
    action_scale = scale[0].cpu().tolist() if isinstance(scale, torch.Tensor) else scale

    return {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness.tolist(),
        "joint_damping": joint_damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": action_scale,
    }


class GetupOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        try:
            ckpt = Path(path)
            run_dir = ckpt.parent
            filename = f"{run_dir.name}.onnx"
            self.export_policy_to_onnx(str(run_dir), filename=filename)
            onnx_path = run_dir / filename
            metadata = _get_getup_metadata(self.env.unwrapped, str(run_dir))
            attach_metadata_to_onnx(str(onnx_path), metadata)
            print(f"[INFO] ONNX exported → {onnx_path}")
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
