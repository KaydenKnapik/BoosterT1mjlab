"""One-shot script to export ONNX + metadata from an existing getup checkpoint."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

import booster_t1_mjlab.tasks  # noqa: F401 — registers tasks
from booster_t1_mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from booster_t1_mjlab.tasks.getup.rl.runner import _get_getup_metadata
from mjlab.rl.exporter_utils import attach_metadata_to_onnx

TASK_ID = "Mjlab-Getup-Flat-Booster-T1"
CHECKPOINT = Path("/home/robocup/BoosterT1mjlab/logs/rsl_rl/t1_getup/2026-06-13_15-55-54/model_9800.pt")

run_dir = CHECKPOINT.parent
onnx_path = run_dir / f"{run_dir.name}.onnx"

env_cfg = load_env_cfg(TASK_ID)
env_cfg.scene.num_envs = 1

rl_cfg = load_rl_cfg(TASK_ID)
runner_cls = load_runner_cls(TASK_ID)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(env)

runner = runner_cls(env, asdict(rl_cfg), str(run_dir), "cuda:0")
runner.load(str(CHECKPOINT))

runner.export_policy_to_onnx(str(run_dir), filename=onnx_path.name)
metadata = _get_getup_metadata(env.unwrapped, str(run_dir))
attach_metadata_to_onnx(str(onnx_path), metadata)

print(f"[INFO] Exported → {onnx_path}")

import onnx
m = onnx.load(str(onnx_path))
props = {p.key: p.value for p in m.metadata_props}
print(f"[INFO] Metadata keys: {list(props.keys())}")
print(f"[INFO] joint_names: {props.get('joint_names', 'MISSING')[:80]}")
print(f"[INFO] action_scale: {props.get('action_scale', 'MISSING')}")

env.close()
