"""Export a beyondAMP policy checkpoint to ONNX with metadata.

The actor is a plain MLP (no obs normalizer), so we export actor_critic.actor
directly the same way VelocityOnPolicyRunner does for velocity policies.

Usage:
    uv run booster_t1_export_beyondamp \\
        logs/rsl_rl/t1_beyondamp/2026-06-07_12-00-00_amp/model_3200.pt

    # Custom output path:
    uv run booster_t1_export_beyondamp \\
        logs/rsl_rl/t1_beyondamp/.../model_3200.pt \\
        --output /tmp/t1_amp_policy.onnx
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg


def export_amp_checkpoint(
    checkpoint_path: str,
    task_id: str,
    output_path: str | None = None,
    device: str = "cpu",
) -> Path:
    """Export actor MLP from an AMP checkpoint to ONNX with metadata.

    Returns the path of the written ONNX file.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if output_path is None:
        onnx_path = ckpt_path.parent / f"{ckpt_path.parent.name}.onnx"
    else:
        onnx_path = Path(output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint and reconstruct actor ---
    print(f"[INFO] Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Infer architecture from weights: actor.0.weight shape is [hidden0, num_obs]
    weight_keys = sorted(
        [k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    num_actor_obs = state_dict[weight_keys[0]].shape[1]
    num_actions = state_dict[weight_keys[-1]].shape[0]
    hidden_dims = [state_dict[k].shape[0] for k in weight_keys[:-1]]
    print(f"[INFO] Actor: obs={num_actor_obs}, actions={num_actions}, hidden={hidden_dims}")

    # Rebuild actor Sequential matching the original construction in rsl_rl_amp
    actor_layers: list[nn.Module] = []
    actor_layers.append(nn.Linear(num_actor_obs, hidden_dims[0]))
    actor_layers.append(nn.ELU())
    for i in range(len(hidden_dims)):
        out = num_actions if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
        actor_layers.append(nn.Linear(hidden_dims[i], out))
        if i < len(hidden_dims) - 1:
            actor_layers.append(nn.ELU())
    actor = nn.Sequential(*actor_layers)

    actor_state = {
        k[len("actor."):]: v
        for k, v in state_dict.items()
        if k.startswith("actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval().to(device)

    # --- Export to ONNX ---
    dummy_obs = torch.zeros(1, num_actor_obs, device=device)
    os.makedirs(str(onnx_path.parent), exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_obs,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    print(f"[INFO] ONNX exported to: {onnx_path}")

    # --- Attach metadata (needs env for joint names, stiffness, etc.) ---
    print("[INFO] Building env to extract metadata...")
    os.environ.setdefault("MUJOCO_GL", "egl")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AMPRunnerCfg)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env_wrapped = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    run_path = str(ckpt_path.parent)
    metadata = get_base_metadata(env.unwrapped, run_path)
    env_wrapped.close()

    attach_metadata_to_onnx(str(onnx_path), metadata)
    print(f"[INFO] Metadata attached: {list(metadata.keys())}")
    print(f"[INFO] Done → {onnx_path}")
    return onnx_path


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import booster_t1_mjlab.tasks  # noqa: F401
    import amp_tasks_mjlab  # noqa: F401

    import mjlab
    from mjlab.tasks.registry import list_tasks

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]

    import argparse
    parser = argparse.ArgumentParser(description="Export beyondAMP checkpoint to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument(
        "--task",
        default="Mjlab-BeyondAMP-Velocity-Flat-Booster-T1",
        choices=amp_tasks,
        help="Task ID to use for metadata (default: Mjlab-BeyondAMP-Velocity-Flat-Booster-T1)",
    )
    parser.add_argument("--output", default=None, help="Output .onnx path (default: same dir as checkpoint)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    export_amp_checkpoint(
        checkpoint_path=args.checkpoint,
        task_id=args.task,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
