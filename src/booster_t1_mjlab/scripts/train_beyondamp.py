"""Train a beyondAMP agent on the Booster T1 (mjlab backend).

Mirrors beyondAMP/scripts/factoryMjlab/train.py but also registers the
Booster T1 tasks before the task chooser runs.

Usage:
    uv run booster_t1_train_beyondamp Mjlab-BeyondAMP-Velocity-Flat-Booster-T1 \\
        --num-envs 4096

    # Override motion files at runtime:
    uv run booster_t1_train_beyondamp Mjlab-BeyondAMP-Velocity-Flat-Booster-T1 \\
        "--agent.amp-data.motion-files=[src/.../walk_forward.npz]"
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import torch.nn as nn

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


class _AMPOnPolicyRunnerWithOnnx(AMPOnPolicyRunner):
    """AMPOnPolicyRunner that also exports ONNX on every checkpoint save."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        try:
            self._export_onnx(path)
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")

    def _export_onnx(self, checkpoint_path: str) -> None:
        import torch, os
        from pathlib import Path

        ckpt = Path(checkpoint_path)
        onnx_path = ckpt.parent / f"{ckpt.parent.name}.onnx"

        actor: nn.Module = self.alg.actor_critic.actor
        actor.eval()
        num_obs = self.env.num_obs
        dummy = torch.zeros(1, num_obs, device="cpu")
        actor_cpu = actor.to("cpu")
        torch.onnx.export(
            actor_cpu,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
            dynamo=False,
        )
        actor_cpu.to(self.device)

        run_path = str(ckpt.parent)
        metadata = get_base_metadata(self.env.unwrapped, run_path)
        attach_metadata_to_onnx(str(onnx_path), metadata)
        print(f"[INFO] ONNX exported → {onnx_path.name}")


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: AMPRunnerCfg
    num_envs: int | None = None
    """Shortcut for --env.scene.num-envs."""
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    log_root: str = "logs/rsl_rl"
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, AMPRunnerCfg), (
            f"Task '{task_id}' is not an AMP task — got {type(agent_cfg).__name__}."
        )
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"

    configure_torch_backends()
    if cfg.num_envs is not None:
        cfg.env.scene.num_envs = cfg.num_envs
    cfg.env.seed = cfg.agent.seed

    print(f"[INFO] Training beyondAMP on mjlab: task={task_id} device={device}")
    print(f"[INFO] Logging to: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    if cfg.video:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = AMPEnvWrapper(env, clip_actions=cfg.agent.clip_actions, motion_dataset=cfg.agent.amp_data)

    agent_cfg_dict = asdict(cfg.agent)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)

    runner = _AMPOnPolicyRunnerWithOnnx(env, agent_cfg_dict, log_dir=str(log_dir), device=device)

    if cfg.agent.resume:
        log_root_path = log_dir.parent
        resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def main() -> None:
    maybe_print_top_level_help("train")

    import mjlab.tasks  # noqa: F401
    import booster_t1_mjlab.tasks  # noqa: F401 — registers T1 tasks
    import amp_tasks_mjlab  # noqa: F401 — registers G1 reference tasks

    import mjlab

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]
    if not amp_tasks:
        raise RuntimeError("No AMP tasks registered.")

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(amp_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, _ = select_gpus(args.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    run_train(chosen_task, args, log_dir)


if __name__ == "__main__":
    main()
