"""AMP on-policy runner with ONNX export.

Uses the bundled amp_rsl_rl package (AMP fork of rsl_rl, coexists with rsl-rl-lib 5.x).
"""

import os
import inspect

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from booster_t1_mjlab.amp_rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class _OnnxPolicyWrapper(torch.nn.Module):
    """Wraps actor + obs normalizer so the exported ONNX model takes raw obs."""

    def __init__(self, actor_critic, obs_normalizer=None):
        super().__init__()
        self.actor_critic = actor_critic
        self.obs_normalizer = obs_normalizer

    def forward(self, obs):
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        return self.actor_critic.act_inference(obs)


def _onnx_export_kwargs_single_file() -> dict:
    try:
        params = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        return {}
    if "external_data" in params:
        return {"external_data": False}
    if "use_external_data_format" in params:
        return {"use_external_data_format": False}
    return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
    data_path = f"{onnx_path}.data"
    if not os.path.exists(data_path):
        return
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(model, onnx_path, save_as_external_data=False)
        if os.path.exists(data_path):
            os.remove(data_path)
        print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
    except Exception as exc:
        print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


class AMPOnPolicyRunner(AmpOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def load(self, path: str, *, load_cfg=None, strict: bool = True, map_location=None, load_optimizer: bool = True):
        """Bridge rsl-rl-lib 5.x API (load_cfg/strict/map_location) to AMP runner API."""
        return super().load(path, load_optimizer=load_optimizer)

    def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
        policy = self.alg.policy
        obs_normalizer = None
        if self.empirical_normalization:
            obs_normalizer = self.obs_normalizer
            obs_normalizer.to("cpu")
            obs_normalizer.eval()
        wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
        wrapper.to("cpu")
        wrapper.eval()
        num_obs = policy.actor[0].in_features
        dummy_input = torch.zeros(1, num_obs)
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            **_onnx_export_kwargs_single_file(),
        )
        _inline_external_onnx_data(os.path.join(path, filename))
        policy.to(self.device)
        if obs_normalizer is not None:
            obs_normalizer.to(self.device)

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self._export_policy_to_onnx(policy_path, filename)
        run_name: str = (
            wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
        )
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        _inline_external_onnx_data(onnx_path)
        if self.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
