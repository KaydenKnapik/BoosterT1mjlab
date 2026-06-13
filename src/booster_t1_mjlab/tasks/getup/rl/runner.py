"""Getup task runner with automatic ONNX export on every checkpoint save."""

from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata


class GetupOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        try:
            ckpt = Path(path)
            run_dir = ckpt.parent
            filename = f"{run_dir.name}.onnx"
            self.export_policy_to_onnx(str(run_dir), filename=filename)
            onnx_path = run_dir / filename
            metadata = get_base_metadata(self.env.unwrapped, str(run_dir))
            attach_metadata_to_onnx(str(onnx_path), metadata)
            print(f"[INFO] ONNX exported → {onnx_path}")
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
