import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def export_to_onnx(model, sample_input: torch.Tensor, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.cpu().eval(),
        sample_input.cpu(),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_path}")


def save_calib_data(loader, n_samples: int, calib_path: Path) -> None:
    """Collect calibration images from loader and save as a numpy array."""
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    batches = []
    collected = 0
    for imgs, _ in loader:
        batches.append(imgs.numpy())
        collected += imgs.shape[0]
        if collected >= n_samples:
            break
    data = np.concatenate(batches, axis=0)[:n_samples]
    np.save(str(calib_path), data)
    print(f"Saved {len(data)} calibration images to {calib_path}")


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = False,
    fp8: bool = False,
    int8: bool = False,
    calib_path: Path = None,
    workspace_gb: int = 2,
) -> None:
    """Spawn a subprocess for the TRT build to avoid CUDA context conflicts."""
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "_trt_build.py"
    cmd = [
        sys.executable,
        str(script),
        "--onnx",
        str(onnx_path),
        "--engine",
        str(engine_path),
        "--workspace-gb",
        str(workspace_gb),
    ]
    if fp16:
        cmd.append("--fp16")
    if fp8:
        cmd.extend(["--fp8", "--calib-data", str(calib_path)])
    if int8:
        cmd.extend(["--int8", "--calib-data", str(calib_path)])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"TRT engine build subprocess failed (exit {result.returncode})"
        )
    size_mb = engine_path.stat().st_size / 1024**2
    print(f"TRT engine: {engine_path} ({size_mb:.1f} MB)")


class TRTModel:
    """TensorRT engine wrapper matching the PyTorch model call interface."""

    def __init__(self, engine_path: Path, device: torch.device):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self._input_name = self.engine.get_tensor_name(0)
        self._output_name = self.engine.get_tensor_name(1)
        self.device = device
        self.model_path = engine_path

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        self.context.set_input_shape(self._input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(self._input_name, x.data_ptr())
        self.context.set_tensor_address(self._output_name, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return iter([])

    def buffers(self):
        return iter([])
