import copy
from pathlib import Path

import torch
import torch.nn as nn

from src.config import IMG_SCALE


_CHECKPOINTS = {
    0.5: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    1.0: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}


def build_model(scale=IMG_SCALE):
    """Load milesial/Pytorch-UNet pretrained on Carvana, always to CPU first."""
    model = torch.hub.load(
        "milesial/Pytorch-UNet",
        "unet_carvana",
        pretrained=False,
        scale=scale,
    )
    if scale not in _CHECKPOINTS:
        raise ValueError(f"No pretrained checkpoint for scale={scale}. Use 0.5 or 1.0.")
    state_dict = torch.hub.load_state_dict_from_url(
        _CHECKPOINTS[scale], map_location="cpu", progress=True
    )
    state_dict.pop("mask_values", None)
    model.load_state_dict(state_dict)
    return model


def apply_compiled(model):
    """torch.compile with max-autotune (no CUDA graphs to avoid private pool OOM)."""
    return torch.compile(model, mode="max-autotune-no-cudagraphs")


def apply_ptq(model):
    """PyTorch dynamic INT8 quantization (CPU only)."""
    model = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )


def export_to_onnx(model, sample_input: torch.Tensor, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.cpu().eval(),
        sample_input.cpu(),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to {onnx_path}")


def apply_ort_fp16(onnx_path: Path, fp16_path: Path) -> None:
    """Convert ONNX model weights to FP16 (I/O stays FP32 via keep_io_types)."""
    import onnx
    from onnxconverter_common import float16

    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))
    print(f"Saved FP16 ONNX model to {fp16_path}")


def apply_ort_int8(onnx_path: Path, int8_path: Path, calib_loader) -> None:
    """Static INT8 quantization of an ONNX model via calibration data."""
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )

    class _CalibReader(CalibrationDataReader):
        def __init__(self, loader):
            self._iter = iter(loader)

        def get_next(self):
            try:
                imgs, _ = next(self._iter)
                return {"input": imgs.numpy()}
            except StopIteration:
                return None

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(onnx_path),
        str(int8_path),
        _CalibReader(calib_loader),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Saved INT8 ONNX model to {int8_path}")


class ORTModel:
    """ORT InferenceSession wrapped to match the PyTorch model interface."""

    def __init__(self, model_path: Path, device: torch.device):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        active = self.session.get_providers()
        if device.type == "cuda" and "CUDAExecutionProvider" not in active:
            print("ORT: CUDAExecutionProvider unavailable, falling back to CPU")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Fixed output channels from model metadata (dim 1 of output shape)
        out_shape = self.session.get_outputs()[0].shape
        self._out_channels = int(out_shape[1])
        self.device = device
        self.model_path = model_path

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            import numpy as np
            binding = self.session.io_binding()
            x = x.contiguous()
            binding.bind_input(
                name=self.input_name,
                device_type="cuda", device_id=0,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            out = torch.empty(
                (x.shape[0], self._out_channels, x.shape[2], x.shape[3]),
                dtype=torch.float32, device=self.device,
            )
            binding.bind_output(
                name=self.output_name,
                device_type="cuda", device_id=0,
                element_type=np.float32,
                shape=tuple(out.shape),
                buffer_ptr=out.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
            return out
        out = self.session.run(
            [self.output_name], {self.input_name: x.numpy()}
        )[0]
        return torch.from_numpy(out)

    def eval(self): return self
    def to(self, device): self.device = device; return self
    def parameters(self): return iter([])
    def buffers(self): return iter([])
