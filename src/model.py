import copy
import torch
import torch.nn as nn

from src.config import IMG_SCALE


_CHECKPOINTS = {
    0.5: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    1.0: "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth",
}


def build_model(scale=IMG_SCALE):
    """Load milesial/Pytorch-UNet pretrained on Carvana, always to CPU."""
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


def apply_trt_int8(model, calib_loader, device):
    """Compile model to TensorRT INT8 engine using entropy calibration."""
    import torch_tensorrt

    model = copy.deepcopy(model).to(device).eval()

    sample, _ = next(iter(calib_loader))
    B, C, H, W = sample.shape

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        calib_loader,
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=device,
    )

    scripted = torch.jit.trace(model, sample.to(device))
    trt_model = torch_tensorrt.compile(
        scripted,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, C, H, W),
            opt_shape=(B, C, H, W),
            max_shape=(B, C, H, W),
        )],
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        truncate_long_and_double=True,
    )
    return trt_model
