"""Standalone TensorRT engine builder.

Run in a subprocess to get a clean CUDA context.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


class _NumpyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds pre-saved numpy calibration images to TRT one small batch at a time."""

    _BATCH = 4

    def __init__(self, calib_path: str):
        super().__init__()
        data = np.load(calib_path).astype(np.float32)
        n = (len(data) // self._BATCH) * self._BATCH
        self._batches = data[:n].reshape(-1, self._BATCH, *data.shape[1:])
        self._idx = 0
        self._buf = None

    def get_batch_size(self) -> int:
        return self._BATCH

    def get_batch(self, _names):
        if self._idx >= len(self._batches):
            return None
        self._buf = torch.from_numpy(self._batches[self._idx]).cuda().contiguous()
        self._idx += 1
        return [self._buf.data_ptr()]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib-data")
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser_trt = trt.OnnxParser(network, logger)

    ok = parser_trt.parse_from_file(args.onnx)
    if not ok:
        errors = "\n".join(
            str(parser_trt.get_error(i)) for i in range(parser_trt.num_errors)
        )
        raise RuntimeError(errors)

    inp = network.get_input(0)
    _, C, H, W = inp.shape

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)

    calib_batch = _NumpyCalibrator._BATCH if (args.int8 or args.fp8) else 8
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, C, H, W), (calib_batch, C, H, W), (16, C, H, W))
    config.add_optimization_profile(profile)

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if args.fp8:
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = _NumpyCalibrator(args.calib_data)

    if args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _NumpyCalibrator(args.calib_data)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None")

    Path(args.engine).write_bytes(serialized)


if __name__ == "__main__":
    main()
