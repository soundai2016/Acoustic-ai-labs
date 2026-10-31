"""Huawei Ascend backend for a preconfigured CANN and TorchNPU stack."""

import importlib
import os


def probe():
    import torch
    torch_npu = importlib.import_module("torch_npu")
    from acoustic_labs.probe import training_probe

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available")
    if torch.npu.device_count() < 1:
        raise RuntimeError("No visible Ascend devices")
    torch.npu.set_device(0)
    config = getattr(torch.npu, "config", None)
    if config is not None:
        config.allow_internal_format = True
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    loss = training_probe(torch, device)
    torch.npu.synchronize()
    return device, {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "cann_home": os.getenv("ASCEND_HOME_PATH", "unset"),
        "visible_devices": torch.npu.device_count(),
        "probe_loss": loss,
    }
