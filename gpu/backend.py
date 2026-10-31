"""NVIDIA CUDA backend with a real training-step preflight."""


def probe():
    import torch
    from acoustic_labs.probe import training_probe

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    loss = training_probe(torch, device)
    torch.cuda.synchronize(device)
    return device, {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(device), "probe_loss": loss}
