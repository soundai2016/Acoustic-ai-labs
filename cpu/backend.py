"""CPU PyTorch backend."""


def probe():
    import torch
    from acoustic_labs.probe import training_probe

    device = torch.device("cpu")
    loss = training_probe(torch, device)
    return device, {"torch": torch.__version__, "probe_loss": loss}
