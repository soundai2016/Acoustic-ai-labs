"""Run a small FP32 forward, backward, and optimizer step on a device."""


def training_probe(torch, device):
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * 4 * 4, 2),
    ).to(device)
    x = torch.ones((2, 1, 4, 4), dtype=torch.float32, device=device)
    target = torch.tensor([0, 1], device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(net(x), target)
    loss.backward()
    optimizer.step()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Nonfinite training probe")
    query = torch.ones((2, 4), device=device)
    attention = torch.softmax(query @ query.T, dim=-1)
    if not bool(torch.isfinite(attention).all().item()):
        raise FloatingPointError("Nonfinite attention probe")
    features = torch.ones((2, 6), device=device, requires_grad=True)
    core = torch.nn.LayerNorm(6).to(device)(features)
    loss2 = (torch.sin(core) + torch.cos(core)).tanh().square().mean()
    loss2.backward()
    if not bool(torch.isfinite(loss2).item()):
        raise FloatingPointError("Nonfinite neural-operator probe")
    return float(loss.detach().cpu())
