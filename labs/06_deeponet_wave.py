# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""DeepONet for a one-dimensional wave solution operator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from acoustic_labs.results import output_dir, record
from acoustic_labs.device import select_device, seed_torch

OUTPUT = output_dir()
selected = select_device()
DEVICE = selected.device
print(f"Device: requested={selected.requested}, selected={selected.backend}, "
      f"fallback_reason={selected.fallback_reason}")
seed_torch(torch, selected, 5)
POINTS, TRAIN_MODES, TIME = 64, 4, 0.30
x = np.linspace(0, 1, POINTS, dtype="float32")
coordinates = torch.tensor(x[:, None], device=DEVICE)

def make_dataset(size, seed, modes=TRAIN_MODES):
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(-1, 1, (size, modes)).astype("float32")
    harmonics = np.arange(1, modes + 1)
    basis = np.sin(np.pi * harmonics[:, None] * x).astype("float32")
    initial = coefficients @ basis
    evolved = (coefficients * np.cos(np.pi * harmonics * TIME)) @ basis
    return torch.tensor(initial), torch.tensor(evolved.astype("float32"))

train_x, train_y = make_dataset(240, 20)
val_x, val_y = make_dataset(48, 97)
test_x, test_y = make_dataset(48, 137)

class DeepONet(nn.Module):
    def __init__(self, rank=24):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(POINTS, 64), nn.Tanh(), nn.Linear(64, rank))
        self.trunk = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, rank))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, initial, query):
        features = [query]
        features += [torch.sin(k * np.pi * query) for k in range(1, 9)]
        features += [torch.cos(k * np.pi * query) for k in range(1, 9)]
        return self.branch(initial) @ self.trunk(torch.cat(features, dim=1)).T + self.bias

model = DeepONet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
history = []
best_loss = float("inf")
best_state = None
for epoch in range(301):
    for indices in torch.randperm(len(train_x)).split(40):
        optimizer.zero_grad()
        prediction = model(train_x[indices].to(DEVICE), coordinates)
        loss = nn.functional.mse_loss(prediction, train_y[indices].to(DEVICE))
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            prediction = model(val_x.to(DEVICE), coordinates)
            relative = (torch.linalg.vector_norm(prediction - val_y.to(DEVICE), dim=1)
                        / torch.linalg.vector_norm(val_y.to(DEVICE), dim=1).clamp_min(1e-6)).mean()
        history.append((epoch, relative.item()))
        if relative.item() < best_loss:
            best_loss = relative.item()
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:03d}: relative L2={relative.item():.4f}")
        model.train()

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_prediction = model(test_x.to(DEVICE), coordinates)
    test_error = (torch.linalg.vector_norm(test_prediction - test_y.to(DEVICE), dim=1)
                  / torch.linalg.vector_norm(test_y.to(DEVICE), dim=1).clamp_min(1e-6)).mean().item()
    predicted = test_prediction[0].cpu().numpy()
    ood_x, ood_y = make_dataset(10, 121, modes=6)
    ood_pred = model(ood_x.to(DEVICE), coordinates)
    ood_error = (torch.linalg.vector_norm(ood_pred - ood_y.to(DEVICE), dim=1)
                 / torch.linalg.vector_norm(ood_y.to(DEVICE), dim=1).clamp_min(1e-6)).mean().item()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].plot(x, test_x[0], ":", label="Initial")
axes[0].plot(x, test_y[0], label="Analytic")
axes[0].plot(x, predicted, "--", label="DeepONet")
axes[0].set(xlabel="Position", title="Unseen initial function")
axes[0].legend()
axes[1].semilogy(*zip(*history))
axes[1].set(xlabel="Epoch", ylabel="Relative L2", title="Validation functions")
fig.savefig(OUTPUT / "06_deeponet_wave.png", dpi=150)
plt.close(fig)
print(f"ID test relative L2={test_error:.4f}; OOD relative L2={ood_error:.4f}")

record("06_deeponet_wave", {"id_relative_l2": float(test_error),
                             "ood_relative_l2": float(ood_error),
                             "best_validation_relative_l2": float(best_loss)}, selected)
