# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""Scaled dot-product attention and a tiny sequence Transformer."""

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
seed_torch(torch, selected, 2)

queries = np.array([
    [1., 0., 0.], [0.8, 0.2, 0.], [0., 1., 0.],
    [0., 0.7, 0.3], [0., 0., 1.], [0.1, 0., 0.9],
])
scores = queries @ queries.T / np.sqrt(queries.shape[1])
scores -= scores.max(axis=1, keepdims=True)
weights = np.exp(scores)
weights /= weights.sum(axis=1, keepdims=True)
assert np.allclose(weights.sum(axis=1), 1)

def make_dataset(size, seed):
    rng = np.random.default_rng(seed)
    examples, labels = [], []
    for _ in range(size):
        label = int(rng.integers(2))
        ridge = np.linspace(0, 7, 12)
        if label:
            ridge = ridge[::-1]
        bins = np.arange(8)[None, :]
        spectrum = np.exp(-0.5 * ((bins - ridge[:, None]) / 0.9) ** 2)
        spectrum += 0.17 * rng.normal(size=spectrum.shape)
        examples.append(spectrum.astype("float32"))
        labels.append(label)
    return torch.tensor(np.stack(examples)), torch.tensor(labels, dtype=torch.long)

train_x, train_y = make_dataset(240, 12)
val_x, val_y = make_dataset(80, 19)
test_x, test_y = make_dataset(80, 29)

class TinyTransformer(nn.Module):
    """Single multi-head self-attention block, using portable FP32 operators."""

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(8, 24)
        self.position = nn.Parameter(torch.zeros(1, 12, 24))
        self.qkv = nn.Linear(24, 72)
        self.combine = nn.Linear(24, 24)
        self.norm1 = nn.LayerNorm(24)
        self.ff = nn.Sequential(nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 24))
        self.norm2 = nn.LayerNorm(24)
        self.classifier = nn.Linear(24, 2)

    def forward(self, x):
        h = self.project(x) + self.position
        batch, steps, _ = h.shape
        q, k, v = self.qkv(h).reshape(batch, steps, 3, 4, 6).permute(2, 0, 3, 1, 4)
        weights = torch.softmax(q @ k.transpose(-2, -1) / (6 ** 0.5), dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, steps, 24)
        h = self.norm1(h + self.combine(context))
        return self.classifier(self.norm2(h + self.ff(h)).mean(dim=1))

model = TinyTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
accuracies = []
for epoch in range(14):
    model.train()
    for indices in torch.randperm(len(train_y)).split(32):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(train_x[indices].to(DEVICE)), train_y[indices].to(DEVICE))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        correct = (model(val_x.to(DEVICE)).argmax(dim=1) == val_y.to(DEVICE)).float().mean()
    accuracies.append(correct.item())
    print(f"Epoch {epoch + 1:02d}: validation accuracy={accuracies[-1]:.3f}")

with torch.no_grad():
    final_accuracy = (model(test_x.to(DEVICE)).argmax(dim=1) == test_y.to(DEVICE)).float().mean().item()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
image = axes[0].imshow(weights, vmin=0, vmax=0.3, cmap="viridis")
fig.colorbar(image, ax=axes[0])
axes[0].set(xlabel="Key", ylabel="Query", title="Attention weights")
axes[1].plot(range(1, len(accuracies) + 1), accuracies)
axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05),
            title="Synthetic trend classification")
fig.savefig(OUTPUT / "04_attention_transformer.png", dpi=150)
plt.close(fig)
print(f"Attention row-sum: {weights.sum(axis=1)[0]:.3f}; test accuracy: {final_accuracy:.3f}")

record("04_transformer_attention", {"test_accuracy": float(final_accuracy),
                                    "validation_best_accuracy": float(max(accuracies)),
                                    "attention_row_sum": float(weights.sum(axis=1)[0])}, selected)
