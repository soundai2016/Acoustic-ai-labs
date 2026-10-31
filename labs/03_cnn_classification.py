# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""Train a small CNN to recognize three synthetic acoustic patterns."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from acoustic_labs.results import output_dir, record
from acoustic_labs.device import select_device, seed_torch

OUTPUT = output_dir()
selected = select_device()
DEVICE = selected.device
print(f"Device: requested={selected.requested}, selected={selected.backend}, "
      f"fallback_reason={selected.fallback_reason}")
FS, N = 8000, 4096
CLASSES = ["harmonic", "chirp", "transient"]
times = np.arange(N) / FS
seed_torch(torch, selected, 42)

def synth(label, rng, shifted=False):
    freq = rng.uniform(310, 440) if shifted else rng.uniform(190, 330)
    if label == 0:
        samples = np.sin(2 * np.pi * freq * times)
        samples += 0.4 * np.sin(2 * np.pi * 2 * freq * times)
    elif label == 1:
        samples = signal.chirp(times, f0=freq, f1=4.8 * freq, t1=times[-1])
    else:
        samples = np.zeros(N)
        for start in rng.choice(np.arange(100, N - 100), 5, replace=False):
            samples[start:start + 60] += rng.uniform(0.5, 1) * np.exp(-np.arange(60) / 12)
    samples *= rng.uniform(0.5, 1.2)
    samples += rng.normal(0, rng.uniform(0.16, 0.32) if shifted else
                          rng.uniform(0.08, 0.20), N)
    _, _, stft = signal.stft(samples, FS, nperseg=256, noverlap=192,
                             boundary=None, padded=False)
    log_spec = np.log1p(12 * np.abs(stft[:128]))
    return ((log_spec - log_spec.mean()) / (log_spec.std() + 1e-4)).astype("float32")

def make_dataset(per_class, shifted, seed):
    rng = np.random.default_rng(seed)
    signals = [synth(label, rng, shifted)
               for label in range(3) for _ in range(per_class)]
    labels = np.repeat(np.arange(3), per_class)
    order = rng.permutation(len(labels))
    return (torch.tensor(np.asarray(signals)[order, None]),
            torch.tensor(labels[order], dtype=torch.long))

train_x, train_y = make_dataset(75, False, 11)
test_x, test_y = make_dataset(30, True, 17)

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 3),
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
loss_history = []
for epoch in range(10):
    model.train()
    batch_losses = []
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_batch.to(DEVICE)), y_batch.to(DEVICE))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    loss_history.append(np.mean(batch_losses))
    print(f"Epoch {epoch + 1:02d}: train loss={loss_history[-1]:.4f}")

model.eval()
with torch.no_grad():
    predictions = model(test_x.to(DEVICE)).argmax(dim=1).cpu()
accuracy = (predictions == test_y).float().mean().item()
confusion = np.zeros((3, 3), dtype=int)
for truth, predicted in zip(test_y.tolist(), predictions.tolist()):
    confusion[truth, predicted] += 1

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].plot(loss_history)
axes[0].set(xlabel="Epoch", title="Training cross-entropy")
axes[1].imshow(confusion, cmap="Blues")
axes[1].set(xticks=range(3), yticks=range(3),
            xticklabels=CLASSES, yticklabels=CLASSES,
            xlabel="Predicted", ylabel="True", title="Shifted test")
for row in range(3):
    for col in range(3):
        axes[1].text(col, row, str(confusion[row, col]), ha="center", va="center")
fig.savefig(OUTPUT / "03_cnn_training.png", dpi=150)
plt.close(fig)
print(f"Shifted-condition test accuracy: {accuracy:.3f}")

record("03_cnn_classification", {"shifted_accuracy": float(accuracy),
                                 "train_final_loss": float(loss_history[-1]),
                                 "test_count": int(len(test_y))}, selected)
