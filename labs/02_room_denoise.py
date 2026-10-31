# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""Synthetic room response, additive noise and spectral masking."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from acoustic_labs.results import output_dir, record

OUTPUT = output_dir()
FS = 8000
rng = np.random.default_rng(9)
t = np.arange(2 * FS) / FS
source = (0.35 * np.sin(2 * np.pi * (370 * t + 120 * t**2))
          + 0.16 * np.sin(2 * np.pi * 880 * t))
source *= 0.65 + 0.35 * np.sin(2 * np.pi * 1.2 * t) ** 2

h = np.zeros(int(FS * 0.24))
for ms, gain in [(31, 0.58), (68, -0.36), (109, 0.25), (163, 0.13)]:
    h[round(FS * ms / 1000)] = gain
h += 0.013 * np.exp(-np.arange(h.size) / (FS * 0.062)) * rng.normal(size=h.size)
h[0] = 1
clean = signal.fftconvolve(source, h)[:source.size]
noise = 0.30 * rng.normal(size=source.size)
observed = clean + noise

freq, frames, mixture = signal.stft(observed, FS, nperseg=256, noverlap=192)
_, _, noise_spec = signal.stft(noise, FS, nperseg=256, noverlap=192)
noise_power = np.mean(np.abs(noise_spec) ** 2, axis=1, keepdims=True)
mask = np.maximum(0.08, 1 - 1.35 * noise_power / (np.abs(mixture) ** 2 + 1e-9))
_, enhanced = signal.istft(mixture * mask, FS, nperseg=256, noverlap=192)
enhanced = enhanced[:source.size]

def snr_db(reference, estimate):
    error = estimate - reference
    return 10 * np.log10((np.mean(reference**2) + 1e-12)
                         / (np.mean(error**2) + 1e-12))

snr_before = snr_db(clean, observed)
snr_after = snr_db(clean, enhanced)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
axes[0].plot(np.arange(h.size) / FS, h)
axes[0].set(xlabel="Time (s)", title="Synthetic impulse response")
section = slice(1000, 3400)
for values, name in [(clean, "Reference"), (observed, "Observed"),
                     (enhanced, "Enhanced")]:
    axes[1].plot(t[section], values[section], label=name, lw=0.8)
axes[1].legend()
axes[1].set(xlabel="Time (s)", ylabel="Amplitude")
axes[2].bar(["Before", "After"], [snr_before, snr_after])
axes[2].set(ylabel="SNR (dB)", title="Reference: clean reverberant signal")
fig.savefig(OUTPUT / "02_room_denoise.png", dpi=150)
plt.close(fig)
for filename, values in [("02_noisy.wav", observed), ("02_denoised.wav", enhanced)]:
    values = values / max(1, np.max(np.abs(values)))
    wavfile.write(OUTPUT / filename, FS, np.int16(values * 32767))
print(f"SNR: {snr_before:.2f} → {snr_after:.2f} dB; Δ={snr_after - snr_before:+.2f} dB")

record("02_room_denoise", {"snr_before_db": float(snr_before),
                           "snr_after_db": float(snr_after),
                           "snr_gain_db": float(snr_after - snr_before)})
