# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""Sampling, FFT, STFT and log-Mel features from synthetic audio."""

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
FS, SECONDS, MEL_BANDS = 8000, 2.0, 32

rng = np.random.default_rng(13)
t = np.arange(int(FS * SECONDS)) / FS
x = 0.52 * np.sin(2 * np.pi * 440 * t)
x += 0.33 * signal.chirp(t, f0=700, f1=2700, t1=SECONDS)
x += 0.06 * rng.standard_normal(t.size)
x /= max(1, np.max(np.abs(x)))

freq_fft = np.fft.rfftfreq(x.size, 1 / FS)
fft_amp = 2 * np.abs(np.fft.rfft(x)) / x.size
freq, frames, spectrum = signal.stft(
    x, FS, nperseg=256, noverlap=192, boundary=None, padded=False
)
power = np.abs(spectrum) ** 2

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

mel_edges = np.linspace(hz_to_mel(40), hz_to_mel(FS / 2), MEL_BANDS + 2)
hz_edges = mel_to_hz(mel_edges)
filters = np.zeros((MEL_BANDS, freq.size))
for band in range(MEL_BANDS):
    left, center, right = hz_edges[band:band + 3]
    filters[band] = np.maximum(
        0, np.minimum((freq - left) / (center - left),
                      (right - freq) / (right - center))
    )
log_mel = 10 * np.log10(np.maximum(filters @ power, 1e-10))

short_t = np.arange(80) / FS
alias_error = np.max(np.abs(
    np.cos(2 * np.pi * 6000 * short_t)
    - np.cos(2 * np.pi * 2000 * short_t)
))
assert alias_error < 1e-10

fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
axes[0].plot(t[:1500], x[:1500], lw=0.9)
axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Synthetic waveform")
axes[1].plot(freq_fft, fft_amp, lw=0.75)
axes[1].set(xlim=(0, 3500), xlabel="Frequency (Hz)", title="FFT amplitude")
im = axes[2].pcolormesh(frames, np.arange(MEL_BANDS), log_mel,
                        shading="auto", cmap="magma")
axes[2].set(xlabel="Time (s)", ylabel="Mel band", title="Log-Mel power")
fig.colorbar(im, ax=axes[2], label="Relative dB")
fig.savefig(OUTPUT / "01_signals_spectra.png", dpi=150)
plt.close(fig)
wavfile.write(OUTPUT / "01_synthetic_audio.wav", FS, np.int16(x * 32767))
print(f"FFT spacing: {FS / x.size:.2f} Hz; alias error: {alias_error:.2e}")

record("01_signals_spectra", {"fft_spacing_hz": float(FS / x.size),
                              "alias_max_error": float(alias_error),
                              "mel_bands": int(MEL_BANDS)})
