# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""One-dimensional wave equation: FDTD, error and CFL stability."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acoustic_labs.results import output_dir, record

OUTPUT = output_dir()
LENGTH, SPEED, FINAL_TIME = 1.0, 1.0, 0.75

def solve(cells, courant, final_time=FINAL_TIME):
    x = np.linspace(0, LENGTH, cells + 1)
    dx = LENGTH / cells
    steps = int(np.ceil(final_time / (courant * dx / SPEED)))
    dt = final_time / steps
    c = SPEED * dt / dx
    now = np.sin(np.pi * x / LENGTH)
    previous = now.copy()
    # Initial time derivative is zero; endpoints have pressure-release BCs.
    now[1:-1] += 0.5 * c**2 * (now[2:] - 2 * now[1:-1] + now[:-2])
    for _ in range(1, steps):
        nxt = np.zeros_like(now)
        nxt[1:-1] = (2 * now[1:-1] - previous[1:-1]
                     + c**2 * (now[2:] - 2 * now[1:-1] + now[:-2]))
        previous, now = now, nxt
    exact = np.sin(np.pi * x / LENGTH) * np.cos(np.pi * SPEED * final_time / LENGTH)
    relative_error = np.linalg.norm(now - exact) / np.linalg.norm(exact)
    return x, now, exact, relative_error

cells = np.array([20, 40, 80, 160])
results = [solve(int(n), courant=0.5) for n in cells]
errors = np.array([item[3] for item in results])
assert errors[-1] < errors[0]

def instability(courant, steps=65):
    x = np.linspace(0, LENGTH, 65)
    now = np.sin(np.pi * x) + 1e-4 * np.sin(31 * np.pi * x)
    now[[0, -1]] = 0
    previous = now.copy()
    amplitude = [np.max(np.abs(now))]
    now[1:-1] += 0.5 * courant**2 * (now[2:] - 2 * now[1:-1] + now[:-2])
    for _ in range(steps):
        nxt = np.zeros_like(now)
        nxt[1:-1] = (2 * now[1:-1] - previous[1:-1]
                     + courant**2 * (now[2:] - 2 * now[1:-1] + now[:-2]))
        previous, now = now, nxt
        amplitude.append(np.max(np.abs(now)))
    return amplitude

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
axes[0].plot(results[-1][0], results[-1][1], label="FDTD")
axes[0].plot(results[-1][0], results[-1][2], "--", label="Analytic")
axes[0].legend()
axes[0].set(xlabel="Position", title="Wave solution")
axes[1].loglog(cells, errors, "o-", label="Error")
axes[1].loglog(cells, errors[0] * (cells[0] / cells)**2, "--", label="N^-2")
axes[1].legend()
axes[1].set(xlabel="Grid cells", title="Convergence")
axes[2].semilogy(instability(0.8), label="C=0.8")
axes[2].semilogy(instability(1.08), label="C=1.08")
axes[2].legend()
axes[2].set(xlabel="Step", title="CFL stability")
fig.savefig(OUTPUT / "05_wave_fdtd.png", dpi=150)
plt.close(fig)
print("Relative errors:", dict(zip(cells.tolist(), errors.round(7).tolist())))

record("05_wave_fdtd", {"coarse_relative_error": float(errors[0]),
                        "fine_relative_error": float(errors[-1]),
                        "convergence_ratio": float(errors[0] / errors[-1]),
                        "unstable_max_amplitude": float(max(instability(1.08)))})
