# Copyright (c) 2026 SoundAI Technology
# SPDX-License-Identifier: MIT
"""Small dependency-light checks of the physical and numerical invariants."""

import unittest
import numpy as np

class AcousticPropertyTests(unittest.TestCase):
    def test_cosine_aliasing_at_eight_khz(self):
        t = np.arange(80) / 8000
        difference = np.cos(2 * np.pi * 6000 * t) - np.cos(2 * np.pi * 2000 * t)
        self.assertLess(np.max(np.abs(difference)), 1e-10)

    def test_fft_frequency_spacing(self):
        fs, n = 8000, 16000
        frequencies = np.fft.rfftfreq(n, d=1 / fs)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 0.5)

    def test_wave_analytic_solution_obeys_boundary_conditions(self):
        x = np.linspace(0, 1, 65)
        p = np.sin(np.pi * x) * np.cos(np.pi * 0.3)
        self.assertAlmostEqual(p[0], 0, places=12)
        self.assertAlmostEqual(p[-1], 0, places=12)

    def test_attention_softmax_normalization(self):
        q = np.array([[1., 0.], [0., 1.], [1., 1.]])
        score = q @ q.T / np.sqrt(2)
        weights = np.exp(score - score.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(3))

if __name__ == "__main__":
    unittest.main()
