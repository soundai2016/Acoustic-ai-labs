"""Artifact validation and metric-contract regression tests."""

from pathlib import Path
import tempfile
import unittest
import wave

from verify import LABS, TORCH_LABS, check_output, validate_metrics

class ValidationTests(unittest.TestCase):
    def test_six_labs_without_notebooks(self):
        self.assertEqual(len(LABS), 6)
        self.assertEqual(len(TORCH_LABS), 3)
        base = Path(__file__).resolve().parents[1]
        for lab in LABS:
            self.assertTrue((base / "labs" / f"{lab}.py").is_file())
        self.assertFalse(list(base.rglob("*.ipynb")))
        self.assertFalse((base / "TEACHING.md").exists())

    def test_no_stale_or_empty_artifacts_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.png"
            with self.assertRaises(AssertionError):
                check_output(path)
            path.write_bytes(b"1234" * 100)
            with self.assertRaisesRegex(AssertionError, "Invalid PNG"):
                check_output(path)

    def test_wav_metadata_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(4000)
                wav.writeframes(b"\x00\x00" * 500)
            with self.assertRaisesRegex(AssertionError, "Invalid WAV"):
                check_output(path)

    def test_metrics_reject_nonfinite(self):
        with self.assertRaises(AssertionError):
            validate_metrics("01_signals_spectra", {"metrics": {
                "fft_spacing_hz": float("nan"), "alias_max_error": 1e-14,
            }})

    def test_metrics_reject_bad_alias(self):
        with self.assertRaises(AssertionError):
            validate_metrics("01_signals_spectra", {"metrics": {
                "fft_spacing_hz": 1, "alias_max_error": 1,
            }})

    def test_metrics_reject_nonimproving_denoiser(self):
        with self.assertRaises(AssertionError):
            validate_metrics("02_room_denoise", {"metrics": {
                "snr_before_db": 0, "snr_after_db": -1, "snr_gain_db": -1,
            }})

    def test_metrics_reject_bad_cnn_accuracy(self):
        with self.assertRaises(AssertionError):
            validate_metrics("03_cnn_classification", {"metrics": {
                "shifted_accuracy": 0.3, "test_count": 90,
            }})

    def test_metrics_reject_bad_transformer_accuracy(self):
        with self.assertRaises(AssertionError):
            validate_metrics("04_transformer_attention", {"metrics": {
                "test_accuracy": 0.5, "attention_row_sum": 1,
            }})

    def test_metrics_reject_bad_fdtd_convergence(self):
        with self.assertRaises(AssertionError):
            validate_metrics("05_wave_fdtd", {"metrics": {
                "coarse_relative_error": 0.001, "fine_relative_error": 0.1,
            }})

    def test_metrics_reject_bad_deeponet_error(self):
        with self.assertRaises(AssertionError):
            validate_metrics("06_deeponet_wave", {"metrics": {
                "id_relative_l2": 1, "ood_relative_l2": 0.8,
            }})

class CliTests(unittest.TestCase):
    def test_fixed_backend_cannot_be_overridden(self):
        import subprocess
        import sys
        base = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, str(base / "npu" / "run.py"),
                               "--backend", "cpu", "--probe"],
                              capture_output=True, text=True, cwd=base, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("fixed backend", proc.stderr)

    def test_fast_lab_selection_contains_no_training(self):
        from verify import FAST_LABS
        self.assertTrue(set(FAST_LABS).isdisjoint(TORCH_LABS))

if __name__ == "__main__":
    unittest.main()
