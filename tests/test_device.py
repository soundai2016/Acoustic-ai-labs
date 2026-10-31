"""Device selection contracts, including mocked accelerator availability."""

import unittest
from unittest.mock import patch

from acoustic_labs.device import select_device, Selection

class DeviceSelectionTests(unittest.TestCase):
    def _probe(self, available):
        def stub(kind):
            if kind not in available:
                raise RuntimeError(f"{kind} not available")
            return {"cpu": "cpu", "gpu": "cuda:0", "npu": "npu:0"}[kind], {"mocked": True}
        return stub

    def test_cpu_selection(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            result = select_device("cpu")
        self.assertEqual(result.backend, "cpu")
        self.assertFalse(result.metadata()["fallback"])

    def test_cuda_selection_when_available(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"gpu", "cpu"})):
            result = select_device("gpu")
        self.assertEqual(result.backend, "gpu")
        self.assertEqual(str(result.device), "cuda:0")

    def test_npu_selection_when_available(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"npu", "cpu"})):
            result = select_device("npu")
        self.assertEqual(result.backend, "npu")
        self.assertEqual(str(result.device), "npu:0")

    def test_cuda_falls_back_to_cpu(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            with self.assertWarnsRegex(RuntimeWarning, "using CPU"):
                result = select_device("gpu")
        self.assertEqual(result.backend, "cpu")
        self.assertTrue(result.metadata()["fallback"])
        self.assertIn("gpu not available", result.fallback_reason)

    def test_npu_falls_back_to_cpu(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            with self.assertWarnsRegex(RuntimeWarning, "using CPU"):
                result = select_device("npu")
        self.assertEqual(result.backend, "cpu")
        self.assertTrue(result.metadata()["fallback"])

    def test_strict_gpu_refuses_fallback(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            with self.assertRaisesRegex(RuntimeError, "Strict device preflight failed"):
                select_device("gpu", strict=True)

    def test_strict_npu_refuses_fallback(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            with self.assertRaisesRegex(RuntimeError, "Strict device preflight failed"):
                select_device("npu", strict=True)

    def test_auto_prefers_gpu(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu", "gpu", "npu"})) as fn:
            result = select_device("auto")
        self.assertEqual(result.backend, "gpu")
        fn.assert_called_once_with("gpu")

    def test_auto_next_prefers_npu(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu", "npu"})):
            result = select_device("auto")
        self.assertEqual(result.backend, "npu")
        self.assertIn("gpu", result.fallback_reason)

    def test_auto_then_cpu(self):
        with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
            result = select_device("auto")
        self.assertEqual(result.backend, "cpu")
        self.assertIn("npu", result.fallback_reason)

    def test_strict_auto_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit backend"):
            select_device("auto", strict=True)

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            select_device("tpu")

    def test_cpu_preflight_failure_not_masked(self):
        with patch("acoustic_labs.device._probe", side_effect=RuntimeError("CPU broken")):
            with self.assertRaisesRegex(RuntimeError, "CPU preflight failed"):
                select_device("cpu")

    def test_accelerator_and_cpu_both_fail(self):
        with patch("acoustic_labs.device._probe", side_effect=RuntimeError("both broken")):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback preflight failed"):
                select_device("npu")

    def test_env_backend(self):
        with patch.dict("os.environ", {"ACOUSTIC_LABS_BACKEND": "gpu"}):
            with patch("acoustic_labs.device._probe", side_effect=self._probe({"gpu"})):
                self.assertEqual(select_device().backend, "gpu")

    def test_env_strict(self):
        with patch.dict("os.environ", {"ACOUSTIC_LABS_BACKEND": "gpu",
                                       "ACOUSTIC_LABS_STRICT_DEVICE": "1"}):
            with patch("acoustic_labs.device._probe", side_effect=self._probe({"cpu"})):
                with self.assertRaisesRegex(RuntimeError, "Strict"):
                    select_device()

    def test_metadata_contains_safe_fields(self):
        selection = Selection("npu", "cpu", "cpu", "not installed", {"mocked": True})
        self.assertEqual(selection.metadata()["fallback_reason"], "not installed")
        self.assertEqual(selection.metadata()["device"], "cpu")

if __name__ == "__main__":
    unittest.main()
