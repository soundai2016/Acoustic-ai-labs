"""Backend registration tests without requiring accelerator hardware."""

import types
import unittest
from unittest.mock import patch

class BackendTests(unittest.TestCase):
    def test_real_cpu_preflight(self):
        from cpu.backend import probe
        device, info = probe()
        self.assertEqual(str(device), "cpu")
        self.assertGreater(info["probe_loss"], 0)

    def test_real_gpu_preflight_reports_cuda_absence(self):
        import torch
        from gpu.backend import probe
        if torch.cuda.is_available():
            self.skipTest("CUDA hardware is available; use the strict GPU integration test")
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            probe()

    def test_mocked_npu_backend(self):
        from npu.backend import probe
        cfg = types.SimpleNamespace(allow_internal_format=False)
        npu = types.SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            set_device=lambda i: None, config=cfg,
            set_compile_mode=lambda **kw: None,
            synchronize=lambda: None,
        )
        torch = types.SimpleNamespace(npu=npu, __version__="mock", device=lambda s: s)
        torch_npu = types.SimpleNamespace(__version__="mock-npu")
        with patch.dict("sys.modules", {"torch": torch, "torch_npu": torch_npu}):
            with patch("acoustic_labs.probe.training_probe", return_value=0.7):
                device, info = probe()
        self.assertEqual(device, "npu:0")
        self.assertEqual(info["torch_npu"], "mock-npu")
        self.assertTrue(cfg.allow_internal_format)

    def test_mocked_npu_without_available_hardware_fails(self):
        from npu.backend import probe
        npu = types.SimpleNamespace(is_available=lambda: False)
        torch = types.SimpleNamespace(npu=npu, __version__="mock")
        with patch.dict("sys.modules", {"torch": torch, "torch_npu": types.SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "not available"):
                probe()

    def test_npu_missing_extension_is_reported(self):
        from npu.backend import probe
        import importlib
        original = importlib.import_module

        def missing(name):
            if name == "torch_npu":
                raise ModuleNotFoundError("missing torch_npu")
            return original(name)

        with patch("npu.backend.importlib.import_module", side_effect=missing):
            with self.assertRaisesRegex(ModuleNotFoundError, "missing torch_npu"):
                probe()

if __name__ == "__main__":
    unittest.main()
