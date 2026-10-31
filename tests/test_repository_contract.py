"""Check the distributable repository and command-line report contract."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_student_repository_has_no_instructor_notes(self):
        self.assertFalse((ROOT / "TEACHING.md").exists())
        self.assertFalse(list(ROOT.rglob("*.ipynb")))
        self.assertEqual(len(list((ROOT / "labs").glob("[0-9][0-9]_*.py"))), 6)

    def test_cpu_preflight_report_is_real_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cpu-preflight.json"
            completed = subprocess.run(
                [sys.executable, "cpu/run.py", "--probe", "--strict-device", "--report", str(path)],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["requested"], "cpu")
            self.assertEqual(report["selected"], "cpu")
            self.assertFalse(report["fallback"])
            self.assertIn("probe_loss", report["details"])

    def test_missing_explicit_cann_script_fails_early(self):
        if sys.platform == "win32":
            self.skipTest("Bash launcher requires Linux")
        import os
        with tempfile.TemporaryDirectory() as temporary:
            env = os.environ.copy()
            env["CANN_ENV_SCRIPT"] = str(Path(temporary) / "absent.sh")
            completed = subprocess.run(
                ["bash", "npu/run.sh", "--probe"], cwd=ROOT,
                env=env, capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not readable", completed.stderr)


if __name__ == "__main__":
    unittest.main()
