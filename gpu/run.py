"""Run acoustic teaching labs with the gpu backend."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify import main


if __name__ == "__main__":
    if any(arg == "--backend" or arg.startswith("--backend=") for arg in sys.argv[1:]):
        raise SystemExit("This entry point has a fixed backend; use verify.py to select another.")
    raise SystemExit(main(["--backend", "gpu", *sys.argv[1:]]))
