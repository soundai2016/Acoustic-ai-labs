"""Write machine-readable metrics without model weights or raw audio data."""

import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    destination = Path(os.getenv("ACOUSTIC_LABS_OUTPUT_DIR", str(ROOT / "outputs")))
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def record(lab: str, metrics: dict, selected=None) -> None:
    requested = os.getenv("ACOUSTIC_LABS_BACKEND", "cpu")
    device = (selected.metadata() if selected is not None else {
        "requested": requested, "selected": "cpu", "device": "cpu",
        "fallback": False, "fallback_reason": None,
        "details": {"scope": "NumPy/SciPy CPU computation"},
    })
    result = {
        "lab": lab,
        "device": device,
        "metrics": metrics,
        "python": platform.python_version(),
    }
    path = output_dir() / f"{lab}.json"
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    print("RESULT " + json.dumps({"lab": lab, "device": device["selected"], "metrics": metrics}, sort_keys=True))
