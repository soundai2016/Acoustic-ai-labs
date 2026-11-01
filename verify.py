"""Run experiments in fresh subprocesses and validate their artifacts and metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent
LABS = {
    "01_signals_spectra": ("01_signals_spectra.png", "01_synthetic_audio.wav"),
    "02_room_denoise": ("02_room_denoise.png", "02_noisy.wav", "02_denoised.wav"),
    "03_cnn_classification": ("03_cnn_training.png",),
    "04_transformer_attention": ("04_attention_transformer.png",),
    "05_wave_fdtd": ("05_wave_fdtd.png",),
    "06_deeponet_wave": ("06_deeponet_wave.png",),
}
TORCH_LABS = {"03_cnn_classification", "04_transformer_attention", "06_deeponet_wave"}
FAST_LABS = ["01_signals_spectra", "02_room_denoise", "05_wave_fdtd"]


def check_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100:
        raise AssertionError(f"Missing or empty artifact: {path.name}")
    if path.suffix == ".png":
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"Invalid PNG: {path.name}")
    elif path.suffix == ".wav":
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() != 8000 or audio.getnchannels() != 1 or audio.getnframes() < 8000:
                raise AssertionError(f"Invalid WAV parameters: {path.name}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_metrics(lab: str, value: dict) -> None:
    """Check robust teaching invariants, not hardware-specific exact scores."""
    m = value["metrics"]
    import math
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in m.values()):
        raise AssertionError("Nonfinite or nonnumeric metric")
    if lab == "01_signals_spectra":
        require(abs(m["fft_spacing_hz"] - 0.5) < 1e-8 and m["alias_max_error"] < 1e-10, "FFT/aliasing invariant")
    elif lab == "02_room_denoise":
        require(m["snr_gain_db"] > 0 and m["snr_after_db"] > m["snr_before_db"], "SNR must improve")
    elif lab == "03_cnn_classification":
        require(0.60 <= m["shifted_accuracy"] <= 1 and m["test_count"] == 90, "CNN shifted accuracy")
    elif lab == "04_transformer_attention":
        require(0.80 <= m["test_accuracy"] <= 1 and abs(m["attention_row_sum"] - 1) < 1e-6, "Attention evaluation")
    elif lab == "05_wave_fdtd":
        require(0 < m["fine_relative_error"] < m["coarse_relative_error"] < 0.02, "FDTD grid convergence")
    elif lab == "06_deeponet_wave":
        require(0 < m["id_relative_l2"] < 0.20 and m["ood_relative_l2"] > 0, "DeepONet relative errors")
    else:
        raise ValueError(lab)


def execute(lab: str, backend: str, strict: bool, output: Path, timeout: int) -> dict:
    expected = [output / name for name in LABS[lab]]
    result_file = output / f"{lab}.json"
    for path in expected + [result_file]:
        path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({
        "ACOUSTIC_LABS_BACKEND": backend,
        "ACOUSTIC_LABS_STRICT_DEVICE": "1" if strict else "0",
        "ACOUSTIC_LABS_OUTPUT_DIR": str(output.resolve()),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0" if backend == "npu" else env.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "0"),
    })
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"labs.{lab}"], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"lab": lab, "status": "fail", "error": f"Timeout after {timeout}s: {exc}",
                "elapsed_s": round(time.monotonic() - started, 3)}
    row = {"lab": lab, "elapsed_s": round(time.monotonic() - started, 3),
           "return_code": proc.returncode, "stdout_tail": proc.stdout[-1000:],
           "stderr_tail": proc.stderr[-1000:]}
    if proc.returncode != 0:
        return dict(row, status="fail", error="Experiment returned a nonzero exit code")
    try:
        for path in expected:
            check_output(path)
        value = json.loads(result_file.read_text(encoding="utf-8"))
        require(value["lab"] == lab, "Metric file belongs to another lab")
        require(value["device"]["selected"] in {"cpu", "gpu", "npu"}, "Invalid selected backend")
        if lab in TORCH_LABS and strict:
            require(value["device"]["selected"] == backend, "Strict hardware check failed")
        if lab not in TORCH_LABS:
            require(value["device"]["selected"] == "cpu", "NumPy lab must run on CPU")
        validate_metrics(lab, value)
    except (AssertionError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        return dict(row, status="fail", error=f"Artifact/metric validation: {exc}")
    return dict(row, status="pass", device=value["device"], metrics=value["metrics"],
                artifacts=[p.name for p in expected])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu", "npu", "auto"), default="cpu")
    parser.add_argument("--all", action="store_true", help="Include the three PyTorch training labs")
    parser.add_argument("--lab", choices=sorted(LABS))
    parser.add_argument("--strict-device", action="store_true", help="Do not allow accelerator fallback")
    parser.add_argument("--probe", action="store_true", help="Check PyTorch device, including a train step")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--report", type=Path, help="Write a JSON validation report")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    if args.strict_device and args.backend == "auto":
        parser.error("--strict-device requires cpu, gpu, or npu")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.probe:
        from acoustic_labs.device import select_device
        try:
            chosen = select_device(args.backend, args.strict_device)
            result = {"status": "pass", **chosen.metadata()}
            code = 0
        except Exception as exc:
            result = {"status": "fail", "requested": args.backend,
                      "strict_device": args.strict_device, "error": str(exc)}
            code = 1
        print(json.dumps(result, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return code
    selected = ([args.lab] if args.lab else list(LABS) if args.all else FAST_LABS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab in selected:
        row = execute(lab, args.backend, args.strict_device, args.output_dir, args.timeout)
        rows.append(row)
        print(f"{row['status'].upper()} {lab}: {row.get('metrics', row.get('error'))}", flush=True)
        if row["status"] != "pass":
            print(row.get("stderr_tail", ""), file=sys.stderr)
            break
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_backend": args.backend,
        "strict_device": args.strict_device,
        "selected_labs": selected,
        "passed": sum(r["status"] == "pass" for r in rows),
        "failed": sum(r["status"] != "pass" for r in rows),
        "complete": len(rows) == len(selected),
        "results": rows,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")
    return 0 if report["complete"] and report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
