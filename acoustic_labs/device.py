"""Select a verified PyTorch device before any training begins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import warnings


@dataclass(frozen=True)
class Selection:
    requested: str
    backend: str
    device: object
    fallback_reason: str | None = None
    details: dict | None = None

    def metadata(self) -> dict:
        return {
            "requested": self.requested,
            "selected": self.backend,
            "device": str(self.device),
            "fallback": self.requested != self.backend and self.requested != "auto",
            "fallback_reason": self.fallback_reason,
            "details": self.details or {},
        }


def _probe(kind: str):
    module = importlib.import_module(f"{kind}.backend")
    return module.probe()


def select_device(requested: str | None = None, strict: bool | None = None) -> Selection:
    """Fall back only during device preflight, never after training starts."""
    requested = (requested or os.getenv("ACOUSTIC_LABS_BACKEND", "cpu")).lower()
    if requested not in {"cpu", "gpu", "npu", "auto"}:
        raise ValueError(f"Unknown backend: {requested}")
    if strict is None:
        strict = os.getenv("ACOUSTIC_LABS_STRICT_DEVICE", "0") == "1"
    if strict and requested == "auto":
        raise ValueError("Strict mode requires an explicit backend")
    candidates = ["gpu", "npu", "cpu"] if requested == "auto" else [requested]
    failures = []
    for kind in candidates:
        try:
            device, details = _probe(kind)
            return Selection(requested, kind, device,
                             "; ".join(failures) or None, details)
        except Exception as exc:
            if kind == "cpu":
                raise RuntimeError(f"CPU preflight failed: {type(exc).__name__}: {exc}") from exc
            message = f"{kind}: {type(exc).__name__}: {exc}"
            failures.append(message)
            if strict:
                raise RuntimeError(f"Strict device preflight failed: {message}") from exc
    try:
        device, details = _probe("cpu")
    except Exception as exc:
        raise RuntimeError(f"CPU fallback preflight failed: {exc}") from exc
    reason = "; ".join(failures)
    warnings.warn(f"Accelerator unavailable; using CPU. {reason}", RuntimeWarning, stacklevel=2)
    return Selection(requested, "cpu", device, reason, details)


def seed_torch(torch, selected: Selection, seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
    if selected.backend == "gpu":
        torch.cuda.manual_seed_all(seed)
    elif selected.backend == "npu":
        torch.npu.manual_seed_all(seed)
