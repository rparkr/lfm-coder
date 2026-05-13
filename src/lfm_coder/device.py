"""Platform/device detection helpers.

Single source of truth for choosing between CUDA, MPS, and CPU backends.
"""

import torch

Device = str  # "cuda" | "mps" | "cpu"

_VALID = ("auto", "cuda", "mps", "cpu")


def detect_device(preference: str = "auto") -> Device:
    """Resolve a device preference to a concrete backend name.

    "auto" picks the best available: cuda > mps > cpu. Explicit values are
    honored if available; otherwise this falls back to cpu.
    """
    if preference not in _VALID:
        raise ValueError(f"Unknown device preference {preference!r}; expected one of {_VALID}")

    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return "cuda"
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def supports_quantization(device: Device) -> bool:
    """Whether bitsandbytes 4-bit quantization is usable on this backend."""
    return device == "cuda"
