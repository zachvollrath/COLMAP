"""Process-wide PyTorch device used by the reconstruction pipeline.

`cpu` and `cuda` are explicit. `auto` uses CUDA when this PyTorch build can see
a GPU, and CPU otherwise. Factory functions (`torch.zeros`, `torch.tensor`, ...)
follow the choice, and RNGs are created on the same device.
"""

from __future__ import annotations

import torch

_device = torch.device("cpu")


def get_device() -> torch.device:
    return _device


def configure_device(choice: str) -> torch.device:
    """Select `auto`, `cpu`, or `cuda` and make new tensors use that device."""
    global _device
    name = choice.strip().lower()
    if name == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif name == "cpu":
        _device = torch.device("cpu")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build cannot see a GPU.")
        _device = torch.device("cuda")
    else:
        raise ValueError(f"Unknown device {choice!r}. Choose auto, cpu, or cuda.")
    torch.set_default_device(_device)
    return _device


def generator(seed: int | None = None) -> torch.Generator:
    """Random generator on the configured device."""
    gen = torch.Generator(device=_device)
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen
