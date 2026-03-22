"""Detect CuMesh / CUDA failures that are safe to skip for best-effort mesh ops (e.g. fill_holes)."""

from __future__ import annotations

import torch


def skippable_cumesh_fill_holes_error(exc: BaseException) -> bool:
    """OOM and common launch/config CUDA errors from CuMesh — skip hole fill, keep prior topology."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)):
        return True
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    if not isinstance(exc, RuntimeError):
        return False
    if "[cumesh]" not in msg:
        return False
    # cudaErrorInvalidConfiguration (9), launch bounds, etc.
    if "invalid configuration argument" in msg:
        return True
    if "error code: 9" in msg:
        return True
    if "too many resources requested for launch" in msg:
        return True
    return False
