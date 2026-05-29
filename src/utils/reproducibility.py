"""Centralized reproducibility helpers.

All experiment scripts should call :func:`set_global_seed` near the top so
results are reproducible across runs on the same hardware/driver stack.

Determinism guarantees we enforce:
  * Python ``random`` and ``hash`` seeding (the latter via ``PYTHONHASHSEED``
    if the interpreter respects it; for full effect the env var must be set
    before Python launches — :mod:`run_all.py` injects it).
  * NumPy and Torch (CPU + CUDA) RNG seeding.
  * cuDNN deterministic kernels (``deterministic=True``, ``benchmark=False``).
  * cuBLAS workspace pinned via ``CUBLAS_WORKSPACE_CONFIG`` so deterministic
    matmul kernels are selected on Ampere+ GPUs.
  * Optional strict mode via :func:`enable_deterministic_algorithms` which
    sets ``torch.use_deterministic_algorithms(True)``.

Bit-exact reproducibility across CUDA driver versions is NOT guaranteed, but
the seeds + flags below remove all known sources of run-to-run noise on a
fixed environment.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed every RNG we know about and return the seed used.

    Args:
        seed: Integer seed (default 42, our project-wide convention).
        deterministic: When True, also enable cuDNN deterministic kernels
            and pin the cuBLAS workspace. Disable only if you need the last
            few percent of throughput and accept run-to-run variance.

    Returns:
        The seed actually applied (useful when callers want to log it).
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        # Required for deterministic cuBLAS matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def enable_deterministic_algorithms(warn_only: bool = True) -> None:
    """Strict mode: error/warn on any non-deterministic op.

    Off by default because some ops (e.g. interpolation, scatter_add on CUDA)
    have no deterministic implementation and would crash training.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def make_dataloader_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Return a seeded :class:`torch.Generator` for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def worker_init_fn(worker_id: int, base_seed: int = DEFAULT_SEED) -> None:
    """DataLoader worker init: seed numpy/random per-worker deterministically."""
    s = int(base_seed) + int(worker_id)
    np.random.seed(s)
    random.seed(s)
