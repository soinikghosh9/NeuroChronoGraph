"""Shared anchor/evaluation splits for the local operating-point calibration study (217, 221, 224).

A clinic deploying a transferred screener can record k of its own healthy volunteers. Those k
recordings may be used to calibrate (re-threshold, re-reference); the remaining controls and all
cases are used to evaluate. Every method is scored on IDENTICAL splits, generated here from the
sorted control-subject list, so a difference between methods is never a difference of draws.
"""
from __future__ import annotations

import numpy as np

K_VALUES = (5, 10, 15, 20, 25)
N_SPLITS = 200


def anchor_splits(control_subjects, k, n_splits=N_SPLITS):
    """Yield (anchor_subjects, heldout_control_subjects) for split r = 0..n_splits-1."""
    ctrl = np.array(sorted(control_subjects))
    if k >= len(ctrl):
        raise ValueError(f"k={k} leaves no held-out controls (n={len(ctrl)})")
    for r in range(n_splits):
        perm = np.random.default_rng(10_000 * k + r).permutation(len(ctrl))
        yield ctrl[perm[:k]], ctrl[perm[k:]]
