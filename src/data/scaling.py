"""Centralised amplitude scaling, so the choice is explicit and switchable.

Until now every loader inlined the same two lines:

    scaler = RobustScaler(quantile_range=(25, 75))
    data_scaled = scaler.fit_transform(data.T).T

`data` is ``[n_channels, n_times]`` and ``data.T`` is ``[n_times, n_channels]``, so
``fit_transform`` scales each COLUMN -- that is, **each channel independently, by its own
interquartile range**. Repeated in nine files, it had three consequences that only became visible
when they were looked for:

1. **Inter-channel amplitude is destroyed at load.** Three concepts are named after amplitude
   topography (`posterior_maximality`, `ap_gradient_alpha`, `asymmetry_alpha`) and cannot measure
   it, because the information is gone before windowing. Verified in
   `experiments/121_topographic_construct.py`: a synthetic posterior/anterior alpha POWER ratio of
   47.6 changes to 33.8 under per-channel gains while every concept moves by less than 1e-6.
2. **A graph network is deprived of the spatial amplitude structure it exists to exploit.**
3. **The decision was undocumented.** The manuscript's Methods describe a different pipeline
   entirely, and no result file recorded which scaling produced it.

This module makes the choice a named, recorded parameter with three modes:

    "channel"  per-channel robust scale. The historical behaviour; reproduces every result
               computed before 2026-08-04.
    "global"   ONE robust scale for the whole recording, from the pooled interquartile range
               across channels. Harmonises overall amplitude between cohorts while PRESERVING
               the ratios between electrodes, which is what the topographic constructs need.
    "none"     no scaling. Used by the harmonisation pipeline, which scales after cleaning.

`set_mode` is deliberately process-global rather than a per-call argument: the loaders are
constructed deep inside `DatasetFactory` and threading a parameter through would touch more code
than it is worth. Every result writer records the active mode through
`src.utils.provenance`, so a file can always be traced back to the scaling that produced it.
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

Mode = Literal["channel", "global", "none"]

#: Default is the historical behaviour so nothing changes until a caller opts in. The environment
#: variable is read once at import so an existing detached run keeps whatever it was launched with.
_MODE: Mode = os.environ.get("NCG_SCALING", "channel")            # type: ignore[assignment]

_VALID = ("channel", "global", "none")


def set_mode(mode: Mode) -> None:
    if mode not in _VALID:
        raise ValueError(f"scaling mode must be one of {_VALID}, got {mode!r}")
    global _MODE
    _MODE = mode


def get_mode() -> Mode:
    return _MODE


def robust_scale(data: np.ndarray, mode: Mode | None = None) -> np.ndarray:
    """Scale ``[n_channels, n_times]`` according to `mode` (default: the process-global mode).

    Non-finite results are replaced by zero, as before -- a flat channel has an interquartile
    range of zero and would otherwise produce NaN. That substitution is preserved rather than
    improved here so that "channel" mode remains bit-comparable with previously cached features.
    """
    m = mode or _MODE
    if m == "none":
        return data
    x = np.asarray(data, dtype=np.float64)
    if m == "channel":
        med = np.median(x, axis=1, keepdims=True)
        q1, q3 = np.percentile(x, [25, 75], axis=1)
        iqr = (q3 - q1)[:, None]
    elif m == "global":
        # One centre and one scale for the whole recording: amplitude is harmonised across
        # cohorts, but the ratio between any two electrodes is left untouched.
        med = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
    else:                                                          # pragma: no cover
        raise ValueError(f"unknown scaling mode {m!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (x - med) / iqr
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
