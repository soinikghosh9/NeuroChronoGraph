"""
Clinical concept extractor: named, auditable quantities a neurophysiologist already reads.

This is the bottleneck of v6. Every concept has a defined estimator and a clinical referent, so
a decision built on them decomposes exactly and a reader can check each value against their own
reading of the trace.

Design constraints, all inherited from measured results in this project:

* **Gain invariance by construction.** Every concept is a ratio, a frequency, a slope or a rank.
  None changes if the recording is multiplied by a constant. Augmentation-based invariance was
  tested here and failed (spectral tilt randomisation drove fold 0 from 0.487 to 0.335).
* **Subject-level.** Concepts aggregate over a subject's windows before classification, so the
  model trains on the unit it is evaluated on. Window-level training with the label copied is
  pseudo-replication: 896 subjects, not 62,000 independent samples.
* **Region membership by coordinate**, not by channel index, so an arbitrary montage maps in and
  a missing electrode is dropped rather than silently zero-filled.

The 17 concepts, and what each corresponds to in a clinical report:

    pdr_frequency        posterior dominant rhythm, the primary visual item
    pdr_prominence       "well-formed" vs "poorly formed" PDR
    pdr_variability      fluctuation of dominant frequency over the recording
    posterior_maximality alpha should be maximal posteriorly; loss is abnormal
    ap_gradient_alpha    anterior-posterior gradient, cardinal sign of diffuse dysfunction
    ap_gradient_freq     the same gradient measured on peak frequency
    asymmetry_alpha      left-right difference; focal rather than degenerative if large
    asymmetry_freq       left-right peak-frequency difference
    theta_alpha_ratio    slowing
    slowing_ratio        (delta+theta)/(alpha+beta), graded severity
    rel_alpha            reduced alpha
    rel_theta            excess theta
    aperiodic_exponent   non-oscillatory background slope, fitted 2-20 Hz
    spectral_edge        spectral MEDIAN frequency over 1-20 Hz
    spectral_entropy     spectral flattening; DECLARED, not hidden -- see below
    artefact_proxy       relative gamma power, an EMG indicator; DECLARED, not hidden
    log_n_windows        usable recording length; DECLARED, not hidden

Three of these were measured against muscle contamination and two were repaired at source rather
than corrected statistically: the aperiodic slope is no longer fitted through the EMG band, and
the edge is a median rather than a 95th percentile. Spectral entropy could not be repaired. It
asks how flat the spectrum is, and broadband noise inside any analysis band genuinely flattens
that band, so the quantity is partly a measure of muscle tone by construction rather than by an
implementation error. It is therefore computed and shown but excluded from the classifier, on the
same footing as the artefact proxy: a concept that cannot be given an honest clinical name should
not be allowed to drive a clinical decision.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.models.v5.montage import resolve_coords
from src.models.v5.spectral_nodes import SpectralNodeFeatures

#: Index into SpectralNodeFeatures output (12 per channel).
F_DELTA, F_THETA, F_ALPHA, F_BETA, F_GAMMA = 0, 1, 2, 3, 4
F_IAPF, F_PROM, F_EDGE, F_APER, F_TAR, F_SLOW, F_ENT = 5, 6, 7, 8, 9, 10, 11

CONCEPT_NAMES: List[str] = [
    "pdr_frequency", "pdr_prominence", "pdr_variability", "posterior_maximality",
    "ap_gradient_alpha", "ap_gradient_freq", "asymmetry_alpha", "asymmetry_freq",
    "theta_alpha_ratio", "slowing_ratio", "rel_alpha", "rel_theta",
    "aperiodic_exponent", "spectral_edge", "spectral_entropy",
    "artefact_proxy", "log_n_windows",
]

#: The amplitude-sensitive counterparts of the three concepts named after inter-electrode
#: amplitude. Deliberately a SEPARATE list: adding them to CONCEPT_NAMES would silently change the
#: width of every cached feature array and invalidate every result computed against it.
AMPLITUDE_CONCEPT_NAMES: List[str] = [
    "amp_posterior_maximality", "amp_ap_gradient_alpha", "amp_asymmetry_alpha",
]

#: Normal ranges for the report. Indicative adult resting eyes-closed values from the clinical
#: literature, used for FLAGGING ONLY -- never in the classifier, which is fitted from data.
NORMAL_RANGES: Dict[str, tuple] = {
    "pdr_frequency": (8.5, 13.0),
    "theta_alpha_ratio": (0.0, 0.8),
    "ap_gradient_alpha": (0.05, 1.0),
    "aperiodic_exponent": (0.8, 2.2),
    "asymmetry_alpha": (0.0, 0.35),
    "artefact_proxy": (0.0, 0.30),
}


def region_masks(coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Boolean channel masks by anatomical position. Coordinate-driven, so any montage works.

    Axes: x left-right, y posterior-anterior, z inferior-superior.
    """
    x, y = coords[:, 0], coords[:, 1]
    return {
        "anterior": y >= 0.5,
        "posterior": y <= -0.4,
        "occipital": y <= -0.85,
        "left": x < -0.05,
        "right": x > 0.05,
    }


def _masked_mean(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean over the channel axis restricted to `m`; NaN-safe if the region is empty."""
    if m.sum() == 0:
        return torch.full(v.shape[:-1], float("nan"), device=v.device, dtype=v.dtype)
    return v[..., m].mean(dim=-1)


class ConceptExtractor:
    """Raw windows -> per-window concepts -> per-subject concept vector."""

    #: Upper edge of the electromyographically clean analysis band, in Hz.
    #:
    #: Chosen on physiological grounds and fixed before any outcome was examined: surface EMG in
    #: scalp EEG is broadband and rises steeply above roughly 20 Hz, so any descriptor integrating
    #: power above that limit is partly a measure of muscle tone. This is not a tuned parameter --
    #: it was not selected by comparing blind-site performance, and must not be.
    #:
    #: It matters because three concepts previously read the contaminated range: the aperiodic
    #: slope was fitted over 2-40 Hz, spectral entropy over the full spectrum, and the spectral
    #: edge over 1-30 Hz. Measured on this corpus they correlated -0.737, +0.708 and +0.678 with
    #: the muscle-artefact proxy, in every case more strongly than with MMSE, while carrying 59.9
    #: per cent of the screening head's coefficient weight.
    EMG_FREE_HZ: float = 20.0

    def __init__(self, sfreq: float = 256.0, channel_names: Optional[Sequence[str]] = None,
                 device: str = "cpu", emg_free: bool = True,
                 emg_free_hz: Optional[float] = None,
                 coord_table: Optional[Dict[str, tuple]] = None):
        # `emg_free_hz` overrides the class constant so the ceiling can be swept. The default
        # remains 20 Hz, fixed a priori from the electromyographic literature; the override exists
        # to demonstrate that the construct-validity verdicts do not balance on that exact value,
        # not to allow it to be tuned. Anything selected by comparing blind-site performance would
        # be exactly the failure this project exists to document.
        hi = float(emg_free_hz) if emg_free_hz else self.EMG_FREE_HZ
        self.emg_free_hz = hi
        # `edge_fraction=0.5` makes this the spectral MEDIAN frequency, not the 95th percentile.
        # A 95th percentile sits inside the contaminated tail by construction; a median does not.
        # Verified on synthetic signals with known ground truth, before any corpus was touched:
        # 3 x EMG moves SEF95(1-30) by +17.4 Hz and SEF50(1-20) by +0.03 Hz, while a slowed
        # rhythm moves SEF50 further than SEF95 (5.0 Hz against 3.5 Hz). Median frequency is
        # also a standard clinical descriptor, so this replaces an ad hoc choice with a
        # conventional one.
        band = dict(fit_range=(2.0, hi), edge_band=(1.0, hi), entropy_band=(1.0, hi),
                    edge_fraction=0.5) if emg_free else {}
        self.spectral = SpectralNodeFeatures(sfreq=sfreq, normalise=False, **band).to(device)
        self.emg_free = emg_free
        # `coord_table` lets a montage outside the 10-20 table be scored -- a wearable array,
        # for instance -- without editing the shared table. Default None keeps the 10-20
        # lookup and every existing result unchanged.
        coords, mask = resolve_coords(list(channel_names) if channel_names else
                                      ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                                       "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
                                       "Fz", "Cz", "Pz"],
                                      coord_table=coord_table)
        self.coords = coords.to(device)
        self.present = mask.to(device)
        self.regions = {k: (v & mask).to(device) for k, v in region_masks(coords).items()}
        self.device = device

    @torch.no_grad()
    def amplitude_concepts(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T]`` -> ``[B, 3]``, the three constructs relative power cannot express.

        `posterior_maximality`, `ap_gradient_alpha` and `asymmetry_alpha` are named after
        statements about inter-electrode amplitude -- alpha is largest occipitally, the
        anterior-posterior gradient is lost, interhemispheric amplitude asymmetry exceeds 35 to 50
        per cent. All three are computed from relative band power, a within-channel ratio, and are
        therefore exactly invariant to any per-channel rescaling. They cannot measure what they are
        named after, whatever they happen to predict.

        These are the same three constructs computed from ABSOLUTE band power. Each is a ratio
        between electrodes, so a recording-level gain still cancels -- which is wanted, since
        cohorts differ in overall calibration -- while the inter-electrode contrast survives.

        This is only a valid measurement when the upstream scaling preserved inter-channel
        amplitude. Under per-channel scaling every channel is divided by its own interquartile
        range and these quantities are destroyed before they are computed; extract under global
        scaling. The negative control is to run it under per-channel scaling and confirm the
        signal is gone.
        """
        P = self.spectral.absolute_band_power(x.to(self.device))       # [B, C, 5]
        a = P[..., F_ALPHA]
        R, eps = self.regions, 1e-20
        occ = _masked_mean(a, R["occipital"])
        allc = _masked_mean(a, self.present)
        post = _masked_mean(a, R["posterior"])
        ant = _masked_mean(a, R["anterior"])
        left = _masked_mean(a, R["left"])
        right = _masked_mean(a, R["right"])
        return torch.stack([
            torch.log10((occ + eps) / (allc + eps)),      # alpha maximal occipitally
            torch.log10((post + eps) / (ant + eps)),      # anterior-posterior gradient
            (left - right).abs() / (left + right + eps),  # interhemispheric amplitude asymmetry
        ], dim=-1)

    @torch.no_grad()
    def window_concepts(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T]`` -> ``[B, 15]`` (all concepts except variability and n_windows,
        which are defined only across a subject's windows)."""
        f = self.spectral(x.to(self.device))                    # [B, C, 12]
        R = self.regions

        alpha = f[..., F_ALPHA]
        iapf = f[..., F_IAPF]

        post_alpha = _masked_mean(alpha, R["posterior"])
        ant_alpha = _masked_mean(alpha, R["anterior"])
        post_iapf = _masked_mean(iapf, R["posterior"])
        ant_iapf = _masked_mean(iapf, R["anterior"])
        l_alpha = _masked_mean(alpha, R["left"])
        r_alpha = _masked_mean(alpha, R["right"])
        l_iapf = _masked_mean(iapf, R["left"])
        r_iapf = _masked_mean(iapf, R["right"])

        # PDR is read posteriorly, so its frequency and prominence come from posterior channels.
        pdr_freq = post_iapf
        pdr_prom = _masked_mean(f[..., F_PROM], R["posterior"])

        # Posterior maximality: mean rank of occipital channels in alpha power, 1 = strongest.
        # Normal alpha is occipitally maximal; loss of that topography is abnormal.
        order = alpha.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, torch.arange(alpha.shape[-1], device=alpha.device)
                       .expand_as(order))
        occ = R["occipital"]
        post_max = (ranks[..., occ].float().mean(dim=-1) + 1.0 if occ.sum() > 0
                    else torch.full_like(pdr_freq, float("nan")))

        eps = 1e-8
        return torch.stack([
            pdr_freq,
            pdr_prom,
            post_max,
            post_alpha - ant_alpha,
            post_iapf - ant_iapf,
            (l_alpha - r_alpha).abs() / (l_alpha + r_alpha + eps),
            (l_iapf - r_iapf).abs(),
            _masked_mean(f[..., F_TAR], self.present),
            _masked_mean(f[..., F_SLOW], self.present),
            _masked_mean(alpha, self.present),
            _masked_mean(f[..., F_THETA], self.present),
            _masked_mean(f[..., F_APER], self.present),
            _masked_mean(f[..., F_EDGE], self.present),
            _masked_mean(f[..., F_ENT], self.present),
            _masked_mean(f[..., F_GAMMA], self.present),   # artefact proxy (EMG)
        ], dim=-1)


def aggregate_subject(win_concepts: np.ndarray, fill: Optional[np.ndarray] = None,
                      return_flags: bool = False):
    """A subject's window concepts ``[n_win, 15]`` -> one 17-d concept vector.

    The median is used rather than the mean: it is robust to the artefactual windows that
    survive rejection, and one cohort here carries four times the artefact burden in one class.

    **F2 -- missing values are no longer replaced by zero.** The previous implementation ended in
    ``np.nan_to_num(out, nan=0.0)``, so a subject whose posterior region was empty, or whose alpha
    peak was undefined, received ``pdr_frequency = 0 Hz``. Zero is not a neutral value for a
    frequency: under the sign constraint (``pdr_frequency`` decreases with impairment) it is the
    most impaired value the model can see, and it was being produced by a measurement failure
    rather than by pathology. It also contradicted this module's own docstring, which promises
    that a missing electrode is "dropped rather than silently zero-filled".

    A concept that cannot be estimated is now filled from ``fill`` -- the training-set median,
    supplied by the caller -- and the substitution is counted. With ``fill=None`` the NaN is
    preserved so callers that want to handle it themselves can.

    Parameters
    ----------
    fill
        Length-17 vector of replacement values, normally the training-set median. Applied only
        where the estimate is missing.
    return_flags
        If True, also return the per-concept boolean mask of substitutions, so a report can say
        "posterior dominant rhythm could not be measured" instead of printing a number.
    """
    w = np.asarray(win_concepts, dtype=np.float64)
    with warnings.catch_warnings():
        # An all-NaN column is an expected outcome now, not an anomaly: it means the concept
        # could not be estimated for this subject, which is exactly what the caller is told.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(w, axis=0)
        # pdr_variability: dispersion of the dominant frequency over the recording. Clinically the
        # dementia-with-Lewy-bodies fluctuation marker; here it doubles as a within-session
        # instability measure. F5: expressed as a robust coefficient of variation (median absolute
        # deviation over the median) rather than a raw standard deviation, which grew with
        # recording length and was dominated by the 0.5 Hz frequency grid.
        f = w[:, 0][np.isfinite(w[:, 0])]
        if len(f) >= 3 and np.median(f) > 0:
            pdr_var = float(np.median(np.abs(f - np.median(f))) / np.median(f))
        else:
            pdr_var = np.nan

    out = np.empty(len(CONCEPT_NAMES), dtype=np.float64)
    out[0] = med[0]                       # pdr_frequency
    out[1] = med[1]                       # pdr_prominence
    out[2] = pdr_var                      # pdr_variability
    out[3:16] = med[2:15]                 # remaining medians, in CONCEPT_NAMES order
    out[16] = np.log1p(len(w))            # usable recording length, declared

    missing = ~np.isfinite(out)
    if fill is not None and missing.any():
        out[missing] = np.asarray(fill, dtype=np.float64)[missing]
    return (out, missing) if return_flags else out


def impute_from_training(C: np.ndarray, train_mask: np.ndarray):
    """Fill non-finite concept values from the TRAINING-set median. Returns (filled, flags).

    Companion to ``aggregate_subject(fill=None)``. The median comes from the training cohort
    alone, so no information crosses from an evaluation site into the values used to score it --
    the same discipline the model's standardisation already follows.

    BREAKING CHANGE, 2026-08-04: ``aggregate_subject`` previously ended in
    ``np.nan_to_num(..., nan=0.0)`` and therefore never returned a missing value. Scripts written
    against that behaviour (108-112, 119, 120) must now either pass ``fill=`` or route the stacked
    matrix through this function, otherwise NaNs will propagate into the fitted models. New
    extraction code should call this once, immediately after stacking.
    """
    C = np.asarray(C, dtype=np.float64).copy()
    train_mask = np.asarray(train_mask, dtype=bool)
    flags = ~np.isfinite(C)
    if not flags.any():
        return C, flags
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(C[train_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.nonzero(flags)
    C[idx] = med[idx[1]]
    return C, flags
