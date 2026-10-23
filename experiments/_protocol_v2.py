"""
Shared evaluation protocol v2 -- model-agnostic, so NCG and every baseline run identical code.

Fixes six defects carried by the previous protocol (`_clean_arm.py` + `92_clean_protocol.py`).
Each is listed with the evidence, because every one of them silently changed a reported number.

D1. **Fixed-K was applied to TRAINING as well as evaluation.** `92_clean_protocol.py:251` ran
    `fixed_k_indices` over the whole window list before the fold split, taking the corpus from
    106,724 to 59,782 windows and discarding ~42% of CAUEEG's training data. The reason for
    equal window counts is that recording length must not influence the *decision* or make
    sites incomparable -- both are scoring-time concerns. Here: **train on all windows, apply
    fixed-K only at scoring.**

D2. **`ds006036` duplicates `openneuro_ds004504`** -- 88/88 exact subject-id overlap, identical
    class counts. Shared group ids meant no train/test leakage, but double window weight.
    Dropped.

D3. **Single-class cohorts were listed as "blind test sites".** MISP (29 MCI, 0 CN) and P-ADIC
    (28 MCI, 0 CN) cannot yield an AUC. They move to a separate track.

D4. **FTD (n=14 in training, 2 sites) was one of four classes.** Untrainable and unresolvable
    from zero; it made a quarter of every confusion matrix noise. Folded into `dementia`.

D5. **The four-class kappa hid unequal support.** Screening is validatable at 4 sites, staging
    at 2. Reported separately now.

D6. **Metadata leaked a cohort signature** and the masking constant was itself a site tag.
    Retained from the previous protocol: metadata off, missing-sentinel used, T5/T6 zeroed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- data

DATASET_PATHS = {
    "ds004504": ROOT / "datasets" / "openneuro_ds004504",
    # "ds006036" DROPPED: same 88 subjects as ds004504 (D2).
    "Alz_EEG": ROOT / "datasets" / "Alz_EEG_data",
    "Mendeley": ROOT / "datasets" / "Mendeley Dataset",
    "MCI_Dataset": ROOT / "datasets" / "mci dataset",
    "MISP": ROOT / "datasets" / "MISP",
    "P-ADIC": ROOT / "datasets" / "P-ADIC",
    "CAUEEG": ROOT / "datasets" / "caueeg-dataset",
    "GENEEG": ROOT / "datasets" / "GENEEG",
}

TRAIN_COHORT = "CAUEEG"

#: Data-integrity defects found in the loader audit; reported separately, never aggregated.
#:
#: Mendeley added 2026-07-27. Recording duration alone separates its controls from its MCI
#: subjects at AUC 0.810 -- higher than any model result at that site -- and every
#: representation that reads physiology is inverted there: SVM 0.076/0.105, v5 0.076/0.057,
#: logistic 0.381/0.190, across nine model configurations. Only the two tree ensembles score
#: high (0.829, 0.743), and they do so by tracking the duration cue. Its dementia arm behaves
#: the same way at n=28. The site measures acquisition, not pathology.
QUARANTINE = {"AlzEEG", "MCI_Dataset", "Mendeley"}

#: Only MCI subjects -- no negatives, so AUC is undefined and kappa degenerate (D3).
#: Usable for calibration drift, OOD, prevalence shift and reliability only.
SINGLE_CLASS = {"MISP", "P-ADIC"}

# --------------------------------------------------------------------- class scheme

# src.config.config.CLASS_NAMES == ['AD', 'FTD', 'CN', 'MCI'] -- NORMAL IS INDEX 2.
RAW_AD, RAW_FTD, RAW_CN, RAW_MCI = 0, 1, 2, 3

#: Three-class scheme (D4). Dementia absorbs FTD: at screening an FTD patient is correctly
#: impaired and at staging correctly dementia; only the *subtype* claim was unsupported, so
#: those 37 subjects still contribute rather than being thrown away.
CN, MCI, DEM = 0, 1, 2
CLASS3 = ("CN", "MCI", "dementia")

_RAW_TO_3 = {RAW_CN: CN, RAW_MCI: MCI, RAW_AD: DEM, RAW_FTD: DEM}


def to_three_class(labels: np.ndarray) -> np.ndarray:
    out = np.full(len(labels), -1, dtype=np.int64)
    for raw, new in _RAW_TO_3.items():
        out[np.asarray(labels) == raw] = new
    return out


def hierarchical_targets(y3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two binary levels from the 3-class label (D5).

    L1 screening: impaired (MCI or dementia) versus CN -- defined for every subject.
    L2 staging:   dementia versus MCI -- defined only for impaired subjects; CN gets -1 and
                  must be masked out of the L2 loss, not treated as a negative.
    """
    y3 = np.asarray(y3)
    l1 = (y3 != CN).astype(np.int64)
    l2 = np.full(len(y3), -1, dtype=np.int64)
    l2[y3 == MCI] = 0
    l2[y3 == DEM] = 1
    return l1, l2


def cohort_of(subject_id) -> str:
    """Map a subject id to its cohort.

    GENEEG is checked FIRST: its ids look like ``GENEEG_MCI_Y_0001`` and contain ``MCI_``,
    so the generic rule would silently merge them into MCI_Dataset while ``GENEEG_CN_*``
    fell through to "unknown" and was dropped -- corrupting two cohorts at once.
    """
    s = str(subject_id)
    if "GENEEG" in s:
        return "GENEEG"
    for key in ("CAUEEG", "MISP", "AlzEEG", "Mendeley"):
        if key in s:
            return key
    if "PADIC" in s or "P-ADIC" in s:
        return "P-ADIC"
    if "MCIData" in s or "MCI_" in s:
        return "MCI_Dataset"
    if ("ds004504" in s or "ds006036" in s
            or s.lower().startswith("sub-") or s.lower().startswith("sub_")):
        return "OpenNeuro"
    return "unknown"


def evaluable_sites(cohorts: Sequence[str]) -> List[str]:
    """Blind sites that can actually produce a discrimination metric (D3)."""
    return sorted({c for c in cohorts
                   if c not in (TRAIN_COHORT, "unknown")
                   and c not in QUARANTINE and c not in SINGLE_CLASS})


# ------------------------------------------------------------------ window selection

SCORE_K = 40          # windows per subject at SCORING time only
DROP_CHANNELS = [14, 15]      # T5, T6 -- GENEEG supplies only 17 of 19 and zero-fills these
MISSING_METADATA = [0.5, 0.5, -1.0]   # loaders' own sentinel; NOT zeros (mmse 0 == "Moderate+")


def augmented_mask(dataset) -> np.ndarray:
    """Per-window ``is_augmented`` flags, read from the wrappers' ``samples`` lists.

    D7. **Synthetic windows were being evaluated as if they were real.**
    ``DATASET_CONFIG['enable_augmentation']`` is True globally and the minority augmenter
    duplicates windows *inside each cohort* -- GENEEG's MCI goes 1400 -> 2800, so half that
    blind cohort's MCI windows are copies with amplitude scaling and added noise. Neither
    ``92_clean_protocol.py`` nor ``_clean_arm.py`` mentions ``is_augmented``, so every blind
    metric from that protocol was computed partly on synthetic signal. Several older
    experiments (39, 40, 44, 46, 48, 52) do filter it, so the facility exists and was simply
    not used here.

    Read from ``samples`` rather than ``__getitem__`` because the latter loads and filters
    the signal for every window.
    """
    subs = getattr(dataset, "datasets", None) or [dataset]
    flags: List[bool] = []
    for w in subs:
        s = getattr(w, "samples", None)
        if s is None:
            raise RuntimeError(f"{type(w).__name__} exposes no `samples`; cannot identify "
                               f"augmented windows. Refusing to score on possibly synthetic data.")
        flags.extend(bool(rec[2]) if len(rec) > 2 else False for rec in s)
    return np.asarray(flags, dtype=bool)


def scoring_indices(groups: np.ndarray, k: int = SCORE_K, seed: int = 42,
                    is_augmented: Optional[np.ndarray] = None) -> np.ndarray:
    """Deterministically choose at most ``k`` window indices per subject, FOR SCORING ONLY.

    This is the fix for D1. Equalising window counts stops recording length influencing the
    decision and makes sites comparable; it is not a reason to discard training data.
    """
    rng = np.random.default_rng(seed)
    eligible = np.ones(len(groups), bool) if is_augmented is None else ~np.asarray(is_augmented)
    keep: List[np.ndarray] = []
    for g in np.unique(groups):
        idx = np.flatnonzero((groups == g) & eligible)
        if len(idx) == 0:
            continue                       # subject had only synthetic windows
        if len(idx) > k:
            idx = np.sort(rng.choice(idx, k, replace=False))
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def apply_clean_transform(x, meta):
    """Zero the two channels GENEEG lacks, and neutralise metadata (D6).

    Metadata is *zeroed to the loaders' missing sentinel*, not to zeros: the FiLM conditioner
    reads ``mmse >= 0`` as PRESENT and maps mmse=0 to stage "Moderate+", so a zero vector
    tells the model every subject has severe dementia. That bug drove kappa to exactly 0.000
    once already.
    """
    import torch
    x = x.clone()
    x[:, DROP_CHANNELS, :] = 0.0
    if meta is not None:
        meta = torch.as_tensor(MISSING_METADATA, dtype=meta.dtype,
                               device=meta.device).expand_as(meta).clone()
    return x, meta


# ------------------------------------------------------- connectivity + batching

def wpli_from_stream(vec, n_channels: int = 19, n_bands: int = 5):
    """Unpack `window_features.connectivity_stream` into ``[B, n_bands, C, C]``.

    The stream is the flattened upper triangle per band (5 x 171 = 855). Without this the
    wPLI prior cannot be supplied to the model -- and an unsupplied prior is exactly the
    defect that made v2's "wPLI prior is neutral" ablation measure nothing for the entire
    history of this project. Wiring the prior is pointless if it is never passed.
    """
    import torch
    v = torch.as_tensor(vec)
    if v.dim() == 1:
        v = v.unsqueeze(0)
    B = v.shape[0]
    tri = n_channels * (n_channels - 1) // 2
    expected = n_bands * tri
    if v.shape[-1] != expected:
        raise ValueError(f"connectivity stream is {v.shape[-1]}-d, expected {expected} "
                         f"({n_bands} bands x {tri} pairs)")
    v = v.reshape(B, n_bands, tri)
    iu = torch.triu_indices(n_channels, n_channels, offset=1)
    out = torch.zeros(B, n_bands, n_channels, n_channels, dtype=v.dtype, device=v.device)
    out[:, :, iu[0], iu[1]] = v
    return out + out.transpose(-1, -2)          # wPLI is symmetric


def recording_durations(dataset, sfreq: float = 256.0) -> Dict[str, float]:
    """Recording length in seconds per subject, from the wrappers' ``data_store``.

    **Window count is NOT a usable duration proxy in this pipeline.**
    ``max_windows_per_subject = 50`` caps every subject, so the real-window count is constant
    and any AUC computed from it is exactly 0.500 by construction -- which is precisely what
    the first shortcut-battery dry-run returned at three of four sites. A saturated proxy
    reports "no shortcut" no matter how strong the shortcut is, which is worse than not
    measuring it.

    Recording duration is the quantity the earlier audits actually found separates classes
    (CAUEEG CN recordings ~33% shorter), and it survives the cap.
    """
    subs = getattr(dataset, "datasets", None) or [dataset]
    out: Dict[str, float] = {}
    skipped = 0
    for w in subs:
        store = getattr(w, "data_store", None) or {}
        fs = float(getattr(w, "sfreq", sfreq) or sfreq)
        for sid, rec in store.items():
            # Under NCG_MEMMAP=1 the store holds {'path': ..., 'shape': (...)} rather than the
            # array itself (dataset_wrapper.py:206). Reading `.shape` off the dict is both
            # correct and cheaper -- it never touches the NVMe.
            try:
                if isinstance(rec, dict):
                    n_times = int(rec["shape"][-1])
                else:
                    n_times = int(np.asarray(rec).shape[-1])
                out[str(sid)] = n_times / fs
            except Exception:                                          # noqa: BLE001
                skipped += 1
                continue
    if not out:
        raise RuntimeError(
            f"no usable recording shapes found (skipped {skipped}). Cannot measure duration, "
            f"and window counts are capped by max_windows_per_subject so they cannot "
            f"substitute -- a window-count battery returns exactly 0.500 by construction.")
    return out


class SubjectBatchSampler:
    """Yield batches made of a few subjects x many windows each.

    Required by the split-half consistency loss, which needs several windows *from the same
    subject* inside one batch. With ordinary shuffling at batch size 8 nearly every subject
    contributes a single window, `split_half_posteriors` skips everything below
    ``min_windows`` and the loss silently evaluates to zero for the whole run -- present in
    the code, contributing nothing, which is the failure pattern this project keeps hitting.

    ``cycles`` controls epoch coverage. Each pass over the subject list shows only
    ``windows_per_subject`` windows per subject, so a single pass covers roughly
    ``windows_per_subject / mean_windows_per_subject`` of the data -- about **9%** for
    CAUEEG, which would turn a nominal 30 epochs into ~2.8 effective passes and badly
    undertrain the model. Left at ``None`` it is derived so one epoch covers the data once.
    """

    def __init__(self, groups, indices, subjects_per_batch: int = 2,
                 windows_per_subject: int = 8, seed: int = 42, drop_last: bool = True,
                 cycles: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.spb = subjects_per_batch
        self.wps = windows_per_subject
        self.drop_last = drop_last
        idx = np.asarray(indices)
        g = np.asarray(groups)[idx]
        self.by_subject = {s: idx[g == s] for s in np.unique(g)}
        self.subjects = list(self.by_subject)

        if cycles is None:
            mean_w = float(np.mean([len(v) for v in self.by_subject.values()])) or 1.0
            cycles = max(int(round(mean_w / max(self.wps, 1))), 1)
        self.cycles = cycles

    def __iter__(self):
        for _ in range(self.cycles):
            order = self.rng.permutation(len(self.subjects))
            stop = len(order) - (self.spb - 1 if self.drop_last else 0)
            for i in range(0, max(stop, 0), self.spb):
                batch = []
                for j in order[i:i + self.spb]:
                    pool = self.by_subject[self.subjects[j]]
                    take = min(self.wps, len(pool))
                    batch.extend(self.rng.choice(pool, take, replace=False).tolist())
                if batch:
                    yield batch

    def __len__(self):
        return max(len(self.subjects) // self.spb, 1) * self.cycles


# ------------------------------------------------------------------------ metrics

def subject_scores(window_scores: np.ndarray, groups: np.ndarray,
                   k: int = SCORE_K, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate window scores to one score per subject over a fixed-K sample."""
    keep = scoring_indices(groups, k, seed)
    ws, gs = np.asarray(window_scores)[keep], np.asarray(groups)[keep]
    subs = np.unique(gs)
    return subs, np.array([ws[gs == s].mean() for s in subs])


def safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    """AUC, or None when it is undefined (one class present) -- rather than a misleading 0.5."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y[y >= 0])) < 2:
        return None
    m = y >= 0
    return float(roc_auc_score(y[m], np.asarray(s)[m]))


def prevalence_matched_threshold(scores: np.ndarray, prevalence: float) -> float:
    """Threshold placing the predicted positive rate at ``prevalence``.

    Needs the target site's scores but NONE of its labels. Recovered ~74% of the
    transported-to-oracle gap across three independent arms (E1), which is why it is part of
    the protocol rather than a post-hoc script.

    **Where `prevalence` comes from decides what the number means.** Callers here pass the
    site's TRUE prevalence (`y.mean()`), which is an *oracle prevalence* -- label-derived. That
    is the best case and must be reported as "known case mix", not as label-free transfer.
    A clinic genuinely does often know its case mix from registry data, so the scenario is
    defensible; presenting it as requiring no labels at all is not. For the label-free claim,
    pass an assumed prevalence (e.g. the source-cohort rate) and report the sensitivity of
    balanced accuracy to getting it wrong.
    """
    q = float(np.clip(1.0 - prevalence, 0.0, 1.0))
    return float(np.quantile(np.asarray(scores), q))


def site_report(y_true: np.ndarray, scores: np.ndarray, prevalence: Optional[float] = None
                ) -> Dict[str, Optional[float]]:
    """AUC plus operating-point behaviour at a single site."""
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y_true)
    out: Dict[str, Optional[float]] = {"n": int((y >= 0).sum()), "auc": safe_auc(y, scores)}
    if out["auc"] is None:
        return out
    m = y >= 0
    yv, sv = y[m], np.asarray(scores)[m]
    out["prevalence"] = float(yv.mean())
    out["bacc_at_0.5"] = float(balanced_accuracy_score(yv, (sv >= 0.5).astype(int)))
    if prevalence is not None:
        tau = prevalence_matched_threshold(sv, prevalence)
        out["bacc_at_matched_tau"] = float(balanced_accuracy_score(yv, (sv >= tau).astype(int)))
        out["matched_tau"] = tau
    # oracle: best achievable threshold, the ceiling the transported one is judged against
    best = max(balanced_accuracy_score(yv, (sv >= t).astype(int))
               for t in np.quantile(sv, np.linspace(0.02, 0.98, 49)))
    out["bacc_oracle"] = float(best)
    return out


#: The discriminations a site can support, as (negative class, positive class).
#: Reporting one pooled "CN vs impaired" number per site is INVALID for cross-site
#: comparison, because the sites do not contain the same classes: OpenNeuro is CN+dementia
#: with no MCI, GENEEG is CN+MCI with no dementia, Mendeley has both. A spread computed
#: across those three mixes task difficulty with site transfer and cannot be interpreted.
#: Every target below is evaluated only at sites holding both of its classes.
TARGETS: Dict[str, Tuple[int, int]] = {
    "CN_vs_dementia": (CN, DEM),
    "CN_vs_MCI": (CN, MCI),
}


def target_subset(y3: np.ndarray, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """Mask and binary labels for one target. Subjects outside the pair are excluded."""
    neg, pos = TARGETS[target]
    y3 = np.asarray(y3)
    m = (y3 == neg) | (y3 == pos)
    return m, (y3[m] == pos).astype(np.int64)


def pairwise_score(P: np.ndarray, target: str) -> np.ndarray:
    """Conditional posterior restricted to the target's two classes.

    ``P(pos) / (P(neg) + P(pos))`` rather than ``1 - P(CN)``. For CN-vs-dementia the pooled
    impairment score is diluted by P(MCI), which is irrelevant to that decision and differs
    systematically across sites -- so the restricted posterior is both better matched to the
    question and less sensitive to a site's class mix.
    """
    neg, pos = TARGETS[target]
    P = np.asarray(P, dtype=float)
    return P[:, pos] / (P[:, neg] + P[:, pos] + 1e-8)


def auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 42
           ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """AUC with a bootstrap 95% CI. Essential here: blind sites have n=35-88, where the CI
    is wide enough that a point estimate alone invites over-reading."""
    y = np.asarray(y)
    a = safe_auc(y, s)
    if a is None or len(y) < 8:
        return a, None, None
    rng = np.random.default_rng(seed)
    s = np.asarray(s)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        b = safe_auc(y[idx], s[idx])
        if b is not None:
            boots.append(b)
    if len(boots) < 100:
        return a, None, None
    return a, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def spread(values: Sequence[Optional[float]]) -> Optional[float]:
    """Max-minus-min across sites -- the headline cross-site number.

    'AUC 0.72 everywhere' is worth more clinically than '0.85 here, 0.28 there', and no
    published EEG dementia model reports this at all.
    """
    v = [x for x in values if x is not None]
    return float(max(v) - min(v)) if len(v) >= 2 else None
