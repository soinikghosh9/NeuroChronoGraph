"""One preprocessing pipeline, applied identically to every cohort.

Until now the project applied none of the preprocessing its manuscript describes. The complete
path was: read file, standardise channel names, resample, per-channel robust scale. No band-pass,
no notch, no re-referencing, no ICA (`AUDIT_RUN_RESULTS_2026-08-04.md` §P). Whatever each cohort
had received from its own curators was what the model saw, and those differed enormously:
ds004504 arrived band-passed, ASR-corrected and ICA-cleaned; CAUEEG arrived as raw clinical EDF.
Measured with ICLabel, OpenNeuro entered every published comparison at 23.7 per cent non-brain
variance and CAUEEG at 64.4 per cent.

The single largest component is ocular. The median CAUEEG recording carries **58.5 per cent** of
its variance in eye components, against 0.6 per cent in muscle -- and the ocular share differs by
diagnosis in opposite directions at the two sites (CAUEEG: CN 0.590, dementia 0.438; ds004504 raw:
CN 0.333, dementia 0.404). That is a route to the label containing no disease information, present
in every model trained here.

This module removes it, in a fixed order, with every decision recorded.

    1. detect mains from the spectrum (50 or 60 Hz), notch it and its harmonics
    2. band-pass to the analysis band, zero-phase FIR
    3. common-average reference
    4. ICA (extended infomax) + ICLabel, removing eye, heart, line and channel components
    5. resample to the target rate
    6. leave scaling to `src.data.scaling`, applied at load time

**Muscle components are deliberately KEPT.** Three of the model's named concepts were shown to
track muscle, and that finding is the subject of the paper; removing muscle here would erase the
phenomenon under study. It is also the more defensible clinical choice, because muscle removal
takes genuine high-frequency cortical signal with it.

Two details that matter and are easy to get wrong:

*ICA is fitted on a differently filtered copy.* ICLabel is trained on 1-100 Hz data and its
strongest cues for eye and muscle live above the analysis band's 45 Hz ceiling. So the
decomposition is fitted on a 1 Hz to min(100 Hz, Nyquist-5) copy and the resulting exclusion is
applied to the 0.5-45 Hz recording. Fitting on a high-passed copy and applying to a
lower-high-passed one is the standard MNE workflow, because a 0.5 Hz high-pass leaves drift that
destabilises the decomposition.

*ICA is fitted on a crop and applied to everything.* Extended infomax on 19 channels needs of the
order of 20 x 19^2 = 7,220 samples; a 300 s crop at 256 Hz supplies 76,800. Fitting on the whole
recording would multiply runtime across 1,621 subjects for no benefit, and a fixed-length crop
also stops recording length from influencing the decomposition -- which matters here, because
recording length differs by diagnosis in the training cohort.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Component classes removed. "muscle artifact" is absent by design -- see the module docstring.
REMOVE_DEFAULT: Tuple[str, ...] = ("eye blink", "heart beat", "line noise", "channel noise")
ICLABEL_CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise", "other")

#: Minimum posterior probability for a component to be excluded. ICLabel's argmax alone will
#: discard components it is barely confident about; requiring 0.7 keeps borderline components in
#: the data, which is the conservative direction for a cleaning step.
MIN_PROB = 0.70


def detect_mains(raw, candidates: Sequence[float] = (50.0, 60.0),
                 min_ratio: float = 3.0) -> Optional[float]:
    """Return the mains frequency present in the recording, or None.

    Decided from the data rather than assumed from the country of origin: this corpus spans India,
    Korea, Greece and Brazil, and several cohorts have already been notched by their curators, in
    which case notching again would only add filter ringing. A peak must exceed the local spectral
    median by `min_ratio` to count.
    """
    from scipy.signal import welch

    sf = float(raw.info["sfreq"])
    d = raw.get_data()
    n = min(int(4 * sf), d.shape[1])
    if n < 64:
        return None
    fr, p = welch(d, fs=sf, nperseg=n, axis=-1)
    p = p.mean(axis=0)
    best, best_ratio = None, 0.0
    for f0 in candidates:
        if f0 >= sf / 2 - 2:
            continue
        peak = (fr > f0 - 2) & (fr < f0 + 2)
        local = (fr > f0 - 10) & (fr < f0 + 10)
        if peak.sum() < 1 or local.sum() < 5:
            continue
        ratio = float(p[peak].max() / (np.median(p[local]) + 1e-30))
        if ratio > best_ratio:
            best, best_ratio = f0, ratio
    return best if best_ratio >= min_ratio else None


def harmonise(raw, sfreq_out: float = 256.0, l_freq: float = 0.5, h_freq: float = 45.0,
              ica_hp: float = 1.0, remove: Sequence[str] = REMOVE_DEFAULT,
              ica_crop_s: float = 300.0, seed: int = 42, run_ica: bool = True
              ) -> Tuple[object, Dict]:
    """Apply the full pipeline. Returns (cleaned Raw at `sfreq_out`, QC dict).

    The input must be an unscaled Raw in volts with canonical channel names. QC records every
    decision the function made so that a downstream result can be traced to it.
    """
    import mne

    qc: Dict = {"n_channels_in": len(raw.ch_names), "sfreq_in": float(raw.info["sfreq"]),
                "duration_s": float(raw.times[-1]), "ica_run": False}

    # ---- 0. units ----------------------------------------------------------------------
    # MNE assumes volts. Two loaders deliver microvolts: `padic_loader` builds a RawArray
    # straight from a .mat without converting, and MISP's EDF headers do not declare a usable
    # unit. Measured medians: CAUEEG 4.2e-06, ds004504 3.1e-06, GENEEG 4.4e-06 -- against MISP
    # 2.2 and P-ADIC 3.5, a factor of a million.
    #
    # Per-channel robust scaling hid this completely, because it normalises the error away; it
    # only became visible once windows were rejected against an absolute microvolt threshold.
    # The correction is applied here rather than silently in a loader, and is RECORDED, because
    # an undeclared rescaling is the same class of defect as the one this pipeline exists to fix.
    # The median must be taken over CHANNELS THAT CARRY SIGNAL. A loader that zero-pads a
    # low-density montage up to 19 channels leaves the majority of rows exactly zero, which drags
    # the all-channel median to 0.0 and defeats this test precisely when the recording is most
    # broken -- observed on the Mendeley cohort, where 15 of 19 channels are padding, the median
    # read 0.0, no correction fired, and every window was later rejected against the absolute
    # threshold. The cohort disappeared from the corpus without an error.
    d_all = raw.get_data()
    per_ch = np.median(np.abs(d_all), axis=1)
    live = per_ch > 0
    med_abs = float(np.median(per_ch[live])) if live.any() else 0.0
    qc["median_abs_input"] = med_abs
    qc["n_channels_live"] = int(live.sum())
    if live.sum() < 0.5 * len(per_ch):
        qc["low_density_warning"] = (
            f"only {int(live.sum())} of {len(per_ch)} channels carry signal; the rest are padding. "
            "Regional concepts are computed over anatomical masks and cannot be trusted here.")
    if med_abs > 1e-3:                      # >1 mV median is not physiological for scalp EEG
        raw._data = d_all * 1e-6
        qc["unit_correction"] = 1e-6
        qc["unit_correction_reason"] = (
            f"median |x| over live channels = {med_abs:.3g} implies microvolts, not volts")
    else:
        qc["unit_correction"] = 1.0

    # ---- 1. mains ---------------------------------------------------------------------
    sf = float(raw.info["sfreq"])
    mains = detect_mains(raw)
    qc["mains_hz"] = mains
    if mains:
        freqs = [f for f in np.arange(mains, sf / 2 - 1.0, mains)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.notch_filter(freqs, verbose=False)
        qc["notched_hz"] = [float(f) for f in freqs]

    # ---- 2/3. band-pass and reference --------------------------------------------------
    # The ICA copy is branched off BEFORE the analysis band-pass. ICLabel is trained on 1-100 Hz
    # and its strongest evidence for muscle and eye lives above 45 Hz; branching afterwards means
    # the "1-100 Hz" filter is a no-op on data already truncated at 45, which silently degrades
    # every label. Measured on ds004504 sub-001, that ordering error took the eye-component count
    # from 6 to 0.
    ica_source = raw.copy() if run_ica else None
    amp_before = float(np.median(np.abs(raw.get_data())))

    hi_out = min(h_freq, sf / 2 - 1.0)
    raw.filter(l_freq, hi_out, method="fir", phase="zero", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    qc["bandpass_hz"] = [l_freq, float(hi_out)]

    # ---- 4. ICA -------------------------------------------------------------------------
    if run_ica:
        try:
            from mne_icalabel import label_components

            hi_ica = min(100.0, sf / 2 - 5.0)
            if hi_ica > max(ica_hp + 10.0, 30.0):
                fit_raw = ica_source
                if fit_raw.times[-1] > ica_crop_s:
                    fit_raw.crop(0, ica_crop_s)
                fit_raw.filter(ica_hp, hi_ica, verbose=False)

                # Flat channels MUST be detected before the average reference, not after. A
                # zero-filled channel becomes -mean(others) once referenced, so it is no longer
                # constant -- but two zero-filled channels become IDENTICAL to each other, and the
                # data is rank-deficient in a way no std test will show. GENEEG zero-fills T5 and
                # T6; with the check placed after referencing they survived, an 18-component
                # decomposition was fitted to rank-16 data, and `ica.apply` annihilated the
                # recording: median amplitude fell from 3.6e-06 to 3.2e-09 V while every QC field
                # reported success.
                std = fit_raw.get_data().std(axis=1)
                flat = [c for c, s in zip(fit_raw.ch_names, std) if s < 1e-15]
                if flat:
                    fit_raw.drop_channels(flat)
                qc["flat_channels"] = flat

                fit_raw.set_eeg_reference("average", projection=False, verbose=False)

                # Take the component count from the numerical rank rather than the channel count.
                # Average referencing costs one degree of freedom, and any residual collinearity
                # costs more; over-specifying is what caused the collapse above.
                rank = int(np.linalg.matrix_rank(fit_raw.get_data(), tol=1e-10))
                n_comp = max(min(len(fit_raw.ch_names) - 1, rank), 5)
                qc["data_rank"] = rank
                ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax",
                                            fit_params=dict(extended=True),
                                            random_state=seed, max_iter="auto")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ica.fit(fit_raw, verbose=False)
                    res = label_components(fit_raw, ica, method="iclabel")

                labels = np.asarray(res["labels"])
                probs = np.asarray(res["y_pred_proba"], dtype=float)
                excl = [int(i) for i in range(len(labels))
                        if labels[i] in remove and probs[i] >= MIN_PROB]

                qc["ica_n_components"] = int(n_comp)
                qc["ica_labels"] = {c: int((labels == c).sum()) for c in ICLABEL_CLASSES}
                qc["ica_excluded"] = excl
                qc["ica_excluded_labels"] = [str(labels[i]) for i in excl]

                if excl:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        var = ica.get_explained_variance_ratio(fit_raw, components=excl,
                                                               ch_type="eeg")
                    qc["variance_removed"] = float(var.get("eeg", 0.0))
                    # Measured immediately either side of `ica.apply`, so the ratio isolates the
                    # reconstruction. Comparing against the pre-filter amplitude instead would
                    # conflate ICA with the band-pass, and a cohort carrying heavy drift would
                    # look like a collapse: MISP reads 0.108 on that definition purely because
                    # ~90 per cent of its power lies outside 0.5-45 Hz.
                    qc["_amp_pre_ica"] = float(np.median(np.abs(raw.get_data())))
                    ica.exclude = excl
                    target = raw.copy().drop_channels(flat) if flat else raw
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ica.apply(target, verbose=False)
                    if flat:
                        # restore the dropped channels as zeros so the tensor shape is stable
                        pad = mne.io.RawArray(
                            np.zeros((len(flat), target.n_times)),
                            mne.create_info(flat, target.info["sfreq"], "eeg"), verbose="ERROR")
                        target.add_channels([pad], force_update_info=True)
                        target.reorder_channels(raw.ch_names)
                        raw = target
                else:
                    qc["variance_removed"] = 0.0
                qc["ica_run"] = True
            else:
                qc["ica_skipped_reason"] = f"usable band too narrow (hi={hi_ica:.0f} Hz)"
        except Exception as e:                                       # noqa: BLE001
            qc["ica_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    # ---- amplitude sanity check ----------------------------------------------------------
    # ICA reconstruction can silently annihilate a recording when the decomposition is
    # ill-conditioned (see the rank note above). Removing components should reduce amplitude by
    # the variance they explained, never by orders of magnitude, so a collapse is a hard error
    # rather than a warning: a corrupt recording that is written to cache and reported as a
    # success is far more expensive than a run that stops.
    after = float(np.median(np.abs(raw.get_data())))
    qc["median_abs_amplitude"] = after
    qc["amplitude_ratio_vs_input"] = after / amp_before if amp_before > 0 else None
    ref = qc.pop("_amp_pre_ica", None)
    if qc.get("ica_run") and ref:
        ratio = after / ref
        qc["amplitude_ratio"] = ratio
        if ratio < 0.05 or ratio > 20.0:
            raise RuntimeError(
                f"ICA reconstruction changed median amplitude by {ratio:.3g}x "
                f"({amp_before:.3e} -> {after:.3e} V). Refusing to cache a recording the "
                f"decomposition has destroyed; rank={qc.get('data_rank')}, "
                f"n_components={qc.get('ica_n_components')}, flat={qc.get('flat_channels')}.")

    # ---- 5. resample ---------------------------------------------------------------------
    if abs(sf - sfreq_out) > 1e-6:
        raw.resample(sfreq_out, verbose=False)
    qc["sfreq_out"] = float(raw.info["sfreq"])
    qc["n_channels_out"] = len(raw.ch_names)
    qc["n_times_out"] = int(raw.n_times)
    return raw, qc
