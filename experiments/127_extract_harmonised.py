"""Re-extract concepts and spectral features from the harmonised cache, in both scaling arms.

Consumes `outputs/harmonised/*.npy` (cleaned, unscaled, 256 Hz, 19 channels) and produces the two
feature matrices every downstream experiment needs:

    window_concepts.npy   [n_windows, 15]   the v6 clinical concepts
    spectral.npy          [n_windows, 123]  the baseline block used by every classical model
    meta.npz                                subject id, label, cohort, window start, per-subject
                                            rejection count

Run once per scaling arm. The arms differ in one line and answer a question the project has never
tested: whether preserving inter-channel amplitude helps.

    channel   per-channel robust scale -- the historical behaviour, kept so the rebuild can be
              compared against the old numbers on equal terms
    global    one robust scale per recording, preserving the amplitude relationships between
              electrodes. This is what the topographic concepts need in order to mean anything,
              and what a graph network needs in order to have spatial structure to exploit.

Windowing replicates `dataset_wrapper.py` exactly -- 1024-sample windows, stride 1024 (the
wrapper's `overlap_ratio` defaults to 0.0), at most 50 per subject chosen by an RNG seeded from
md5(subject_id) -- so that the only difference from the previous corpus is the preprocessing.

**Artefact rejection now happens in the right order.** The manuscript describes rejecting windows
above 200 microvolts peak-to-peak, but the wrapper never rejected anything, and the PKL path
rejected AFTER scaling, where a microvolt threshold is meaningless. Here rejection is applied to
the unscaled signal in volts, before scaling, which is the only order in which the threshold means
what it says. The per-subject rejection count is recorded because it is itself an artefact-burden
measure -- and one the audit showed can separate diagnostic classes on its own.

    python experiments/127_extract_harmonised.py --mode channel
    python experiments/127_extract_harmonised.py --mode global
Out: outputs/harmonised_features/<mode>/
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result  # noqa: E402

CACHE = ROOT / "outputs" / "harmonised"
OUT_ROOT = ROOT / "outputs" / "harmonised_features"
SUMMARY = ROOT / "outputs" / "results" / "audit"

WINDOW = 1024                 # 4 s at 256 Hz, matching DATASET_CONFIG['n_times']
STRIDE = 1024                 # dataset_wrapper's overlap_ratio defaults to 0.0
MAX_WINDOWS = 50              # DATASET_CONFIG['max_windows_per_subject']
REJECT_V = 200e-6             # PREPROCESSING['reject_threshold'], peak-to-peak, in VOLTS
SFREQ = 256.0


def window_starts(n_samples: int, sid: str):
    """Reproduce dataset_wrapper.py's deterministic per-subject window selection."""
    if n_samples < WINDOW:
        return []
    starts = list(range(0, n_samples - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        seed = int(hashlib.md5(str(sid).encode("utf-8")).hexdigest()[:8], 16)
        starts = sorted(np.random.default_rng(seed).choice(starts, MAX_WINDOWS, replace=False))
    return list(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["channel", "global"], required=True)
    ap.add_argument("--estimators", choices=["repaired", "original"], default="repaired",
                    help="'repaired' = the EMG-aware estimators (2-20 Hz aperiodic fit, SEF50, "
                         "1-20 Hz entropy). 'original' = the estimators as first written. "
                         "Running both on the harmonised cache answers whether ICA cleaning "
                         "alone removes the contamination, or whether the estimator repair is "
                         "still needed -- which decides how much of the repair narrative "
                         "survives.")
    ap.add_argument("--emg-hz", type=float, default=None,
                    help="override the 20 Hz EMG-free analysis ceiling. Used only for the "
                         "sensitivity sweep; the reported model uses the default.")
    ap.add_argument("--skip-spectral", action="store_true",
                    help="the 123-d spectral block does not depend on the EMG ceiling, so the "
                         "sweep can reuse the existing one instead of recomputing it.")
    ap.add_argument("--tag", default=None,
                    help="output directory name, so an alternative corpus (e.g. all cohorts) "
                         "can be built without overwriting the one every result uses")
    ap.add_argument("--cohorts", default=None,
                    help="comma-separated cohort allowlist; default is everything cached")
    ap.add_argument("--truncate", default=None,
                    help="COHORT:SECONDS[,COHORT:SECONDS] -- truncate every recording in a "
                         "cohort to a common prefix before windowing, so recording length "
                         "cannot carry class information. Window selection then happens "
                         "within the truncated span, which is the only way to get an equal "
                         "sampling; filtering windows afterwards leaves long recordings with "
                         "a handful of windows and short ones with all of theirs.")
    ap.add_argument("--fixed-windows", type=int, default=0,
                    help="take exactly K clean windows per subject and drop subjects with "
                         "fewer, so every subject contributes equal evidence. Removes the "
                         "window-count shortcut, which survives duration matching wherever "
                         "artefact burden is class-related. The per-class retention is "
                         "reported, because a quota that excludes one class more than another "
                         "trades one bias for a worse one.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    from src.data.scaling import robust_scale
    from src.data.window_features import spectral_stream
    from src.models.v6 import ConceptExtractor

    tag = args.mode if args.estimators == "repaired" else f"{args.mode}_orig"
    if args.emg_hz:
        tag = f"{args.mode}_emg{int(args.emg_hz)}"
    if args.tag:
        tag = args.tag
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = sorted(CACHE.glob("*.json"))
    if args.cohorts:
        want = {c.strip() for c in args.cohorts.split(",")}
        sides = [s for s in sides if json.loads(s.read_text()).get("cohort") in want]
    if args.limit:
        sides = sides[:args.limit]
    if not sides:
        raise SystemExit(f"no harmonised recordings in {CACHE}; run 126 first")
    print(f"{len(sides)} harmonised recordings | scaling mode: {args.mode}")

    trunc = {}
    if args.truncate:
        for part in args.truncate.split(","):
            c, s = part.split(":")
            trunc[c.strip()] = float(s)
        print(f"  duration-matching: {trunc}")

    ex = ConceptExtractor(emg_free=(args.estimators == "repaired"),
                          emg_free_hz=args.emg_hz)
    A_rows = []
    C_rows, S_rows = [], []
    quota_dropped = []
    groups, labels, cohorts, starts_all = [], [], [], []
    rejected = {}
    t0 = time.time()

    for k, side in enumerate(sides, 1):
        meta = json.loads(side.read_text())
        sid, cohort, grp = meta["subject"], meta["cohort"], meta.get("group", "?")
        npy = side.with_suffix(".npy")
        if not npy.exists():
            continue
        data = np.load(npy)                                   # [19, T] volts, UNSCALED
        if trunc and cohort in trunc:
            data = data[:, :int(trunc[cohort] * SFREQ)]

        keep, n_rej = [], 0
        for s in window_starts(data.shape[1], sid):
            w = data[:, s:s + WINDOW]
            # Rejection on the unscaled signal, in volts, BEFORE scaling -- the only order in
            # which a 200 uV peak-to-peak threshold means what it says.
            if float(np.ptp(w, axis=1).max()) > REJECT_V:
                n_rej += 1
                continue
            keep.append((s, w))
        rejected[sid] = {"n_kept": len(keep), "n_rejected": n_rej, "cohort": cohort}
        if args.fixed_windows:
            if len(keep) < args.fixed_windows:
                quota_dropped.append((sid, cohort, grp, len(keep)))
                continue
            # deterministic subsample, seeded per subject so the choice is reproducible and
            # independent of the order recordings happen to be read in
            r = np.random.default_rng(
                int(hashlib.md5(("q" + str(sid)).encode("utf-8")).hexdigest()[:8], 16))
            pick = sorted(r.choice(len(keep), args.fixed_windows, replace=False))
            keep = [keep[i] for i in pick]
        if not keep:
            continue

        W = np.stack([robust_scale(w, args.mode) for _, w in keep]).astype(np.float32)
        with torch.no_grad():
            Wt = torch.from_numpy(W)
            C_rows.append(ex.window_concepts(Wt).cpu().numpy())
            # The three amplitude-referenced constructs are only a valid measurement when the
            # upstream scaling preserved inter-channel amplitude, so they are emitted under
            # global scaling only. Kept in a separate array: widening window_concepts.npy would
            # invalidate every cached result computed against its current 15 columns.
            if args.mode == "global":
                A_rows.append(ex.amplitude_concepts(Wt).cpu().numpy())
        if not args.skip_spectral:
            S_rows.append(np.stack([spectral_stream(w, SFREQ) for w in W]))
        for s, _ in keep:
            groups.append(sid); labels.append(grp); cohorts.append(cohort); starts_all.append(s)

        if k % 100 == 0 or k == len(sides):
            el = (time.time() - t0) / 60
            print(f"  [{k}/{len(sides)}] {len(groups):,} windows | {el:.1f} min "
                  f"(eta {el/k*(len(sides)-k):.0f} min)", flush=True)

    # A cohort that yields no windows must be an error, not a smaller table. Two cohorts have
    # vanished silently in this project -- GENEEG to an ICA rank collapse, Mendeley to an
    # undetected unit error -- and in both cases every downstream result was computed on a corpus
    # quietly missing a site.
    if args.fixed_windows:
        print("")
        print(f"  fixed-window quota K={args.fixed_windows}: dropped "
              f"{len(quota_dropped)} subjects with too few clean windows")
        kept = {}
        for g, c, l in zip(groups, cohorts, labels):
            kept[g] = (c, l)
        kept_subj = collections.Counter(kept.values())
        drop_by = collections.Counter((c, g) for _, c, g, _ in quota_dropped)
        for key in sorted(set(kept_subj) | set(drop_by)):
            k, d = kept_subj.get(key, 0), drop_by.get(key, 0)
            if k + d:
                print(f"    {key[0]:14s} {key[1]:10s} retained {k:4d}/{k+d:4d} "
                      f"({100*k/(k+d):5.1f}%)")

    seen = collections.Counter(cohorts)
    offered = collections.Counter(
        json.loads(s.read_text()).get("cohort") for s in sides)
    lost = {c: n for c, n in offered.items() if seen.get(c, 0) == 0}
    # A cohort can reach zero for two very different reasons and only one of them is a fault.
    # If every subject was excluded by the window QUOTA, the cohort simply cannot supply the
    # required span of clean EEG -- that is an informative exclusion and is reported as such.
    # If subjects survived the quota and still produced nothing, windows are being rejected
    # wholesale, which means a unit error or a mostly-flat montage and must stop the run.
    quota_only = {coh for _, coh, _, _ in quota_dropped}   # (sid, cohort, group, n_kept)
    by_quota = {c: n for c, n in lost.items() if c in quota_only}
    by_rejection = {c: n for c, n in lost.items() if c not in quota_only}
    if by_quota:
        print(f"  excluded by the K={args.fixed_windows} window quota: {by_quota}")
        print("    (these cohorts cannot supply the required span of clean EEG; this is a "
              "property of the recordings, not a failure of the run)")
    if by_rejection:
        raise SystemExit(
            f"cohorts produced ZERO windows: {by_rejection}. Every window failed the "
            f"{REJECT_V*1e6:.0f} uV rejection, which usually means a unit error or a mostly-flat "
            f"montage. Inspect outputs/harmonised/<subject>.json qc before continuing.")
    for c, n_off in offered.items():
        kept_subj = len({g for g, cc in zip(groups, cohorts) if cc == c})
        if kept_subj < 0.5 * n_off:
            print(f"  [warn] {c}: only {kept_subj}/{n_off} recordings survived window rejection")

    C = np.concatenate(C_rows).astype(np.float32)
    np.save(out_dir / "window_concepts.npy", C)
    if A_rows:
        A = np.concatenate(A_rows).astype(np.float32)
        assert len(A) == len(C), "amplitude and standard concept rows disagree"
        np.save(out_dir / "window_amp_concepts.npy", A)
        print(f"  wrote window_amp_concepts.npy {A.shape}")
    # --skip-spectral rewrites meta but leaves any existing spectral.npy untouched. If the corpus
    # changed, that file now has a different number of rows from the meta it sits beside, and any
    # join by index silently reads the wrong window. Observed: channel_emg15 held 61,582 spectral
    # rows against a 65,540-row meta. Remove it rather than leave a misaligned file on disk.
    sp = out_dir / "spectral.npy"
    if args.skip_spectral and sp.exists():
        n_rows = np.load(sp, mmap_mode="r").shape[0]
        if n_rows != len(C):
            sp.unlink()
            print(f"  removed stale spectral.npy ({n_rows} rows against {len(C)} windows)")
    if S_rows:
        S = np.concatenate(S_rows).astype(np.float32)
        np.save(out_dir / "spectral.npy", S)
    else:
        S = np.empty((len(C), 0), dtype=np.float32)
        # Borrowing the base mode's spectral block is only valid when this tag holds the SAME
        # corpus. Combining --skip-spectral with a corpus-changing flag (--truncate,
        # --fixed-windows, --cohorts) plants a misaligned file that joins by row index against
        # the wrong windows. Check before copying rather than relying on a downstream guard.
        src = OUT_ROOT / args.mode / "spectral.npy"
        if src.exists():
            n_src = np.load(src, mmap_mode="r").shape[0]
            if n_src == len(C):
                import shutil; shutil.copy2(src, out_dir / "spectral.npy")
            else:
                print(f"  [skip] not copying {args.mode}/spectral.npy: {n_src} rows against "
                      f"{len(C)} windows here. This tag holds a different corpus, so the "
                      f"spectral block must be re-extracted without --skip-spectral if a "
                      f"downstream experiment needs it.")
    np.savez_compressed(out_dir / "meta.npz",
                        groups=np.array(groups), labels=np.array(labels),
                        cohort=np.array(cohorts), start=np.array(starts_all))

    rej = np.array([v["n_rejected"] for v in rejected.values()])
    kept = np.array([v["n_kept"] for v in rejected.values()])
    rate = rej / np.maximum(rej + kept, 1)
    by_cohort = {}
    for c in sorted({v["cohort"] for v in rejected.values()}):
        r = np.array([v["n_rejected"] / max(v["n_rejected"] + v["n_kept"], 1)
                      for v in rejected.values() if v["cohort"] == c])
        by_cohort[c] = {"n_subjects": int(len(r)), "rejection_rate_median": float(np.median(r))}

    print(f"\nconcepts {C.shape} | spectral {S.shape} | {len(np.unique(groups)):,} subjects")
    print(f"window rejection rate: median {np.median(rate):.3f}, "
          f"{int((kept == 0).sum())} subjects lost every window")
    for c, v in by_cohort.items():
        print(f"  {c:14s} n={v['n_subjects']:5d} rejection {v['rejection_rate_median']:.3f}")

    write_result(SUMMARY / f"extract_harmonised_{tag}.json", {
        "mode": args.mode, "estimators": args.estimators, "n_windows": int(len(C)),
        "truncated_cohorts": trunc,
        "n_subjects": int(len(np.unique(groups))),
        "window": WINDOW, "stride": STRIDE, "max_windows_per_subject": MAX_WINDOWS,
        "reject_threshold_v": REJECT_V,
        "rejection_rate_median": float(np.median(rate)),
        "by_cohort": by_cohort,
        "elapsed_min": (time.time() - t0) / 60,
    }, script="127_extract_harmonised.py")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
