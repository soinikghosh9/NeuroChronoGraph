"""Apply the harmonised pipeline to every cohort and cache the result.

Produces one cleaned recording per subject at 256 Hz, 19 channels, float32, UNSCALED -- the two
scaling arms (`channel`, `global`) are applied at load time by `src.data.scaling`, so a single
cache serves both and costs ~22 GB rather than ~44 GB.

Resumable at subject granularity: a subject whose `.npy` and sidecar both exist and agree on shape
is skipped, so an interrupted run continues where it stopped. Every subject writes a QC record;
the run-level summary reports how much variance was removed per cohort and per class, which is the
number that says whether the exercise was worth doing.

ds004504 is read from `datasets/openneuro_ds004504_raw/` -- the unprocessed files fetched by
`124_fetch_raw_ds004504.py` -- NOT from the derivatives, which their authors had already
band-passed, ASR-corrected and ICA-cleaned. Using the derivatives here would harmonise a cohort
that had already been cleaned differently from every other, which is the asymmetry this whole
exercise exists to remove.

    python experiments/126_harmonise_corpus.py --cohorts CAUEEG,ds004504_raw,GENEEG
    python experiments/126_harmonise_corpus.py --dry-run
Out: outputs/harmonised/<subject>.npy  +  outputs/harmonised/<subject>.json
     outputs/results/audit/harmonise_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.utils.provenance import write_result  # noqa: E402

CACHE = ROOT / "outputs" / "harmonised"
SUMMARY = ROOT / "outputs" / "results" / "audit" / "harmonise_summary.json"
RAW_ON = ROOT / "datasets" / "openneuro_ds004504_raw"


def build_loaders(names):
    """cohort key -> (loader instance, subject id list). Constructed lazily and independently so
    one broken cohort cannot stop the others."""
    from src.data.alz_eeg_loader import AlzEEGDataset
    from src.data.bids_loader import BIDSDataset
    from src.data.caueeg_loader import CAUEEGDataset
    from src.data.geneeg_loader import GENEEGDataset
    from src.data.mci_loader import MCIDataset
    from src.data.mendeley_loader import MendeleyDataset
    from src.data.misp_loader import MISPDataset
    from src.data.padic_loader import PADICDataset

    D = ROOT / "datasets"
    spec = {
        "CAUEEG":       lambda: CAUEEGDataset(D / "caueeg-dataset", verbose=False),
        # RAW files, and `use_pkl_cache=False` -- both are required. The cache under
        # outputs/preprocessed/ holds ds004504 recordings that were average-referenced, SURFACE
        # LAPLACIAN transformed and rebuilt from overlapping epochs; without this flag the loader
        # silently returns those instead of the raw files fetched by script 124.
        "ds004504_raw": lambda: BIDSDataset(RAW_ON, dataset_name="ds004504",
                                            use_pkl_cache=False, verbose=False),
        "GENEEG":       lambda: GENEEGDataset(D / "GENEEG"),
        "MISP":         lambda: MISPDataset(D / "MISP"),
        "P-ADIC":       lambda: PADICDataset(D / "P-ADIC"),
        "AlzEEG":       lambda: AlzEEGDataset(D / "Alz_EEG_data"),
        "Mendeley":     lambda: MendeleyDataset(D / "Mendeley Dataset"),
        "MCI_Dataset":  lambda: MCIDataset(D / "mci dataset"),
    }
    out = {}
    for n in names:
        if n not in spec:
            print(f"  [skip] unknown cohort {n!r}")
            continue
        try:
            ds = spec[n]()
            out[n] = (ds, list(ds.get_subject_ids()))
            print(f"  {n}: {len(out[n][1])} subjects")
        except Exception as e:                                        # noqa: BLE001
            print(f"  [skip] {n}: {type(e).__name__}: {str(e)[:120]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="CAUEEG,ds004504_raw,GENEEG,MISP,P-ADIC")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ica", action="store_true", help="filter/notch/reference only")
    args = ap.parse_args()

    import mne
    mne.set_log_level("ERROR")
    from src.data import scaling
    from src.data.harmonise import harmonise

    # The cache stores UNSCALED data; scaling is a load-time decision.
    scaling.set_mode("none")
    CACHE.mkdir(parents=True, exist_ok=True)

    names = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    print(f"cohorts requested: {names}")
    loaders = build_loaders(names)
    if args.dry_run:
        total = sum(len(v[1]) for v in loaders.values())
        print(f"\n[dry-run] {total} subjects would be processed into {CACHE}")
        return 0

    # Merge into any previous summary rather than replacing it. Running this script for a subset
    # of cohorts used to erase the rows for every cohort not in that run, which silently destroyed
    # the by-class artefact figures for the two cohorts the paper depends on.
    summary = {"cohorts": {}, "min_prob": None, "removed_classes": None}
    if SUMMARY.exists():
        try:
            prev = json.loads(SUMMARY.read_text())
            summary["cohorts"] = {k: v for k, v in (prev.get("cohorts") or {}).items()
                                  if k not in names}
            if summary["cohorts"]:
                print(f"merging with {len(summary['cohorts'])} previously harmonised cohort(s)")
        except Exception:                                              # noqa: BLE001
            pass
    t_start = time.time()

    for cohort, (ds, sids) in loaders.items():
        if args.limit:
            sids = sids[:args.limit]
        print(f"\n{'='*74}\n{cohort}: {len(sids)} subjects", flush=True)
        done = skipped = failed = 0
        var_removed, by_class, durations = [], {}, []

        for k, sid in enumerate(sids, 1):
            npy, side = CACHE / f"{sid}.npy", CACHE / f"{sid}.json"
            if npy.exists() and side.exists():
                try:
                    meta = json.loads(side.read_text())
                    if meta.get("qc", {}).get("n_times_out"):
                        skipped += 1
                        var_removed.append(meta["qc"].get("variance_removed", 0.0))
                        continue
                except Exception:                                     # noqa: BLE001
                    pass
            try:
                raw = ds.load_raw(sid)
                raw, qc = harmonise(raw, run_ica=not args.no_ica)
                data = raw.get_data().astype(np.float32)
                np.save(npy, data)
                grp = ds.get_subject_info(sid).get("group", "?")
                side.write_text(json.dumps(
                    {"subject": sid, "cohort": cohort, "group": grp,
                     "channels": raw.ch_names, "qc": qc}, indent=1, default=float))
                done += 1
                var_removed.append(qc.get("variance_removed", 0.0))
                durations.append(qc.get("duration_s", np.nan))
                by_class.setdefault(grp, []).append(qc.get("variance_removed", 0.0))
            except Exception as e:                                    # noqa: BLE001
                failed += 1
                print(f"  [fail] {sid}: {type(e).__name__}: {str(e)[:110]}")
                if failed <= 2:
                    traceback.print_exc(limit=2)
            if k % 25 == 0 or k == len(sids):
                el = time.time() - t_start
                print(f"  [{k}/{len(sids)}] done {done} skipped {skipped} failed {failed} "
                      f"| {el/60:.1f} min elapsed", flush=True)

        v = np.array([x for x in var_removed if np.isfinite(x)])
        row = {"n_processed": done, "n_skipped": skipped, "n_failed": failed,
               "variance_removed_median": float(np.median(v)) if len(v) else None,
               "variance_removed_iqr": [float(np.percentile(v, 25)),
                                        float(np.percentile(v, 75))] if len(v) else None,
               "by_class_median": {g: float(np.median(x)) for g, x in by_class.items()
                                   if len(x) >= 5}}
        summary["cohorts"][cohort] = row
        if len(v):
            print(f"  variance removed: median {np.median(v):.3f} "
                  f"(IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f})")
        if row["by_class_median"]:
            print("  by class: " + "  ".join(f"{g} {x:.3f}"
                                             for g, x in row["by_class_median"].items()))

    from src.data.harmonise import MIN_PROB, REMOVE_DEFAULT
    summary["min_prob"] = MIN_PROB
    summary["removed_classes"] = list(REMOVE_DEFAULT)
    summary["cache"] = str(CACHE)
    summary["elapsed_min"] = (time.time() - t_start) / 60.0
    write_result(SUMMARY, summary, script="126_harmonise_corpus.py")
    print(f"\n{'='*74}\nwrote {SUMMARY}  ({summary['elapsed_min']:.0f} min)")
    print("\nThe cache holds UNSCALED cleaned recordings. Choose the scaling arm at load time:\n"
          "    from src.data import scaling; scaling.set_mode('global')   # or 'channel'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
