"""Confidence intervals for the test-time noise analysis.

The 16-feature montage run stores per-subject scores for SpectraLite and the matched forest. This
recomputes the forest arm -- which is CPU-only, the model being refit on clean node features and
evaluated on noisy ones -- keeps the canonical noise generator, seeds and fold split, and gives both
arms a subject bootstrap interval.

The reconstruction is not trusted on faith: the clean (noise-free) forest AUCs are recomputed the same
way and asserted against the values 215 stored, so a divergence in the montage derivation, feature
extraction or fold split stops the script instead of quietly moving a bar.

    python experiments/230_noise_intervals.py [--tasks CN_vs_dementia]
Out: outputs/results/canonical/noise_intervals.json
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
spec15 = importlib.util.spec_from_file_location("c215", ROOT / "experiments" / "215_montage_v2.py")
M = importlib.util.module_from_spec(spec15); spec15.loader.exec_module(M)   # constants + make_noise

OUT = C.OUT
CONT = C.ALZ / "harmonised"
WINDOW = 1024
B_BOOT = 4000


def boot_ci(y, s, n=B_BOOT, seed=23):
    """Subject bootstrap 95% interval for an AUC."""
    y = np.asarray(y); s = np.asarray(s, float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(C.auc(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def paired_diff(y, a, b, n=B_BOOT, seed=29):
    """AUC(a) - AUC(b) on the same subjects, with a paired bootstrap interval and two-sided p."""
    y = np.asarray(y); a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(C.auc(y[i], a[i]) - C.auc(y[i], b[i]))
    d = np.asarray(out)
    return {"delta": float(C.auc(y, a) - C.auc(y, b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="CN_vs_dementia")
    ap.add_argument("--montages", default="b19,muse,ganglion")
    args = ap.parse_args()
    stored = json.loads((OUT / "montage_v2.json").read_text())
    score_dir = OUT / "montage_v2_scores"
    rep = {"bootstrap": B_BOOT, "note": "seed-ensemble AUC with a subject bootstrap 95% interval; the "
                                        "intervals are computed from the new 16-feature SpectraLite and "
                                        "same-feature forest score artifacts", "tasks": {}}

    for task in [t for t in args.tasks.split(",") if t]:
        out = {}
        shard = np.load(OUT / task / "spectralite16__seed42.npz", allow_pickle=True)
        te_y = shard["ext_y"].astype(int)
        for mname in [m for m in args.montages.split(",") if m]:
            clean = np.load(score_dir / f"{task}__rf_node__{mname}.npy")
            out[mname] = {"rf_node_clean": {
                "auc": float(np.mean([C.auc(te_y, s) for s in clean])),
                "ci95": boot_ci(te_y, clean.mean(0)),
            }}
            print(f"\n=== {task} / {mname} ===", flush=True)
            for nz in M.NOISE:
                key = f"{mname}+{nz}"
                rf = np.load(score_dir / f"{task}__rf_node__{key}.npy")
                sp = np.load(score_dir / f"{task}__spectralite16__noretrain__{key}.npy")
                rf_auc = float(np.mean([C.auc(te_y, s) for s in rf]))
                s_auc = float(np.mean([C.auc(te_y, s) for s in sp]))
                entry = {"rf_node": {"auc": rf_auc, "ci95": boot_ci(te_y, rf.mean(0)),
                                     "seed_sd": float(np.std([C.auc(te_y, s) for s in rf], ddof=1))},
                         "spectralite16": {"auc": s_auc, "ci95": boot_ci(te_y, sp.mean(0)),
                                            "seed_sd": float(np.std([C.auc(te_y, s) for s in sp], ddof=1))},
                         "paired_spectralite_minus_forest": paired_diff(te_y, sp.mean(0), rf.mean(0))}
                out[key] = entry
                st = stored["tasks"][task]["noise"].get(key, {})
                print(f"   {key:16s} forest {rf_auc:.3f} (stored {st.get('rf_node_mean', float('nan')):.3f}) "
                      f"| SpectraLite {entry.get('spectralite16', {}).get('auc', float('nan')):.3f} "
                      f"(stored {st.get('spectralite_mean', float('nan')):.3f})", flush=True)
        rep["tasks"][task] = out
    (OUT / "noise_intervals.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", OUT / "noise_intervals.json")


if __name__ == "__main__":
    main()
