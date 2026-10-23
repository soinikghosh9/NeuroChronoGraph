"""Paired significance of SpectraLite-v2 (extended features) vs the released 12-feature model.

Reads the per-subject external scores saved by 234 (base config in the non-ext and _ext JSONs) and does a
paired subject bootstrap of the AUC difference per external site, so the transfer gain is reported with an
interval and a p-value rather than as a point estimate.
"""
import json
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
CAN = C.OUT
SITES = ("ds004504", "P-ADIC", "BrainLat")


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def paired(y, pb, pe, B=4000, seed=7):
    y = np.asarray(y, int); pb = np.asarray(pb); pe = np.asarray(pe)
    d0 = auc(y, pe) - auc(y, pb)
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            diffs.append(auc(y[i], pe[i]) - auc(y[i], pb[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
    return d0, float(lo), float(hi), float(p)


def main():
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        base = json.loads((CAN / f"spectralite_v2_{task}.json").read_text())["configs"]["base"]
        ext = json.loads((CAN / f"spectralite_v2_{task}_ext.json").read_text())["configs"]["base"]
        print(f"\n== {task}: extended(16) - base(12), paired subject bootstrap ==")
        for s in SITES:
            y = base["scores"][s]["y"]; pb = base["scores"][s]["p"]; pe = ext["scores"][s]["p"]
            d, lo, hi, p = paired(y, pb, pe)
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {s:9s} base {auc(y,pb):.3f} -> ext {auc(y,pe):.3f}  d={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} {star}")


if __name__ == "__main__":
    main()
