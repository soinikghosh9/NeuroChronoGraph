"""Does a difference in AUC become a different referral? The empirical basis for the 0.05 margin.

The paper claims 0.05 AUC as the smallest difference worth acting on. Rather than asserting it, this
measures, for every pair of models at a common operating rule, how much their referral decisions agree
(Cohen's kappa and the fraction of discordant subjects) and relates that to their AUC difference.

It also replaces an earlier figure that showed three models' confusion matrices at a
prevalence-quota threshold and concluded that they referred "near-identical patients": a quota forces
every model to refer the same NUMBER of subjects, so similar matrices are partly built in. Agreement
is therefore reported per subject, not per margin count, and at two rules (the transferred source
threshold and the local conformal threshold with k=20 controls).

    python experiments/228_referral_agreement.py
Out: outputs/results/canonical/referral_agreement.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from _calib_splits import anchor_splits  # noqa: E402

CAN = ROOT / "outputs" / "results" / "canonical"
TASKS = ("CN_vs_AD", "CN_vs_dementia", "CN_vs_impaired")
SKIP = ("logistic",)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C221 = _load("c221", "221_local_calibration.py")


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa(a, b):
    po = float(np.mean(a == b)); pa = a.mean(); pb = b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


RULES = ["source_threshold", "local_conformal_k20"]


def analyse(task, site):
    """Pairwise decision agreement at one site. `site` selects which scores the shard supplies:
    "external" is ds004504 (GENEEG for the prodromal contrast), "padic" the second site written by
    224. Rules, bins and anchor splits are identical, so the two sites are directly comparable."""
    ky, ks, kv = {"external": ("ext_y", "ext_subjects", "ext_score"),
                  "padic": ("padic_y", "padic_subjects", "padic_score")}[site]
    models = {}
    for f in sorted((CAN / task).glob("*__seed42.npz")):
        name = f.stem.split("__")[0]
        if name.startswith("ladder") or name in SKIP:
            continue
        z = np.load(f, allow_pickle=True)
        if "oof_score" not in z or kv not in z:
            continue
        models[name] = {"y": z[ky], "subs": z[ks], "s": z[kv],
                        "thr_src": float(np.quantile(z["oof_score"][z["train_y"] == 0], 0.80))}
    if len(models) < 4:
        return None
    y = models[next(iter(models))]["y"]; subs = models[next(iter(models))]["subs"]
    ctrl = subs[y == 0]
    # decisions under each rule
    dec = {}
    for name, m in models.items():
        dec[(name, "source_threshold")] = (m["s"] >= m["thr_src"]).astype(int)
        votes = []
        for anc, _ in anchor_splits(ctrl, 20, n_splits=50):
            thr = C221.conformal_threshold(m["s"][np.isin(subs, anc)])
            votes.append((m["s"] >= thr).astype(int))
        dec[(name, "local_conformal_k20")] = (np.mean(votes, 0) >= 0.5).astype(int)
    rows = []
    for a, b in combinations(sorted(models), 2):
        d_auc = abs(auc(y, models[a]["s"]) - auc(y, models[b]["s"]))
        row = {"a": a, "b": b, "abs_delta_auc": d_auc}
        for rule in RULES:
            da, db = dec[(a, rule)], dec[(b, rule)]
            row[rule] = {"kappa": kappa(da, db), "discordant": int((da != db).sum()),
                         "n": int(len(da)), "referred_a": int(da.sum()), "referred_b": int(db.sum())}
        rows.append(row)
    summary = {}
    for rule in RULES:
        for lo, hi, lab in ((0, 0.02, "<0.02"), (0.02, 0.05, "0.02-0.05"), (0.05, 1, ">0.05")):
            sel = [r for r in rows if lo <= r["abs_delta_auc"] < hi]
            if sel:
                summary.setdefault(rule, {})[lab] = {
                    "n_pairs": len(sel),
                    "kappa_median": float(np.median([r[rule]["kappa"] for r in sel])),
                    "kappa_min": float(np.min([r[rule]["kappa"] for r in sel])),
                    "discordant_median": float(np.median([r[rule]["discordant"] for r in sel])),
                    "discordant_pct_median": float(np.median([100 * r[rule]["discordant"] / r[rule]["n"] for r in sel]))}
    print(f"\n=== {task} [{site}]: {len(models)} models, {len(rows)} pairs, n={len(y)}")
    for rule, bins in summary.items():
        print(f"  {rule}")
        for lab, st in bins.items():
            print(f"     |dAUC| {lab:10s} pairs {st['n_pairs']:3d}  kappa median {st['kappa_median']:.2f} "
                  f"(min {st['kappa_min']:.2f})  discordant {st['discordant_median']:.0f} subjects "
                  f"({st['discordant_pct_median']:.0f}%)")
    return {"n_models": len(models), "n_subjects": int(len(y)), "pairs": rows, "by_auc_gap": summary}


def main():
    rep = {"rules": RULES, "tasks": {}, "padic_tasks": {}}
    for task in TASKS:
        for site, key in (("external", "tasks"), ("padic", "padic_tasks")):
            r = analyse(task, site)
            if r is not None:
                rep[key][task] = r
    (CAN / "referral_agreement.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", CAN / "referral_agreement.json")


if __name__ == "__main__":
    main()
