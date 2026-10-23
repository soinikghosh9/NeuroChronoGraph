"""Clinical performance metrics for every model, task and external cohort (protocol:
model_paper/notes/IMPROVEMENT_PROTOCOL_2026-10-02.md).

Threshold-free: AUC with DeLong 95% CI, Brier score, expected calibration error (10 equal-count bins),
calibration intercept and slope (logistic recalibration on logit p).

Three operating points, all evaluated on the SAME seeded anchor splits as the calibration study
(_calib_splits.anchor_splits, k = 20, 200 splits; metrics on held-out controls plus all cases):
  transferred   80th percentile of the model's CAUEEG out-of-fold control scores (no local data)
  conformal     split-conformal threshold from the k anchor controls (no patient labels)
  youden_oracle threshold maximising sensitivity + specificity on the evaluated cohort itself
                (uses the cohort's labels: an optimistic reference, never a deployable rule)

From each split's sensitivity and specificity, confusion-matrix metrics are computed at a stated
prevalence pi: the cohort's own case fraction (case-enriched; labelled "sample") and 10% and 20%
(clinical scenarios). Metrics: sensitivity, specificity, PPV, NPV, F1, accuracy, balanced accuracy,
MCC, Youden J. Values are the mean over splits with the 2.5-97.5th percentile range.

Inputs: every seed shard under outputs/results/canonical/<task>/ and .../improvement/<task>/; scores are
seed-ensembled (mean probability). The ensemble `ens` = mean of sl16 (canonical spectralite16) and f16.
Output: outputs/results/canonical/clinical_metrics.json

    python experiments/250_clinical_metrics.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from _calib_splits import anchor_splits  # noqa: E402

CAN = ROOT / "outputs" / "results" / "canonical"
IMP = CAN / "improvement"
OUT = CAN / "clinical_metrics.json"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired", "CN_vs_MCI")
SITE_NAME = {"ext": "ds004504", "padic": "P-ADIC", "brainlat": "BrainLat"}
K, TARGET = 20, 0.80
PREVS = {"sample": None, "10%": 0.10, "20%": 0.20}


def site_name(task, pref):
    return "GENEEG" if (task == "CN_vs_MCI" and pref == "ext") else SITE_NAME[pref]


def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s, method="average")
    n1 = int(y.sum()); n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong_ci(y, s):
    y = np.asarray(y).astype(int); pos, neg = s[y == 1], s[y == 0]
    v10 = np.array([((p > neg) + 0.5 * (p == neg)).mean() for p in pos])
    v01 = np.array([((pos > n) + 0.5 * (pos == n)).mean() for n in neg])
    a = auc(y, s); se = np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))
    return [float(max(0, a - 1.96 * se)), float(min(1, a + 1.96 * se))]


def calibration(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0                                   # 10 equal-count bins, as in 223_calibration_dca.py
    for chunk in np.array_split(np.argsort(p), 10):
        if len(chunk):
            ece += len(chunk) / len(y) * abs(p[chunk].mean() - y[chunk].mean())
    ece = float(ece)
    from sklearn.linear_model import LogisticRegression
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
    return {"brier": brier, "ece": ece, "cal_slope": float(m.coef_[0, 0]), "cal_intercept": float(m.intercept_[0])}


def conformal(anchor):
    a = np.sort(anchor); k = len(a); j = int(np.ceil((k + 1) * TARGET))
    return float(np.nextafter(a[min(j, k) - 1], np.inf))


def confusion_metrics(se, sp, pi):
    tp, fn, tn, fp = se * pi, (1 - se) * pi, sp * (1 - pi), (1 - sp) * (1 - pi)
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
    npv = tn / (tn + fn) if tn + fn > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"ppv": ppv, "npv": npv, "f1": f1, "accuracy": tp + tn, "mcc": mcc}


def summarise(rows):
    out = {}
    for key in rows[0]:
        v = np.asarray([r[key] for r in rows], float); v = v[np.isfinite(v)]
        out[key] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)),
                    "p97.5": float(np.percentile(v, 97.5))} if v.size else None
    return out


def operating_points(y, s, subs, src_thr):
    ctrl = subs[y == 0]; pi_s = float(y.mean())
    rows = defaultdict(list)
    youden = None
    thr_grid = np.unique(s)
    best = max(thr_grid, key=lambda t: np.mean(s[y == 1] >= t) + np.mean(s[y == 0] < t))
    youden = float(best)
    for anc, held in anchor_splits(ctrl, K):
        hc = np.isin(subs, held); am = np.isin(subs, anc)
        for rule, thr in (("transferred", src_thr), ("conformal", conformal(s[am])), ("youden_oracle", youden)):
            se = float(np.mean(s[y == 1] >= thr)); sp = float(np.mean(s[hc] < thr))
            row = {"sensitivity": se, "specificity": sp, "balanced_accuracy": (se + sp) / 2, "youden_j": se + sp - 1}
            for lab, pi in PREVS.items():
                for m, v in confusion_metrics(se, sp, pi_s if pi is None else pi).items():
                    row[f"{m}@{lab}"] = v
            rows[rule].append(row)
    out = {}
    for rule, r in rows.items():
        # rates: mean and 2.5-97.5th percentile over splits. Derived metrics (PPV, NPV, F1, MCC, accuracy)
        # are computed from the MEAN sensitivity and specificity (the expected operating point); their
        # per-split percentiles are given as the range. A mean of per-split PPVs would be biased upward
        # by splits in which no held-out control is referred.
        s = summarise(r)
        se, sp = s["sensitivity"]["mean"], s["specificity"]["mean"]
        for lab, pi in PREVS.items():
            for m, v in confusion_metrics(se, sp, pi_s if pi is None else pi).items():
                if s.get(f"{m}@{lab}"):
                    s[f"{m}@{lab}"]["mean"] = float(v)
        out[rule] = s
    return out, pi_s


def ensemble(files):
    zs = [np.load(f, allow_pickle=True) for f in files]; z0 = zs[0]; out = {"n_seeds": len(zs)}

    def align(z, ks, kv):
        o = {str(v): i for i, v in enumerate(z[ks])}
        return np.array([z[kv][o[str(v)]] for v in z0[ks]], float)
    out["train_y"] = z0["train_y"]; out["oof"] = np.mean([align(z, "train_subjects", "oof_score") for z in zs], 0)
    for pref in SITE_NAME:
        if all(f"{pref}_score" in z.files for z in zs):
            out[pref] = (np.asarray([str(v) for v in z0[f"{pref}_subjects"]]), z0[f"{pref}_y"].astype(int),
                         np.mean([align(z, f"{pref}_subjects", f"{pref}_score") for z in zs], 0))
    return out


def collect(task):
    """model key -> ensemble dict. Canonical shards, improvement R1 shards, R2 shards (one key per hold-out)."""
    groups = defaultdict(list)
    for f in sorted((CAN / task).glob("*__seed*.npz")):
        groups[f.name.split("__seed")[0]].append(f)
    for f in sorted((IMP / task).glob("*__seed*.npz")) if (IMP / task).exists() else []:
        groups["imp:" + f.name.split("__seed")[0]].append(f)
    models = {k: ensemble(v) for k, v in groups.items()}
    # R2: merge the three hold-out shards of one candidate into one model with three sites
    merged = {}
    for k in list(models):
        m = re.match(r"imp:(.+)__holdout-(.+)$", k)
        if m:
            merged.setdefault("imp:" + m.group(1), {}).update({p: models[k][p] for p in SITE_NAME if p in models[k]})
            merged["imp:" + m.group(1)].setdefault("oof_by_holdout", {})[m.group(2)] = models[k]["oof"]
            merged["imp:" + m.group(1)]["train_y"] = models[k]["train_y"]
            merged["imp:" + m.group(1)]["n_seeds"] = models[k]["n_seeds"]
            del models[k]
    models.update(merged)
    if "spectralite16" in models and "imp:f16" in models:
        a, b = models["spectralite16"], models["imp:f16"]
        e = {"train_y": a["train_y"], "oof": (a["oof"] + b["oof"]) / 2, "n_seeds": min(a["n_seeds"], b["n_seeds"])}
        for p in SITE_NAME:
            if p in a and p in b and np.array_equal(a[p][0], b[p][0]):
                e[p] = (a[p][0], a[p][1], (a[p][2] + b[p][2]) / 2)
        models["imp:ens"] = e
    return models


def main():
    report = {"k": K, "target_specificity": TARGET, "n_splits": 200, "prevalences": list(PREVS), "tasks": {}}
    for task in TASKS:
        models = collect(task); report["tasks"][task] = {}
        for key, E in sorted(models.items()):
            for pref in SITE_NAME:
                if pref not in E:
                    continue
                subs, y, s = E[pref]
                if len(np.unique(y)) < 2 or (y == 0).sum() <= K:
                    continue
                oof = E["oof_by_holdout"][site_name(task, pref)] if "oof_by_holdout" in E else E["oof"]
                src = float(np.quantile(oof[E["train_y"] == 0], TARGET))
                ops, pi = operating_points(y, s, subs, src)
                entry = {"n": int(y.size), "n_pos": int(y.sum()), "prevalence_sample": pi, "n_seeds": E["n_seeds"],
                         "auc": auc(y, s), "auc_ci95_delong": delong_ci(y, s), **calibration(y, s), "operating": ops}
                report["tasks"][task].setdefault(site_name(task, pref), {})[key] = entry
        for site, ms in report["tasks"][task].items():
            sl = ms.get("spectralite16")
            if sl:
                c = sl["operating"]["conformal"]
                print(f"{task:15s} {site:9s} SpectraLite AUC {sl['auc']:.3f} | conformal sens {c['sensitivity']['mean']:.2f} "
                      f"spec {c['specificity']['mean']:.2f} F1@sample {c['f1@sample']['mean']:.2f} "
                      f"MCC@sample {c['mcc@sample']['mean']:.2f} PPV@10% {c['ppv@10%']['mean']:.2f} "
                      f"NPV@10% {c['npv@10%']['mean']:.2f} | Brier {sl['brier']:.3f} ECE {sl['ece']:.3f}", flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
