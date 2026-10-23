"""Probability calibration and decision-curve net benefit for the canonical models (review W6/E13).

For each model (seed ensemble) and task, internal OOF and external:
  * expected calibration error (10 equal-count bins), Brier score,
  * calibration intercept and slope (logistic recalibration of y on logit(p)),
  * decision-curve net benefit at threshold probabilities 0.05-0.50, re-weighted from the case-enriched
    cohort prevalence to service prevalences of 5%, 10% and 20% (Vickers & Elkin 2006; the re-weighting
    follows the prevalence net-benefit contract of exp 198), reported as the maximum advantage over
    treat-all/treat-none and at pt = prevalence.

    python experiments/223_calibration_dca.py
Out: outputs/results/canonical/calibration_dca.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CAN = ROOT / "outputs" / "results" / "canonical"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired", "CN_vs_MCI")
MODELS = ("rf123", "rf_node", "concept", "spectralite16", "ncg_v5", "ncg_v3", "atcnet", "eegnet", "conformer",
          "shallow", "deep4", "eegitnet", "eegnex", "gb", "svm", "logistic")
PREVS = (0.05, 0.10, 0.20)


def ece(y, p, bins=10):
    order = np.argsort(p); e = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            e += len(chunk) / len(y) * abs(p[chunk].mean() - y[chunk].mean())
    return float(e)


def cal_slope(y, p):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    f = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    return float(f.intercept_[0]), float(f.coef_[0, 0])


def net_benefit_curve(y, p, prev, pts):
    obs = y.mean(); wp = prev / obs; wn = (1 - prev) / (1 - obs); ne = wp * y.sum() + wn * (1 - y).sum()
    nb, nb_all = [], []
    for pt in pts:
        fl = p >= pt
        tp = ((y == 1) & fl).sum() * wp; fp = ((y == 0) & fl).sum() * wn
        nb.append(tp / ne - fp / ne * pt / (1 - pt)); nb_all.append(prev - (1 - prev) * pt / (1 - pt))
    return np.array(nb), np.array(nb_all)


def seed_ensemble(task, model):
    fs = sorted((CAN / task).glob(f"{model}__seed*.npz"))
    if not fs:
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]; z0 = zs[0]
    def al(z, ks, kv):
        o = {s: i for i, s in enumerate(z[ks])}; return np.array([z[kv][o[s]] for s in z0[ks]], float)
    out = dict(ytr=z0["train_y"], oof=np.mean([al(z, "train_subjects", "oof_score") for z in zs], 0),
               yte=z0["ext_y"], ext=np.mean([al(z, "ext_subjects", "ext_score") for z in zs], 0))
    if all("padic_score" in z for z in zs):
        out.update(ypa=z0["padic_y"], padic=np.mean([al(z, "padic_subjects", "padic_score") for z in zs], 0))
    return out


def main():
    pts = np.round(np.arange(0.02, 0.51, 0.01), 3)
    rep = {"thresholds": pts.tolist(), "prevalences": PREVS, "tasks": {}}
    for task in TASKS:
        rep["tasks"][task] = {}
        for model in MODELS:
            E = seed_ensemble(task, model)
            if E is None:
                continue
            r = {}
            sides = [("internal", E["ytr"], E["oof"]), ("external", E["yte"], E["ext"])]
            if "padic" in E:
                sides.append(("padic", E["ypa"], E["padic"]))
            for side, y, p in sides:
                y = np.asarray(y).astype(int); p = np.asarray(p, float)
                ci, cs = cal_slope(y, p)
                d = {"ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)), "cal_intercept": ci, "cal_slope": cs,
                     "mean_pred": float(p.mean()), "prevalence": float(y.mean()), "net_benefit": {}}
                for prev in PREVS:
                    nb, nb_all = net_benefit_curve(y, p, prev, pts)
                    adv = nb - np.maximum(nb_all, 0)
                    i_prev = int(np.argmin(np.abs(pts - prev)))
                    d["net_benefit"][str(prev)] = {"curve": nb.tolist(), "treat_all": nb_all.tolist(),
                                                   "max_advantage_per_1000": float(adv.max() * 1000),
                                                   "advantage_at_pt_equal_prev_per_1000": float(adv[i_prev] * 1000)}
                r[side] = d
            rep["tasks"][task][model] = r
            pa = (f" || P-ADIC ECE {r['padic']['ece']:.3f} slope {r['padic']['cal_slope']:.2f}"
                  if "padic" in r else "")
            print(f"{task:15s} {model:12s} ext ECE {r['external']['ece']:.3f} Brier {r['external']['brier']:.3f} "
                  f"slope {r['external']['cal_slope']:.2f} | NB@10% max adv "
                  f"{r['external']['net_benefit']['0.1']['max_advantage_per_1000']:.1f}/1000" + pa)
    (CAN / "calibration_dca.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
