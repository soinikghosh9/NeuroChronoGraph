"""How far apart are the sites, and does that distance predict what a transferred model loses?

Every model loses roughly the same amount of AUC between the two external sites (225's second-site
block), which is a statement about the corpora rather than about the architectures. This quantifies
the corpora directly, on healthy controls only so that disease cannot drive any of it:

  * site decodability -- can a classifier tell a control recording's cohort apart from a CAUEEG
    control's, from exactly the features the models are given? Subject-level, 5-fold subject CV,
    logistic regression on the 123-d QEEG block and a forest on the 228-d node block. An AUC near 1
    means the receiving clinic's healthy recordings are trivially distinguishable from the ones the
    model was trained on, which bounds what any transferred threshold can do.
  * which features move -- standardised mean difference (Hedges' g) between CAUEEG controls and each
    site's controls for the twelve per-electrode descriptors, averaged over electrodes.
  * the score's own shift -- for each canonical model, the median score of that site's controls minus
    the median out-of-fold score of CAUEEG controls, which is the quantity a transferred threshold
    is wrong by.
  * the relation -- site decodability against the mean external AUC at that site.

    python experiments/229_site_shift.py
Out: outputs/results/canonical/site_shift.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

FEATURES = ("rel. delta", "rel. theta", "rel. alpha", "rel. beta", "rel. gamma", "alpha peak freq.",
            "alpha peak prominence", "spectral edge (95%)", "aperiodic exponent", "theta/alpha",
            "slowing ratio", "spectral entropy")
CONTROL_LABELS = {"CN", "C", "CONTROL"}
MODELS = ("rf123", "rf_node", "concept", "spectral75k", "ncg_v5", "ncg_v3", "atcnet", "eegnet")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("c211", "211_canonical_benchmark.py")
CAN = C.OUT


def subject_features(tag):
    """Subject-level median of the window features, per cohort, controls flagged."""
    meta = np.load(C.FEAT / tag / "meta.npz", allow_pickle=False)
    g = meta["groups"]
    coh = np.array([str(x) for x in meta["cohort"]]); lab = np.array([str(x) for x in meta["labels"]])
    N = np.load(C.FEAT / tag / "node12.npy", mmap_mode="r")
    S = np.load(C.FEAT / tag / "spectral.npy", mmap_mode="r")
    subs = np.unique(g)
    rows = []
    for s in subs:
        i = np.flatnonzero(g == s)
        rows.append((str(s), coh[i[0]], lab[i[0]],
                     np.nanmedian(np.asarray(N[i]), 0), np.nanmedian(np.asarray(S[i]), 0)))
    return rows


def cv_auc(X, y, kind, seed=42, ci=False):
    """Subject-level 5-fold AUC of a cohort-membership classifier.

    With ci=True, also returns a subject bootstrap 95% interval on the out-of-fold scores, so the
    decodability bars can be read with their uncertainty rather than as point values."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y, int)
    if min(np.bincount(y)) < 5:
        return None
    oof = np.zeros(len(y), float)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if kind == "logistic":
            sc = StandardScaler().fit(X[a])
            f = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X[a]), y[a])
            oof[b] = f.predict_proba(sc.transform(X[b]))[:, 1]
        else:
            f = RandomForestClassifier(400, random_state=seed, n_jobs=6, class_weight="balanced").fit(X[a], y[a])
            oof[b] = f.predict_proba(X[b])[:, 1]
    a0 = float(roc_auc_score(y, oof))
    if not ci:
        return a0
    rng = np.random.default_rng(7); boot = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            boot.append(roc_auc_score(y[i], oof[i]))
    return a0, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def hedges_g(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))       # small-sample correction


def score_shift():
    """Median control score at each site minus the median out-of-fold control score."""
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        d = CAN / task
        if not d.exists():
            continue
        out[task] = {}
        for model in MODELS:
            fs = sorted(d.glob(f"{model}__seed*.npz"))
            if not fs:
                continue
            z = np.load(fs[0], allow_pickle=True)
            ref = float(np.median(z["oof_score"][z["train_y"] == 0]))
            r = {"internal_control_median": ref,
                 "ds004504": float(np.median(z["ext_score"][z["ext_y"] == 0])) - ref}
            if "padic_score" in z:
                r["padic"] = float(np.median(z["padic_score"][z["padic_y"] == 0])) - ref
            out[task][model] = r
    return out


def main():
    rows = (subject_features("channel_clean") + subject_features("padic_channel_clean")
            + subject_features("brainlat_channel_clean"))
    # P-ADIC appears in both stores (its MCI file was extracted with the main corpus); keep the
    # dedicated extraction, which is the one 224 scored.
    seen, keep = set(), []
    for r in sorted(rows, key=lambda r: r[1] != "P-ADIC"):
        if r[0] in seen:
            continue
        seen.add(r[0]); keep.append(r)
    rows = keep
    ctrl = [r for r in rows if r[2] in CONTROL_LABELS]
    by = {}
    for r in ctrl:
        by.setdefault(r[1], []).append(r)
    print({k: len(v) for k, v in by.items()})
    base = by.get("CAUEEG", [])
    rep = {"n_control_subjects": {k: len(v) for k, v in by.items()}, "features": list(FEATURES), "sites": {}}
    # Null control: the same classifiers on a random split of the CAUEEG controls. Without it, a high
    # decodability AUC could be a property of the estimator and the feature count rather than of the
    # cohorts. The split-in-half null has far more subjects per class than a real site comparison, so
    # a size-matched null is also run: 20 draws of n_site CAUEEG controls against a disjoint n_site,
    # which is the same estimator at the same sample size as the smallest site.
    if len(base) >= 40:
        rng = np.random.default_rng(20260918)
        half = rng.permutation(len(base)) < len(base) // 2
        Xn0 = np.array([r[3].ravel() for r in base]); Xs0 = np.array([r[4] for r in base])
        rep["null_within_CAUEEG_controls"] = {
            "logistic_qeeg123": cv_auc(Xs0, half.astype(int), "logistic"),
            "forest_node228": cv_auc(Xn0, half.astype(int), "forest")}
        print("null (CAUEEG controls split at random):", rep["null_within_CAUEEG_controls"], flush=True)
        sized = {}
        for n_site in sorted({len(v) for k, v in by.items() if k != "CAUEEG"}):
            if 2 * n_site > len(base):
                continue
            aucs = []
            for d in range(20):
                idx = np.random.default_rng(1000 + d).permutation(len(base))[: 2 * n_site]
                y = np.array([0] * n_site + [1] * n_site)
                a = cv_auc(Xs0[idx], y, "logistic")
                if a is not None:
                    aucs.append(a)
            if aucs:
                sized[str(n_site)] = {"mean": float(np.mean(aucs)), "p2.5": float(np.percentile(aucs, 2.5)),
                                      "p97.5": float(np.percentile(aucs, 97.5)), "n_draws": len(aucs)}
        rep["null_size_matched"] = sized
        print("size-matched nulls (CAUEEG controls vs CAUEEG controls):",
              {k: round(v["mean"], 3) for k, v in sized.items()}, flush=True)
    for site in ("ds004504_raw", "P-ADIC", "BrainLat", "GENEEG"):
        other = by.get(site, [])
        if len(other) < 10 or not base:
            continue
        Xn = np.array([r[3].ravel() for r in base + other])
        Xs = np.array([r[4] for r in base + other])
        y = np.array([0] * len(base) + [1] * len(other))
        g = {f: hedges_g([r[3][:, i].mean() for r in other], [r[3][:, i].mean() for r in base])
             for i, f in enumerate(FEATURES)}
        a_log, ci_log = cv_auc(Xs, y, "logistic", ci=True)
        a_for, ci_for = cv_auc(Xn, y, "forest", ci=True)
        rep["sites"][site] = {
            "n_controls": len(other),
            "decodability_auc": {"logistic_qeeg123": a_log, "forest_node228": a_for},
            "decodability_ci95": {"logistic_qeeg123": ci_log, "forest_node228": ci_for},
            "hedges_g_vs_CAUEEG_controls": g,
            "largest_shifts": sorted(g, key=lambda f: -abs(g[f]))[:4]}
        s = rep["sites"][site]
        print(f"{site:14s} n={len(other):3d} controls | site decodability: logistic(QEEG-123) "
              f"{s['decodability_auc']['logistic_qeeg123']:.3f}, forest(node-228) "
              f"{s['decodability_auc']['forest_node228']:.3f} | largest shifts "
              f"{', '.join(f'{f} g={g[f]:+.2f}' for f in s['largest_shifts'])}", flush=True)
    rep["score_shift"] = score_shift()
    # mean external AUC per site, for the relation between distance and loss
    mean_auc = {}
    summ = CAN / "summary.json"
    if summ.exists():
        S = json.loads(summ.read_text())
        for task, t in S.get("tasks", {}).items():
            for set_name, key in (("external", "ds004504_raw"), ("padic", "P-ADIC")):
                rowsx = t.get("evaluation_sets", {}).get(set_name, {})
                nets = [m for m in rowsx if "params" in rowsx[m]]
                if nets:
                    mean_auc.setdefault(task, {})[key] = float(np.mean([rowsx[m]["auc"] for m in nets]))
    rep["mean_network_auc_by_site"] = mean_auc
    (CAN / "site_shift.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", CAN / "site_shift.json")


if __name__ == "__main__":
    main()
