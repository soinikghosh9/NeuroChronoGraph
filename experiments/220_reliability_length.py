"""Within-recording reliability and recording-length dependence of the subject score (review W6/E11).

Uses the window-level probabilities saved by 211/237 (networks) and refits nothing for them. For the forests
(212) the subject features are re-aggregated from window features and the canonical fold models are
refitted (cheap) so the same windows can be scored.

  * Split-half reliability: subject score from the first 12 vs the last 12 of the 24 matched windows
    (window order = recording time). ICC(2,1) on the logit scale, Spearman rho, and the fraction of
    subjects whose referral decision flips between halves at the source 80%-specificity threshold.
  * Recording length: AUC when the subject score uses only K in {1, 2, 4, 8, 12, 24} windows
    (20 random draws of which windows), internal OOF and external.

    python experiments/220_reliability_length.py
Out: outputs/results/canonical/reliability_length.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CAN = ROOT / "outputs" / "results" / "canonical"
FEAT = ROOT / "outputs" / "harmonised_features" / "channel_clean"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired", "CN_vs_MCI")
DEEP = ("spectralite16", "spectral75k", "ncg_v5", "ncg_v3", "atcnet", "eegnet", "conformer", "shallow", "deep4", "eegitnet", "eegnex")
KS = (1, 2, 4, 8, 12, 24)
N_DRAWS = 20


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def icc21(a, b):
    X = np.column_stack([a, b]); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def per_subject_windows(groups, subjects):
    return {s: np.flatnonzero(groups == s) for s in subjects}


def analyse(sub_windows, win_scores, subjects, y, thr, rng):
    halves_a = np.array([win_scores[w[:len(w) // 2]].mean() for w in (sub_windows[s] for s in subjects)])
    halves_b = np.array([win_scores[w[len(w) // 2:]].mean() for w in (sub_windows[s] for s in subjects)])
    from scipy.stats import spearmanr
    out = {"icc21_logit": icc21(logit(halves_a), logit(halves_b)),
           "spearman": float(spearmanr(halves_a, halves_b).statistic),
           "decision_flip_rate": float(np.mean((halves_a >= thr) != (halves_b >= thr))),
           "auc_half_a": auc(y, halves_a), "auc_half_b": auc(y, halves_b), "length": {}}
    for K in KS:
        vals = []
        for _ in range(N_DRAWS if K < 24 else 1):
            s = np.array([win_scores[rng.choice(sub_windows[x], K, replace=False)].mean() for x in subjects])
            vals.append(auc(y, s))
        out["length"][str(K)] = {"mean": float(np.mean(vals)), "p2.5": float(np.percentile(vals, 2.5)),
                                 "p97.5": float(np.percentile(vals, 97.5))}
    return out


def main():
    rng = np.random.default_rng(20260917)
    rep = {"ks": KS, "n_draws": N_DRAWS, "tasks": {}}
    for task in TASKS:
        rep["tasks"][task] = {}
        for model in DEEP:
            f = CAN / task / f"{model}__seed42.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            thr = float(np.quantile(z["oof_score"][z["train_y"] == 0], 0.80))
            r = {}
            sw = per_subject_windows(z["win_train_groups"], z["train_subjects"])
            r["internal"] = analyse(sw, z["oof_win"].astype(float), z["train_subjects"], z["train_y"], thr, rng)
            sw = per_subject_windows(z["win_ext_groups"], z["ext_subjects"])
            r["external"] = analyse(sw, z["ext_win_folds"].astype(float).mean(0), z["ext_subjects"], z["ext_y"], thr, rng)
            rep["tasks"][task][model] = r
            print(f"{task:15s} {model:12s} ext ICC {r['external']['icc21_logit']:.2f} flip {r['external']['decision_flip_rate']:.2f} "
                  f"| AUC K=1 {r['external']['length']['1']['mean']:.3f} K=4 {r['external']['length']['4']['mean']:.3f} "
                  f"K=24 {r['external']['length']['24']['mean']:.3f}", flush=True)
        # Forest on the same 16-feature node representation used by SpectraLite. Refit canonical
        # folds on 24-window subject medians, then score window subsets.
        F = np.concatenate([
            np.load(FEAT / "node12.npy", mmap_mode="r"),
            np.load(FEAT / "node_ext4.npy", mmap_mode="r")], axis=-1)
        meta = np.load(FEAT / "meta.npz", allow_pickle=False); g = meta["groups"]
        f212 = CAN / task / "rf_node__seed42.npz"
        if f212.exists():
            from sklearn.ensemble import RandomForestClassifier
            z = np.load(f212, allow_pickle=True)
            tr, ytr, te, yte, fold = z["train_subjects"], z["train_y"], z["ext_subjects"], z["ext_y"], z["fold_of_subject"]
            win = {s: np.asarray(F[g == s]).reshape(-1, 19 * 16) for s in np.concatenate([tr, te])}
            med = lambda s, idx=None: np.nanmedian(win[s] if idx is None else win[s][idx], 0)
            Xtr = np.nan_to_num(np.stack([med(s) for s in tr]))
            fits = [(k, RandomForestClassifier(400, random_state=42, n_jobs=6, class_weight="balanced")
                     .fit(Xtr[fold != k], ytr[fold != k])) for k in range(5)]
            thr = float(np.quantile(z["oof_score"][ytr == 0], 0.80))

            def score(subjects, sel, internal):
                X = np.nan_to_num(np.stack([med(s, sel(s)) for s in subjects]))
                if internal:
                    out = np.zeros(len(subjects))
                    for k, f in fits:
                        m = fold == k; out[m] = f.predict_proba(X[m])[:, 1]
                    return out
                return np.mean([f.predict_proba(X)[:, 1] for _, f in fits], 0)
            r = {}
            for side, subs, y, internal in (("internal", tr, ytr, True), ("external", te, yte, False)):
                a = score(subs, lambda s: slice(0, len(win[s]) // 2), internal)
                b = score(subs, lambda s: slice(len(win[s]) // 2, None), internal)
                from scipy.stats import spearmanr
                d = {"icc21_logit": icc21(logit(a), logit(b)), "spearman": float(spearmanr(a, b).statistic),
                     "decision_flip_rate": float(np.mean((a >= thr) != (b >= thr))), "length": {}}
                for K in KS:
                    vals = [auc(y, score(subs, lambda s: rng.choice(len(win[s]), K, replace=False), internal))
                            for _ in range(5 if K < 24 else 1)]
                    d["length"][str(K)] = {"mean": float(np.mean(vals)), "p2.5": float(np.percentile(vals, 2.5)),
                                           "p97.5": float(np.percentile(vals, 97.5))}
                r[side] = d
            rep["tasks"][task]["rf_node"] = r
            print(f"{task:15s} rf_node      ext ICC {r['external']['icc21_logit']:.2f} flip {r['external']['decision_flip_rate']:.2f} "
                  f"| AUC K=1 {r['external']['length']['1']['mean']:.3f} K=24 {r['external']['length']['24']['mean']:.3f}", flush=True)
        (CAN / "reliability_length.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
