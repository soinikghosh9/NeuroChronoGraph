"""Does residual artifact burden carry the external discrimination? (internal review, major comment 4)

The pipeline keeps ICA components labelled muscle, because muscle overlaps the analysed band, and SpectraLite
reads relative gamma power and an aperiodic fit up to 40 Hz. This checks, at the three external dementia
cohorts and for SpectraLite and NCG-v3 (control versus dementia, seed ensembles), whether the score separates
cases from controls beyond two subject-level measures of artifact burden:

  muscle proxy     relative 30-45 Hz power, median over the 24 analysed windows, mean over electrodes
                   (the "rel gamma" node feature; EMG dominates this band in scalp EEG)
  rejection rate   fraction of 4 s windows over 200 uV peak-to-peak within the analysed 235 s span
                   (BrainLat rejected no window, so only the muscle proxy is used there)

Per cohort and model:
  * AUC of each artifact measure alone (case versus control);
  * Spearman correlation of the score with the muscle proxy within controls and within cases;
  * covariate-adjusted AUC: the score residualized on the artifact measures by a linear fit in controls
    (placement-value approach, Janes and Pepe), with a stratified subject-bootstrap 95% interval;
  * likelihood-ratio test of the score beyond the artifact measures (logistic regression, 1 df);
  * AUC after excluding the third of subjects with the highest muscle proxy, and after excluding subjects
    with any rejected window.

Nothing here selects or changes a model; it is a sensitivity analysis written after the review.

    python experiments/265_artifact_sensitivity.py
Out: outputs/results/canonical/artifact_sensitivity.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAN = ROOT / "outputs" / "results" / "canonical"
FEAT = ROOT / "outputs" / "harmonised_features"
HARM = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles")) / "harmonised"
OUT = CAN / "artifact_sensitivity.json"
TASK = "CN_vs_dementia"
MODELS = {"spectral75k": "SpectraLite", "ncg_v3": "NCG-v3"}
SETS = {"ds004504": ("ext", "channel_clean", "ds004504_raw"), "P-ADIC": ("padic", "padic_channel_clean", None),
        "BrainLat": ("brainlat", "brainlat_channel_clean", None)}
GAMMA = 4            # index of "rel gamma" among the 12 node features (experiment 264 NAMES)
B = 2000


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def auc(y, x):
    y = np.asarray(y, int); x = np.asarray(x, float)
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, fn, rng):
    i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0); v = []
    for _ in range(B):
        ii = np.r_[rng.choice(i1, i1.size), rng.choice(i0, i0.size)]
        v.append(fn(ii))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def seed_scores(model, key):
    """Seed-mean scores of one model at one evaluation set, keyed by subject id."""
    fs = sorted((CAN / TASK).glob(f"{model}__seed*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    subj, yk, sk = f"{key}_subjects", f"{key}_y", f"{key}_score"
    ids = [str(v) for v in zs[0][subj]]
    sc = []
    for z in zs:
        o = {str(v): i for i, v in enumerate(z[subj])}
        sc.append(np.array([z[sk][o[v]] for v in ids], float))
    return ids, np.asarray(zs[0][yk]).astype(int), np.mean(sc, 0), len(zs)


def muscle_proxy(tag, cohort):
    m = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    g = m["groups"].astype(str); coh = m["cohort"].astype(str)
    N = np.load(FEAT / tag / "node12.npy", mmap_mode="r")
    keep = np.ones(len(g), bool) if cohort is None else coh == cohort
    out = {}
    for s in np.unique(g[keep]):
        idx = np.flatnonzero(g == s)
        out[s] = float(np.nanmean(np.nanmedian(np.asarray(N[idx, :, GAMMA], float), 0)))
    return out


def rejection_rates(cohort_name):
    """Per-recording rejection rate inside the analysed span, computed exactly as experiment 226 does."""
    c226 = _mod("c226", "226_corpus_confounds.py")
    out = {}
    for side in sorted(HARM.glob("*.json")):
        meta = json.loads(side.read_text())
        if meta.get("cohort") != cohort_name or not side.with_suffix(".npy").exists():
            continue
        a = np.load(side.with_suffix(".npy"), mmap_mode="r")
        span = a[:, :int(c226.TRUNC[cohort_name] * c226.SFREQ)]
        kept = rej = 0
        for s in c226.starts(span.shape[1], meta["subject"]):
            if float(np.ptp(np.asarray(span[:, s:s + c226.WINDOW]), axis=1).max()) > c226.REJECT_V:
                rej += 1
            else:
                kept += 1
        out[str(meta["subject"])] = rej / max(1, kept + rej)
    return out


def loglik(X, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def zs(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def analyse(y, s, cov, rng):
    """cov: dict name -> per-subject values aligned with y and s."""
    names = [k for k in cov if np.nanstd(cov[k]) > 0]
    C = np.column_stack([cov[k] for k in names]) if names else np.zeros((len(y), 0))
    res = {"n": int(len(y)), "n_cases": int(y.sum()), "auc_score": auc(y, s),
           "auc_score_ci95": boot_ci(y, lambda ii: auc(y[ii], s[ii]), rng),
           "artifact_auc": {k: auc(y, cov[k]) for k in names},
           "spearman_score_muscle": {
               "controls": float(spearmanr(s[y == 0], cov["muscle"][y == 0])[0]),
               "cases": float(spearmanr(s[y == 1], cov["muscle"][y == 1])[0])}}

    def adjusted(ii):
        yy, ss, CC = y[ii], s[ii], C[ii]
        D = np.column_stack([np.ones(len(ii)), CC])
        beta, *_ = np.linalg.lstsq(D[yy == 0], ss[yy == 0], rcond=None)
        return auc(yy, ss - D @ beta)
    res["adjusted_auc"] = adjusted(np.arange(len(y)))
    res["adjusted_auc_ci95"] = boot_ci(y, adjusted, rng)
    Xc = np.column_stack([zs(C[:, j]) for j in range(C.shape[1])]) if C.shape[1] else np.zeros((len(y), 0))
    l0 = loglik(Xc, y) if Xc.shape[1] else float(np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())))
    l1 = loglik(np.column_stack([Xc, zs(s)]), y)
    res["lr_beyond_artifacts"] = {"chi2": 2 * (l1 - l0), "p": float(chi2.sf(max(0.0, 2 * (l1 - l0)), 1)),
                                  "covariates": names}
    keep = cov["muscle"] <= np.quantile(cov["muscle"], 2 / 3)
    res["excl_top_muscle_third"] = {"n": int(keep.sum()), "n_cases": int(y[keep].sum()), "auc": auc(y[keep], s[keep]),
                                    "auc_ci95": boot_ci(y[keep], lambda ii: auc(y[keep][ii], s[keep][ii]), rng)}
    if "rejection" in names:
        k2 = cov["rejection"] == 0
        res["excl_any_rejection"] = {"n": int(k2.sum()), "n_cases": int(y[k2].sum()), "auc": auc(y[k2], s[k2])}
    return res


def main():
    rng = np.random.default_rng(2651)
    rep = {"task": TASK, "bootstrap": B, "muscle_proxy": "rel gamma (30-45 Hz), median over windows, mean over electrodes",
           "results": {}}
    for site, (key, tag, coh) in SETS.items():
        mus = muscle_proxy(tag, coh)
        rej = rejection_rates("ds004504_raw" if site == "ds004504" else "P-ADIC") if site != "BrainLat" else {}
        for model, disp in MODELS.items():
            ids, y, s, nseed = seed_scores(model, key)
            miss = [i for i in ids if i not in mus]
            if miss:
                raise SystemExit(f"{site}: {len(miss)} subjects without features, e.g. {miss[:3]}")
            cov = {"muscle": np.array([mus[i] for i in ids])}
            if rej:
                missing_r = [i for i in ids if i not in rej]
                if missing_r:
                    raise SystemExit(f"{site}: {len(missing_r)} subjects without rejection rate, e.g. {missing_r[:3]}")
                cov["rejection"] = np.array([rej[i] for i in ids])
            r = analyse(y, s, cov, rng); r["n_seeds"] = nseed
            rep["results"].setdefault(site, {})[disp] = r
            print(f"{site:9s} {disp:12s} AUC {r['auc_score']:.3f} adj {r['adjusted_auc']:.3f} "
                  f"[{r['adjusted_auc_ci95'][0]:.3f},{r['adjusted_auc_ci95'][1]:.3f}] LR P {r['lr_beyond_artifacts']['p']:.2g} "
                  f"| muscle AUC {r['artifact_auc'].get('muscle', float('nan')):.3f} "
                  f"rej AUC {r['artifact_auc'].get('rejection', float('nan')):.3f} | rho ctrl "
                  f"{r['spearman_score_muscle']['controls']:+.2f} case {r['spearman_score_muscle']['cases']:+.2f} | "
                  f"excl top third {r['excl_top_muscle_third']['auc']:.3f} (n={r['excl_top_muscle_third']['n']})"
                  + (f" | excl any rej {r['excl_any_rejection']['auc']:.3f} (n={r['excl_any_rejection']['n']})"
                     if "excl_any_rejection" in r else ""), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
