"""Age-adjusted discrimination and decision curves for the external cohorts (audit 2026-10-02, section 8).

Part A, age adjustment (ds004504, P-ADIC, BrainLat; SpectraLite, SpectraLite-12, QEEG forest):
  * covariate-adjusted AUC (Janes & Pepe 2008, linear control reference): score residualised on age with the
    slope fitted on that cohort's controls only, then the AUC of the residual. It is the probability that a case
    scores higher than a control of the same age. Subject bootstrap (4,000; slope refit in every resample).
  * likelihood-ratio test of the score beyond age: logistic y ~ age versus y ~ age + z(logit score), 1 df,
    fitted on the cohort itself (an association test, not a deployable model); odds ratio per SD of score
    adjusted for age with a Wald 95% CI.
Part B, net benefit of the referral rules at clinical prevalence (10% and 20%):
  NB(pt) = pi*sens - (1-pi)*(1-spec)*pt/(1-pt), for threshold probabilities pt in 0.02-0.30, for
  * the transferred threshold (80th percentile of CAUEEG out-of-fold control scores; no local data),
  * the split-conformal threshold from k = 20 local controls (_calib_splits anchor splits, as in Table 3),
  * refer everyone (sens = spec complement = 1) and refer no one (NB = 0).
  NB is linear in sens and spec, so the point value is the mean over the 200 anchor splits. Intervals: 2,000
  resamples, each drawing one anchor split at random and resampling its held-out controls and all cases with
  replacement (captures both calibration-split and sampling variability).
-> outputs/results/canonical/age_adjusted_net_benefit.json

    python experiments/255_age_adjusted_and_net_benefit.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


A = _module("a248", "248_age_confound_all_sites.py")
CM = _module("c250", "250_clinical_metrics.py")
from _calib_splits import anchor_splits  # noqa: E402

CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "age_adjusted_net_benefit.json"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
MODELS = {"spectralite16": "SpectraLite", "spectral75k": "SpectraLite-12", "rf123": "Forest (QEEG-123)"}
SITES = list(A.SITES)   # CAUEEG = out-of-fold; net benefit only for external cohorts
BOOT_AUC, BOOT_NB, SEED = 4000, 2000, 20261003
PT = np.round(np.arange(0.02, 0.301, 0.01), 3)
PREV = (0.10, 0.20)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def adjusted_auc(y, s, age):
    c = y == 0
    b = np.polyfit(age[c], s[c], 1)[0]
    return A.auc(y, s - b * age)


def _logit_fit(X, y, iters=100):
    """Unpenalised logistic MLE by Newton-Raphson; returns (beta, se, log-likelihood)."""
    X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H))), float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), X @ b


def lr_test(y, s, age):
    from scipy.stats import chi2
    z = logit(s); z = (z - z.mean()) / z.std()
    a = (age - age.mean()) / age.std()
    _, _, ll0, eta0 = _logit_fit(a[:, None], y)
    b1, se1, ll1, eta1 = _logit_fit(np.column_stack([a, z]), y)
    lr = 2 * (ll1 - ll0)
    beta, se = b1[2], se1[2]
    return {"lr_chi2": float(lr), "p": float(chi2.sf(lr, 1)), "or_per_sd": float(np.exp(beta)),
            "or_ci95": [float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))],
            "auc_age_only_insample": A.auc(y, eta0), "auc_age_plus_score_insample": A.auc(y, eta1)}


def net_benefit(se, sp, pi):
    w = PT / (1 - PT)
    return pi * se - (1 - pi) * (1 - sp) * w


def main():
    ages_of = A.age_maps()
    rng = np.random.default_rng(SEED)
    out = {"bootstrap_auc": BOOT_AUC, "bootstrap_nb": BOOT_NB, "pt": PT.tolist(), "prevalences": list(PREV),
           "k": CM.K, "target_specificity": CM.TARGET, "tasks": {}}
    for task in TASKS:
        for model, disp in MODELS.items():
            src = A.ensemble(task, model, A.SITES[0])
            if src is None:
                continue
            _, ytr, oof, _ = src
            tau_tr = float(np.quantile(oof[ytr == 0], CM.TARGET))
            for site in SITES:
                e = A.ensemble(task, model, site)
                if e is None:
                    continue
                subj, y, s, nseed = e
                age = np.array([ages_of(x, site[0]) for x in subj], float)
                ok = np.isfinite(age); subj, y, s, age = subj[ok], y[ok], s[ok], age[ok]
                row = {"model": disp, "n": int(len(y)), "n_cases": int(y.sum()), "n_seeds": nseed,
                       "auc": A.auc(y, s), "auc_age_alone": A.auc(y, age)}
                # A. covariate-adjusted AUC with subject bootstrap
                aa = adjusted_auc(y, s, age); bs = []
                i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
                for _ in range(BOOT_AUC):
                    i = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
                    bs.append(adjusted_auc(y[i], s[i], age[i]))
                row["auc_age_adjusted"] = aa
                row["auc_age_adjusted_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                row["beyond_age"] = lr_test(y, s, age)
                # B. net benefit (dementia/AD referral) for the SpectraLite models only
                if model in ("spectralite16", "spectral75k") and site[0] != "CAUEEG" and task != "CN_vs_impaired":
                    ctrl = subj[y == 0]; cases = s[y == 1]
                    splits = list(anchor_splits(ctrl, CM.K))
                    pos = {x: i for i, x in enumerate(subj)}
                    se_c, sp_c, thr_c = [], [], []
                    for anc, held in splits:
                        t = CM.conformal(s[[pos[x] for x in anc]]); thr_c.append(t)
                        sh = s[[pos[x] for x in held]]
                        se_c.append(float(np.mean(cases >= t))); sp_c.append(float(np.mean(sh < t)))
                    se_t, sp_t = float(np.mean(cases >= tau_tr)), float(np.mean(s[y == 0] < tau_tr))
                    nb = {}
                    for pi in PREV:
                        draws_c, draws_t = [], []
                        for _ in range(BOOT_NB):
                            j = rng.integers(len(splits)); anc, held = splits[j]
                            sh = s[[pos[x] for x in held]]; sh = sh[rng.integers(0, len(sh), len(sh))]
                            cs = cases[rng.integers(0, len(cases), len(cases))]
                            draws_c.append(net_benefit(np.mean(cs >= thr_c[j]), np.mean(sh < thr_c[j]), pi))
                            sc = s[y == 0][rng.integers(0, (y == 0).sum(), (y == 0).sum())]
                            draws_t.append(net_benefit(np.mean(cs >= tau_tr), np.mean(sc < tau_tr), pi))
                        dc, dt = np.array(draws_c), np.array(draws_t)
                        nb[str(pi)] = {
                            "conformal": {"nb": net_benefit(np.mean(se_c), np.mean(sp_c), pi).tolist(),
                                          "lo": np.percentile(dc, 2.5, 0).tolist(), "hi": np.percentile(dc, 97.5, 0).tolist()},
                            "transferred": {"nb": net_benefit(se_t, sp_t, pi).tolist(),
                                            "lo": np.percentile(dt, 2.5, 0).tolist(), "hi": np.percentile(dt, 97.5, 0).tolist()},
                            "refer_all": net_benefit(1.0, 0.0, pi).tolist()}
                    row["operating"] = {"conformal_sens": float(np.mean(se_c)), "conformal_spec": float(np.mean(sp_c)),
                                        "transferred_sens": se_t, "transferred_spec": sp_t}
                    row["net_benefit"] = nb
                out["tasks"].setdefault(task, {}).setdefault(site[0], {})[model] = row
                print(f"{task:15s} {site[0]:9s} {disp:18s} AUC {row['auc']:.3f} age-adj {aa:.3f} "
                      f"[{row['auc_age_adjusted_ci95'][0]:.3f}, {row['auc_age_adjusted_ci95'][1]:.3f}] "
                      f"LR p {row['beyond_age']['p']:.2g} OR/SD {row['beyond_age']['or_per_sd']:.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
