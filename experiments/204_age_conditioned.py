"""Age-conditioned normative deviation (NeuroIPS Deviation layer) as a transfer/operating-point lever.

Adding age as a feature failed on transfer (-0.064 external) because the age->outcome map inverts
between sites. The normative-deviation layer instead uses the age->normal-EEG map, fitted on CONTROLS
only (never a label), which is a physiological ageing curve and transports far better. Each concept x
is replaced by its signed deviation z = (x - mu(age)) / sigma(age); a district clinic can re-anchor
the reference on ~20 healthy volunteers (fit_anchor), no labels, no retraining.

Three arms, one estimator (L2 logistic, standardised), same subjects/protocol:
    raw          concept features
    dev          normative deviation (reference fit on source controls, refit per fold internally)
    dev_anchor   dev + control-anchoring on the TARGET site's controls (external only)

Reported per contrast: internal OOF AUC; external AUC; and the operating point at a threshold fixed
to 80% specificity on SOURCE out-of-fold controls, applied once to the target -- target specificity
(over/under-flagging) and sensitivity. The Deviation virtue is transport-robust specificity, not AUC.

    python experiments/204_age_conditioned.py
Out: outputs/results/audit/age_conditioned.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "outputs" / "results" / "audit"
SEED = 42
_FALLBACK = Path("C:/Users/user/Downloads/Alzfiles")
FLOOR = 1e-9
_LOG_ABS_TO_SD = (np.euler_gamma + np.log(2)) / 2


def _resolve(rel):
    for base in (ROOT, _FALLBACK):
        if (base / rel).exists():
            return base / rel
    return ROOT / rel


CAU_ANN = _resolve("datasets/caueeg-dataset/annotation.json")
ON_TSV = _resolve("datasets/openneuro_ds004504_raw/participants.tsv")
CONTRASTS = [("CN_vs_dementia", ("AD", "FTD"), ("CN",)),
             ("CN_vs_AD", ("AD",), ("CN",)),
             ("CN_vs_impaired", ("AD", "FTD", "MCI"), ("CN",))]


# ---- normative core (verbatim from D:/NeuroIPS eegbudget/normative.py) ----
def _design(age):
    age = np.asarray(age, float)
    return np.column_stack([np.ones_like(age), age, age ** 2])


def norm_fit(X, age, min_n=30):
    X = np.asarray(X, float); age = np.asarray(age, float)
    nf = X.shape[1]; beta = np.full((nf, 3), np.nan); gamma = np.full((nf, 2), np.nan)
    usable = np.zeros(nf, bool)
    for j in range(nf):
        ok = np.isfinite(X[:, j]) & np.isfinite(age)
        if ok.sum() < min_n:
            continue
        A = _design(age[ok]); b, *_ = np.linalg.lstsq(A, X[ok, j], rcond=None)
        resid = X[ok, j] - A @ b
        L = np.column_stack([np.ones(ok.sum()), age[ok]])
        g, *_ = np.linalg.lstsq(L, np.log(np.abs(resid) + FLOOR), rcond=None)
        beta[j], gamma[j], usable[j] = b, g, True
    return {"beta": beta, "gamma": gamma, "usable": usable}


def _moments(m, age):
    A = _design(age); L = np.column_stack([np.ones(len(A)), np.asarray(age, float)])
    mu = A @ m["beta"].T
    sigma = np.exp(L @ m["gamma"].T + _LOG_ABS_TO_SD)
    return mu, np.maximum(sigma, FLOOR)


def norm_dev(m, X, age, anchor=None):
    mu, sigma = _moments(m, age)
    z = (np.asarray(X, float) - mu) / sigma
    if anchor is not None:
        z = (z - anchor["shift"]) / anchor["scale"]
    z[:, ~m["usable"]] = 0.0
    return np.nan_to_num(z)


def fit_anchor(m, Xc, agec):
    z = norm_dev(m, Xc, agec)
    shift = np.median(z, axis=0)
    mad = np.median(np.abs(z - shift), axis=0) * 1.4826
    scale = np.where(mad > FLOOR, mad, 1.0)
    return {"shift": shift, "scale": scale}


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(s, method="average"); n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_ages():
    age = {}
    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}
    for s, r in ann.items():
        if r.get("age") is not None:
            age[("CAUEEG", str(s).zfill(5))] = float(r["age"])
    if ON_TSV.exists():
        for r in csv.DictReader(open(ON_TSV, newline="", encoding="utf-8"), delimiter="\t"):
            try:
                age[("ds004504_raw", r["participant_id"].strip())] = float(r["Age"])
            except (TypeError, ValueError, KeyError):
                pass
    return age


def keyf(s, c):
    s = str(s)
    if c == "CAUEEG":
        m = re.search(r"(\d+)$", s); return ("CAUEEG", m.group(1).zfill(5)) if m else None
    if c == "ds004504_raw":
        m = re.search(r"(sub-\d+)", s); return ("ds004504_raw", m.group(1)) if m else None
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
    T = tb.subject_table("channel_clean")
    subs, coh, cls_, X = T["subjects"], T["cohort"], T["cls"], T["X_concept"]
    amap = load_ages()
    ages = np.array([amap.get(keyf(s, c), np.nan) for s, c in zip(subs, coh)], float)
    have = np.isfinite(ages)

    def fit_logit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        f = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return lambda Z: f.predict_proba(sc.transform(Z))[:, 1]

    report = {"seed": SEED, "arms": ["raw", "dev", "dev_anchor"],
              "note": "L2 logistic; normative deviation fit on controls; 80%-source-spec operating point.",
              "contrasts": {}}

    for name, pos, neg in CONTRASTS:
        cau = have & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
        ext = have & (coh == "ds004504_raw") & np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        entry = {"n_train": int(cau.sum()), "n_ext": int(ext.sum()), "n_ext_pos": int(y[ext].sum()),
                 "arms": {}}
        Xc, yc, ac = X[cau], y[cau], ages[cau]
        Xe, ye, ae = X[ext], y[ext], ages[ext]

        for arm in ("raw", "dev", "dev_anchor"):
            # ---- internal OOF ----
            oof = np.full(len(yc), np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc, yc):
                if arm == "raw":
                    Xtr, Xte = Xc[tr], Xc[te]
                else:
                    m = norm_fit(Xc[tr][yc[tr] == 0], ac[tr][yc[tr] == 0])   # controls in train fold
                    Xtr, Xte = norm_dev(m, Xc[tr], ac[tr]), norm_dev(m, Xc[te], ac[te])
                pred = fit_logit(Xtr, yc[tr])
                oof[te] = pred(Xte)
            a_int = auc(yc, oof)

            # ---- external transfer ----
            if arm == "raw":
                Xtr_all, Xte_all = Xc, Xe
            else:
                m = norm_fit(Xc[yc == 0], ac[yc == 0])                       # all source controls
                anchor = fit_anchor(m, Xe[ye == 0], ae[ye == 0]) if arm == "dev_anchor" else None
                Xtr_all = norm_dev(m, Xc, ac)
                Xte_all = norm_dev(m, Xe, ae, anchor=anchor)
            fits = [fit_logit(Xtr_all[tr], yc[tr])
                    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr_all, yc)]
            # source OOF for the threshold
            soof = np.full(len(yc), np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr_all, yc):
                soof[te] = fit_logit(Xtr_all[tr], yc[tr])(Xtr_all[te])
            thr = float(np.quantile(soof[yc == 0], 0.80))                    # 80% spec on source controls
            s_ext = np.mean([f(Xte_all) for f in fits], axis=0)
            a_ext = auc(ye, s_ext)
            tspec = float(np.mean(s_ext[ye == 0] < thr))
            tsens = float(np.mean(s_ext[ye == 1] >= thr))
            entry["arms"][arm] = {"auc_internal": round(a_int, 4), "auc_external": round(a_ext, 4),
                                  "ext_specificity_at80src": round(tspec, 3),
                                  "ext_sensitivity_at80src": round(tsens, 3),
                                  "delta_spec_vs_0.80": round(tspec - 0.80, 3)}
        report["contrasts"][name] = entry
        print(f"\n[{name}]  train {entry['n_train']}  ext {entry['n_ext']}/{entry['n_ext_pos']}")
        for arm, r in entry["arms"].items():
            print(f"   {arm:11s} AUC int {r['auc_internal']:.3f} ext {r['auc_external']:.3f}  "
                  f"| ext@80%src-spec: spec {r['ext_specificity_at80src']:.3f} "
                  f"(d{r['delta_spec_vs_0.80']:+.3f}) sens {r['ext_sensitivity_at80src']:.3f}")

    (OUT / "age_conditioned.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {OUT / 'age_conditioned.json'}")


if __name__ == "__main__":
    main()
