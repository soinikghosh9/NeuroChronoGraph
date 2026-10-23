"""Per-subject transferability check (231).

The site-distance result (229) is a property of a whole cohort: reading only healthy recordings, a
classifier tells the site apart from CAUEEG at AUC 0.96-1.00. This script asks the same question one
recording at a time, so the check can act on an individual referral rather than a whole clinic:

  * OOD distance -- for every subject, the Mahalanobis distance of its subject-median features to the
    CAUEEG *control* manifold, using the same 123-d QEEG block the models are given (node-228 block as a
    robustness check). No labels are used to build it, so it is available before any patient is scored.

  * Does distance predict error? -- within each external site, Spearman(OOD distance, per-subject
    absolute error |y - p|) for each canonical model and pooled, with a subject bootstrap 95% interval.

  * Selective prediction -- rank external subjects by OOD distance and report AUC on the recordings that
    survive as the most-OOD fraction is deferred (a risk-coverage curve). This is the actionable form:
    "abstain on the recordings the check flags, score the rest."

  * Sanity -- the subject-level OOD distance should be small inside CAUEEG and large at the external
    sites, i.e. the subject-level echo of 229.

Reads the cached features (outputs/harmonised_features) and the per-subject score shards written by 211
(outputs/results/canonical/<task>/<model>__seed42.npz). Writes ood_per_subject.json. No GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "outputs" / "harmonised_features"
CAN = ROOT / "outputs" / "results" / "canonical"

MODELS = ("spectral75k", "rf123", "rf_node", "atcnet", "eegnet", "ncg_v5", "concept")
TASK = "CN_vs_dementia"
RNG = np.random.default_rng(7)


def subject_features(tag):
    """Return {subject_id: (cohort, label, qeeg123 vector, node228 vector)} of window medians."""
    meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    g = np.array([str(x) for x in meta["groups"]])
    coh = np.array([str(x) for x in meta["cohort"]])
    lab = np.array([str(x) for x in meta["labels"]])
    N = np.load(FEAT / tag / "node12.npy", mmap_mode="r")     # (win, 19, 12)
    S = np.load(FEAT / tag / "spectral.npy", mmap_mode="r")   # (win, 123)
    out = {}
    for s in np.unique(g):
        i = np.flatnonzero(g == s)
        node = np.nanmedian(np.asarray(N[i]), 0).reshape(-1)  # 228
        spec = np.nanmedian(np.asarray(S[i]), 0)              # 123
        out[str(s)] = (coh[i[0]], lab[i[0]], np.nan_to_num(spec), np.nan_to_num(node))
    return out


def mahalanobis_scorer(ref):
    """Fit a distance-to-CAUEEG-controls scorer on a reference matrix (standardised + LedoitWolf)."""
    mu = ref.mean(0)
    sd = ref.std(0) + 1e-8
    prec = LedoitWolf().fit((ref - mu) / sd).precision_
    def dist(X):
        z = (np.atleast_2d(X) - mu) / sd
        return np.sqrt(np.einsum("ij,jk,ik->i", z, prec, z))
    return dist


def boot_spearman(x, y, n=2000):
    a = spearmanr(x, y).correlation
    b = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        c = spearmanr(x[i], y[i]).correlation
        if np.isfinite(c):
            b.append(c)
    return float(a), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def load_scores(model):
    """Per-subject (id -> (y, prob)) for internal OOF, ds004504 and the full P-ADIC CN+AD set."""
    z = np.load(CAN / TASK / f"{model}__seed42.npz", allow_pickle=True)
    def d(ids, ys, ps):
        return {str(s): (int(y), float(p)) for s, y, p in zip(z[ids], z[ys], z[ps])}
    return {"CAUEEG": d("train_subjects", "train_y", "oof_score"),
            "ds004504": d("ext_subjects", "ext_y", "ext_score"),
            "P-ADIC": d("padic_all_subjects", "padic_all_y", "padic_all_score")}


def main():
    feats = {}
    feats.update(subject_features("channel_clean"))          # CAUEEG, ds004504, GENEEG, MISP
    feats.update(subject_features("padic_channel_clean"))    # full P-ADIC
    # normalise-map so score-shard ids and feature ids meet (strip a cohort prefix if present)
    fid = {k: k for k in feats}
    norm = {}
    for k in feats:
        norm.setdefault(k, k)
        norm.setdefault(k.split("ds004504_")[-1], k)
        norm.setdefault(k.replace("ds004504_raw_", ""), k)

    def feat_of(sid):
        if sid in feats:
            return feats[sid]
        return feats.get(norm.get(sid)) if norm.get(sid) else None

    # reference manifold: CAUEEG controls (label CN)
    ref_ids = [k for k, v in feats.items() if v[0] == "CAUEEG" and v[1] == "CN"]
    ref_q = np.array([feats[k][2] for k in ref_ids])
    ref_n = np.array([feats[k][3] for k in ref_ids])
    dist_q = mahalanobis_scorer(ref_q)
    dist_n = mahalanobis_scorer(ref_n)
    print(f"reference: {len(ref_ids)} CAUEEG controls; qeeg123 d={ref_q.shape[1]}, node228 d={ref_n.shape[1]}")

    rep = {"task": TASK, "n_reference_controls": len(ref_ids), "models": list(MODELS), "sites": {}}

    # one OOD distance per subject, per site (built once, model-independent)
    site_ids = {"CAUEEG": None, "ds004504": None, "P-ADIC": None}
    any_scores = load_scores("spectral75k")
    site_dist = {}
    for site in site_ids:
        ids = [s for s in any_scores[site] if feat_of(s) is not None]
        q = np.array([feat_of(s)[2] for s in ids])
        rep.setdefault("_match", {})[site] = f"{len(ids)}/{len(any_scores[site])} matched"
        site_dist[site] = {"ids": ids, "dq": dist_q(q), "dn": dist_n(np.array([feat_of(s)[3] for s in ids]))}

    # sanity: distance distribution by site
    rep["ood_distance_qeeg123"] = {site: {"median": float(np.median(v["dq"])),
                                          "p25_p75": [float(np.percentile(v["dq"], 25)),
                                                      float(np.percentile(v["dq"], 75))]}
                                   for site, v in site_dist.items()}

    # per-model, per-site: does distance predict error, and selective prediction
    for site in ("ds004504", "P-ADIC"):
        ids = site_dist[site]["ids"]
        dq = site_dist[site]["dq"]
        order = np.argsort(-dq)                    # most OOD first
        S = {"n": len(ids), "spearman_dist_vs_abserr": {}, "selective_auc": {}, "quartile_abserr": {}}
        # decisive confound check: is distance-to-controls entangled with disease?
        y0 = np.array([load_scores("spectral75k")[site][s][0] for s in ids])
        rho_lab = spearmanr(dq, y0).correlation
        S["ood_vs_label"] = {
            "median_controls": round(float(np.median(dq[y0 == 0])), 2),
            "median_cases": round(float(np.median(dq[y0 == 1])), 2),
            "spearman_dist_vs_label": round(float(rho_lab), 3),
            "controls_only_vs_CAUEEG_controls": round(float(np.median(dq[y0 == 0])), 2)}
        pooled_d, pooled_e = [], []
        for model in MODELS:
            sc = load_scores(model)[site]
            y = np.array([sc[s][0] for s in ids])
            p = np.array([sc[s][1] for s in ids])
            err = np.abs(y - p)
            rho, ci = boot_spearman(dq, err)
            S["spearman_dist_vs_abserr"][model] = {"rho": rho, "ci95": ci}
            # selective prediction: AUC on the least-OOD kept fraction
            keep = {}
            for cov in (1.0, 0.9, 0.75, 0.5):
                k = max(int(round(cov * len(ids))), 10)
                idx = order[::-1][:k]              # keep the k least-OOD
                yy, pp = y[idx], p[idx]
                keep[f"{cov:.2f}"] = (round(float(roc_auc_score(yy, pp)), 3)
                                      if 0 < yy.sum() < len(yy) else None)
            S["selective_auc"][model] = keep
            # top vs bottom OOD quartile error
            qn = len(ids) // 4
            S["quartile_abserr"][model] = {"top_ood": round(float(err[order[:qn]].mean()), 3),
                                           "bottom_ood": round(float(err[order[-qn:]].mean()), 3)}
            pooled_d.append(dq); pooled_e.append(err)
        rho, ci = boot_spearman(np.concatenate(pooled_d), np.concatenate(pooled_e))
        S["spearman_dist_vs_abserr"]["POOLED"] = {"rho": rho, "ci95": ci}
        rep["sites"][site] = S

    (CAN / "ood_per_subject.json").write_text(json.dumps(rep, indent=1))
    print("\n== OOD distance (qeeg123) by site ==")
    for site, v in rep["ood_distance_qeeg123"].items():
        print(f"  {site:9s} median {v['median']:.2f}  IQR {v['p25_p75'][0]:.2f}-{v['p25_p75'][1]:.2f}  ({rep['_match'][site]})")
    for site in ("ds004504", "P-ADIC"):
        S = rep["sites"][site]
        pl = S["spearman_dist_vs_abserr"]["POOLED"]
        s75 = S["spearman_dist_vs_abserr"]["spectral75k"]
        sel = S["selective_auc"]["spectral75k"]
        print(f"\n== {site} (n={S['n']}) ==")
        print(f"  Spearman(OOD, |error|) pooled {pl['rho']:+.3f} {pl['ci95']}  SpectraLite {s75['rho']:+.3f} {s75['ci95']}")
        ol = S["ood_vs_label"]
        print(f"  OOD by label: controls {ol['median_controls']} vs cases {ol['median_cases']}  "
              f"Spearman(OOD, label) {ol['spearman_dist_vs_label']:+.3f}")
        print(f"  SpectraLite selective AUC by coverage: {sel}")
        print(f"  SpectraLite quartile |err| top-OOD {S['quartile_abserr']['spectral75k']['top_ood']} "
              f"vs bottom-OOD {S['quartile_abserr']['spectral75k']['bottom_ood']}")
    print("\nwrote", CAN / "ood_per_subject.json")


if __name__ == "__main__":
    main()
