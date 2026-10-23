"""A continuous, exploratory transferability index from healthy recordings.

The site-distance of 229 is a decodability AUC, which saturates near 1.0 at every external site (0.96,
0.99, 0.998, 1.00) and so ranks sites only coarsely. Here we replace it with continuous divergences between
a receiving site's control distribution and the development (CAUEEG) control distribution, computed on the
same 123-d QEEG features, using controls only:

  * MMD    -- squared maximum mean discrepancy, RBF kernel, median-heuristic bandwidth (non-parametric);
  * Maha   -- Mahalanobis distance between the control means under the CAUEEG-control covariance;
  * energy -- energy distance.

Each is related to the mean external AUC at that site. For the three dementia sites, the script also
reports the matched-model transfer loss (internal AUC minus external AUC). The leave-one-site-out fit is
descriptive because only four cohorts are available and GENEEG uses a different clinical contrast.

    python experiments/238_transferability_index.py
Out: outputs/results/canonical/transferability_index.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
FEAT, CAN = C.FEAT, C.OUT
CONTROL = {"CN", "C", "CONTROL"}
# (site key, feature tag, cohort in that tag, task + evaluation-set for its transfer AUC)
SITES = [("ds004504", "channel_clean", "ds004504_raw", ("CN_vs_dementia", "external")),
         ("P-ADIC", "padic_channel_clean", "P-ADIC", ("CN_vs_dementia", "padic")),
         ("BrainLat", "brainlat_channel_clean", "BrainLat", ("CN_vs_dementia", "brainlat")),
         ("GENEEG", "channel_clean", "GENEEG", ("CN_vs_MCI", "external"))]


def controls(tag, cohort):
    meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    g = np.array([str(x) for x in meta["groups"]]); coh = np.array([str(x) for x in meta["cohort"]])
    lab = np.array([str(x) for x in meta["labels"]])
    S = np.load(FEAT / tag / "spectral.npy", mmap_mode="r")
    rows = []
    for s in np.unique(g):
        i = np.flatnonzero(g == s)
        if coh[i[0]] == cohort and lab[i[0]] in CONTROL:
            rows.append(np.nan_to_num(np.nanmedian(np.asarray(S[i]), 0)))
    return np.array(rows)


def mmd2(X, Y):
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sigma2 = np.median(d2[d2 > 0]) or 1.0
    k = np.exp(-d2 / sigma2)
    nx, ny = len(X), len(Y)
    kxx = k[:nx, :nx]; kyy = k[nx:, nx:]; kxy = k[:nx, nx:]
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def energy_dist(X, Y):
    def m(A, B):
        return np.sqrt(np.sum((A[:, None] - B[None]) ** 2, -1)).mean()
    return float(2 * m(X, Y) - m(X, X) - m(Y, Y))


def site_metrics(task, set_name):
    """Return matched internal AUC, external AUC and transfer loss for one site.

    Ladder rows are excluded because they are additive ablations, not independent entries in the
    capacity benchmark. Keeping the paired model set explicit prevents mean external AUC from being
    mislabeled as transfer loss.
    """
    tasks = C_summary["tasks"].get(task, {})
    internal = tasks.get("evaluation_sets", {}).get("internal", {})
    external = tasks.get("evaluation_sets", {}).get(set_name, {})
    common = [m for m in external if m in internal and not m.startswith("ladder")]
    if not common:
        return None
    a_int = float(np.mean([internal[m]["auc"] for m in common]))
    a_ext = float(np.mean([external[m]["auc"] for m in common]))
    return {"n_models": len(common), "mean_internal_auc": a_int,
            "mean_external_auc": a_ext, "mean_transfer_loss": a_int - a_ext}


C_summary = json.loads((CAN / "summary.json").read_text())


def main():
    cau = controls("channel_clean", "CAUEEG")
    mu, sd = cau.mean(0), cau.std(0) + 1e-8
    Zc = (cau - mu) / sd
    prec = LedoitWolf().fit(Zc).precision_
    rep = {"n_caueeg_controls": len(cau), "sites": {}}
    for name, tag, cohort, (task, sset) in SITES:
        Xs = controls(tag, cohort)
        if len(Xs) < 5:
            continue
        Z = (Xs - mu) / sd
        dmu = Z.mean(0)
        metrics = site_metrics(task, sset)
        rep["sites"][name] = {
            "n_controls": len(Xs),
            "mmd": mmd2(Zc, Z),
            "mahalanobis": float(np.sqrt(dmu @ prec @ dmu)),
            "energy": energy_dist(Zc, Z),
            "decodability_auc": (json.loads((CAN / "site_shift.json").read_text())["sites"]
                                 .get({"ds004504": "ds004504_raw"}.get(name, name), {})
                                 .get("decodability_auc", {}).get("logistic_qeeg123")),
            **(metrics or {})}
    # relate each distance to transfer AUC, and leave-one-site-out predict the AUC
    names = [n for n in rep["sites"] if rep["sites"][n].get("mean_external_auc") is not None]
    y = np.array([rep["sites"][n]["mean_external_auc"] for n in names])
    print(f"sites: {names}")
    for metric in ("mmd", "mahalanobis", "energy", "decodability_auc"):
        x = np.array([rep["sites"][n][metric] for n in names], float)
        if np.any(~np.isfinite(x)):
            continue
        pr = pearsonr(x, y); sr = spearmanr(x, y)
        loo = {}
        if len(names) >= 3:
            for i, n in enumerate(names):
                tr = [j for j in range(len(names)) if j != i]
                lr = LinearRegression().fit(x[tr, None], y[tr])
                loo[n] = round(float(lr.predict(x[i:i + 1, None])[0]), 3)
        rep.setdefault("relation", {})[metric] = {
            "values": {n: round(float(v), 4) for n, v in zip(names, x)},
            "pearson_r_vs_auc": round(float(pr.statistic), 3), "pearson_p": round(float(pr.pvalue), 3),
            "spearman_rho": round(float(sr.correlation), 3),
            "loso_predicted_auc": loo, "actual_auc": {n: round(float(v), 3) for n, v in zip(names, y)}}
        print(f"  {metric:16s} vals {[round(v,3) for v in x]}  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.3f}) "
              f"Spearman={sr.correlation:+.2f}")
        if loo:
            print(f"      LOSO predicted vs actual: " +
              ", ".join(f"{n} {loo[n]:.3f}/{y[i]:.3f}" for i, n in enumerate(names)))
    # Transfer loss is comparable across the three dementia sites. GENEEG is a separate MCI task,
    # so it remains in the descriptive external-AUC relation but is excluded here.
    dementia_names = [n for n in ("ds004504", "P-ADIC", "BrainLat") if n in rep["sites"]]
    if len(dementia_names) == 3:
        for metric in ("mmd", "mahalanobis", "energy", "decodability_auc"):
            x = np.array([rep["sites"][n][metric] for n in dementia_names], float)
            y_loss = np.array([rep["sites"][n]["mean_transfer_loss"] for n in dementia_names], float)
            sr_loss = spearmanr(x, y_loss)
            rep.setdefault("relation", {}).setdefault(metric, {})["dementia_transfer_loss"] = {
                "sites": dementia_names,
                "actual_loss": {n: round(float(v), 3) for n, v in zip(dementia_names, y_loss)},
                "spearman_rho": round(float(sr_loss.correlation), 3),
                "p": round(float(sr_loss.pvalue), 3)}
    (CAN / "transferability_index.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", CAN / "transferability_index.json")


if __name__ == "__main__":
    main()
