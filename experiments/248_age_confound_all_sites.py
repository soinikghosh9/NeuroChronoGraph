"""Age as a competing explanation at every cohort that records age.

For the released SpectraLite (16 features), its pre-specified 12-feature base (SpectraLite-12) and the
QEEG reference forest, this script reports, per task and cohort:

  * the AUC of age alone (higher age = higher score), on the full evaluation set;
  * the model's AUC on age-matched control/case pairs (greedy nearest-age matching within a 3-year
    caliper, controls taken in age order; the same rule as 243_age_norm_spectralite16.py), together
    with the AUC of age alone on those pairs and a pair-bootstrap 95% interval for the model AUC.

Cohorts: CAUEEG (internal out-of-fold), ds004504, P-ADIC (age-restricted evaluation set) and BrainLat.
GENEEG releases no per-subject age. Scores are seed ensembles (mean over the saved seeds).
Output: outputs/results/canonical/age_confound_all_sites.json, plus the per-subject ages used for the
supplementary age-distribution figure.

    python experiments/248_age_confound_all_sites.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "age_confound_all_sites.json"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
MODELS = {"spectralite16": "SpectraLite", "spectral75k": "SpectraLite-12", "rf123": "Forest (QEEG-123)"}
# Added 2026-10-14 for Fig. 2c: the NeuroChronoGraph version with the highest development AUC. It draws from its own
# random stream so the bootstrap intervals of MODELS are unchanged.
EXTRA = {"ncg_v3": "NCG-v3"}
SITES = (("CAUEEG", "train_subjects", "train_y", "oof_score"),
         ("ds004504", "ext_subjects", "ext_y", "ext_score"),
         ("P-ADIC", "padic_subjects", "padic_y", "padic_score"),
         ("BrainLat", "brainlat_subjects", "brainlat_y", "brainlat_score"))
BOOT = 4000


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def auc(y, s):
    """Mann-Whitney AUC with ties counted as one half (mid-ranks)."""
    y = np.asarray(y).astype(int); r = rankdata(s, method="average")
    n1 = int(y.sum()); n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def age_maps():
    a204 = _module("age204_for248", "204_age_conditioned.py")
    dev = a204.load_ages()
    padic = {k: float(v["age"]) for k, v in json.loads((CAN / "padic_subject_meta.json").read_text()).items()
             if v.get("age") is not None}
    blat = {}
    for k, v in json.loads((CAN / "brainlat_subject_meta.json").read_text()).items():
        try:
            blat[k] = float(v["age"])
        except (TypeError, ValueError, KeyError):
            pass

    def get(subject, site):
        if site == "CAUEEG":
            return dev.get(a204.keyf(subject, "CAUEEG"), np.nan)
        if site == "ds004504":
            return dev.get(a204.keyf(subject, "ds004504_raw"), np.nan)
        if site == "P-ADIC":
            return padic.get(str(subject), np.nan)
        return blat.get(str(subject), np.nan)
    return get


def match(y, ages, caliper=3.0):
    """Greedy nearest-age matching, controls in age order (identical to script 243)."""
    controls = [i for i in range(len(y)) if y[i] == 0]
    cases = [i for i in range(len(y)) if y[i] == 1]
    used, pairs = set(), []
    for ci in sorted(controls, key=lambda i: ages[i]):
        avail = [j for j in cases if j not in used]
        if not avail:
            break
        d = np.abs(ages[avail] - ages[ci]); k = int(np.argmin(d))
        if d[k] <= caliper:
            used.add(avail[k]); pairs.append((ci, avail[k]))
    return pairs


def ensemble(task, model, site_keys):
    files = sorted((CAN / task).glob(f"{model}__seed*.npz"))
    if not files:
        return None
    subj, y, s = None, None, []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if site_keys[3] not in z.files:
            return None
        subj, y = z[site_keys[1]], z[site_keys[2]]; s.append(z[site_keys[3]])
    return np.asarray([str(v) for v in subj]), np.asarray(y).astype(int), np.mean(s, 0), len(files)


def main():
    get_age = age_maps(); rng = np.random.default_rng(248); rng_x = np.random.default_rng(2481)
    out = {"caliper_years": 3.0, "bootstrap": BOOT, "models": {**MODELS, **EXTRA}, "tasks": {}, "ages_by_class": {}}
    for task in TASKS:
        out["tasks"][task] = {}
        for site_keys in SITES:
            site = site_keys[0]; entry = {}
            for model, label in {**MODELS, **EXTRA}.items():
                r = rng if model in MODELS else rng_x
                e = ensemble(task, model, site_keys)
                if e is None:
                    continue
                subj, y, s, nseed = e
                ages = np.asarray([get_age(v, site) for v in subj], float)
                ok = np.isfinite(ages)
                if "n" not in entry:
                    entry.update({"n": int(y.size), "n_with_age": int(ok.sum()),
                                  "auc_age_alone": auc(y[ok], ages[ok])})
                    pairs = match(y[ok], ages[ok])
                    entry["matched_pairs"] = len(pairs)
                    if pairs:
                        ids = np.ravel(np.asarray(pairs))
                        entry["matched_auc_age_alone"] = auc(y[ok][ids], ages[ok][ids])
                        entry["matched_mean_age"] = {"control": float(ages[ok][ids][y[ok][ids] == 0].mean()),
                                                     "case": float(ages[ok][ids][y[ok][ids] == 1].mean())}
                    if task == "CN_vs_dementia":
                        out["ages_by_class"][site] = {"control": ages[ok][y[ok] == 0].tolist(),
                                                      "case": ages[ok][y[ok] == 1].tolist()}
                res = {"seeds": nseed, "auc_full": auc(y, s), "auc_full_with_age": auc(y[ok], s[ok])}
                pairs = match(y[ok], ages[ok])
                if len(pairs) >= 8:
                    P = np.asarray(pairs); yy, ss = y[ok], s[ok]
                    ids = np.ravel(P)
                    res["auc_matched"] = auc(yy[ids], ss[ids])
                    bs = []
                    for _ in range(BOOT):
                        b = np.ravel(P[r.integers(0, len(P), len(P))])
                        bs.append(auc(yy[b], ss[b]))
                    res["auc_matched_ci95"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
                entry[model] = res
            out["tasks"][task][site] = entry
            sl = entry.get("spectralite16", {})
            print(f"{task:15s} {site:9s} n={entry.get('n')} age-alone {entry.get('auc_age_alone', np.nan):.3f} "
                  f"| pairs {entry.get('matched_pairs')} age {entry.get('matched_auc_age_alone', np.nan):.3f} "
                  f"SL {sl.get('auc_full', np.nan):.3f} -> matched {sl.get('auc_matched', np.nan):.3f} "
                  f"{sl.get('auc_matched_ci95')}")
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
