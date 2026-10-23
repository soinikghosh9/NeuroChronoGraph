"""Analysis of the SpectraLite improvement experiments, exactly as fixed in
model_paper/notes/IMPROVEMENT_PROTOCOL_2026-10-02.md (written before the runs).

  H1 (primary)  sl16_mc vs sl16 at the held-out cohort, paired DeLong; dementia and AD x 3 cohorts, Holm over 6
  H2            f16_mc vs f16, same family structure, Holm over 6
  H3            nested leave-one-cohort-out selected R1 model vs sl12, Holm over 6
  Selection     nested LOCO: for cohort X, the R1 candidate with the highest mean AUC over the other two external
                cohorts (same task); internal selection: highest CAUEEG out-of-fold AUC
Every candidate is reported at every cohort. The any-impairment task is reported descriptively.

Output: outputs/results/canonical/improvement_summary.json

    python experiments/251_improvement_summary.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CAN = ROOT / "outputs" / "results" / "canonical"
IMP = CAN / "improvement"
OUT = CAN / "improvement_summary.json"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
EXT = (("ext", "ds004504"), ("padic", "P-ADIC"), ("brainlat", "BrainLat"))
R1 = ("sl12", "sl16", "sl16_dr", "sl16_tuned", "f16", "ens")
BOOT = 4000

spec = importlib.util.spec_from_file_location("cm250s", ROOT / "experiments" / "250_clinical_metrics.py")
CM = importlib.util.module_from_spec(spec); spec.loader.exec_module(CM)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def delong_paired(y, a, b):
    y = np.asarray(y).astype(int)

    def comps(s):
        pos, neg = s[y == 1], s[y == 0]
        return (np.array([((p > neg) + 0.5 * (p == neg)).mean() for p in pos]),
                np.array([((pos > n) + 0.5 * (pos == n)).mean() for n in neg]))
    a10, a01 = comps(a); b10, b01 = comps(b)
    S = np.cov(np.vstack([a10, b10])) / len(a10) + np.cov(np.vstack([a01, b01])) / len(a01)
    d = CM.auc(y, a) - CM.auc(y, b); se = float(np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12)))
    return float(d), float(2 * norm.sf(abs(d / se)))


def boot_ci(y, a, b, seed):
    rng = np.random.default_rng(seed); y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); out = []
    for _ in range(BOOT):
        i = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
        out.append(CM.auc(y[i], a[i]) - CM.auc(y[i], b[i]))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def scores(task):
    """candidate -> {'internal': (y, s), 'ext'/'padic'/'brainlat': (subj, y, s)}"""
    def from_files(fs):
        E = CM.ensemble(fs); d = {"internal": (E["train_y"].astype(int), E["oof"]), "n_seeds": E["n_seeds"]}
        for p, _ in EXT:
            if p in E:
                d[p] = E[p]
        return d
    out = {"sl16": from_files(sorted((CAN / task).glob("spectralite16__seed*.npz"))),
           "sl12": from_files(sorted((CAN / task).glob("spectral75k__seed*.npz")))}
    for c in ("sl16_dr", "sl16_tuned", "f16"):
        fs = sorted((IMP / task).glob(f"{c}__seed*.npz"))
        if fs:
            out[c] = from_files(fs)
    if "f16" in out:
        a, b = out["sl16"], out["f16"]; e = {"internal": (a["internal"][0], (a["internal"][1] + b["internal"][1]) / 2)}
        for p, _ in EXT:
            assert np.array_equal(a[p][0], b[p][0])
            e[p] = (a[p][0], a[p][1], (a[p][2] + b[p][2]) / 2)
        out["ens"] = e
    for c in ("sl16_mc", "f16_mc"):
        d = {}
        for p, nm in EXT:
            fs = sorted((IMP / task).glob(f"{c}__holdout-{nm}__seed*.npz"))
            if fs:
                E = CM.ensemble(fs)
                d[p] = E[p]; d.setdefault("internal_by_holdout", {})[nm] = CM.auc(E["train_y"], E["oof"])
        if d:
            out[c] = d
    return out


def main():
    rep = {"protocol": "model_paper/notes/IMPROVEMENT_PROTOCOL_2026-10-02.md", "tasks": {}, "tests": {}}
    fam = {"H1": [], "H2": [], "H3": []}
    for task in TASKS:
        S = scores(task); T = rep["tasks"][task] = {"auc": {}, "nested_selection": {}, "internal_selection": None}
        for c, d in S.items():
            T["auc"][c] = {}
            if "internal" in d:
                T["auc"][c]["CAUEEG"] = CM.auc(*d["internal"])
            for p, nm in EXT:
                if p in d:
                    T["auc"][c][nm] = CM.auc(d[p][1], d[p][2])
            if "internal_by_holdout" in d:
                T["auc"][c]["CAUEEG_by_holdout"] = d["internal_by_holdout"]
        r1 = [c for c in R1 if c in S]
        T["internal_selection"] = max(r1, key=lambda c: T["auc"][c]["CAUEEG"])
        for p, nm in EXT:
            others = [n for _, n in EXT if n != nm]
            pick = max(r1, key=lambda c: np.mean([T["auc"][c][o] for o in others]))
            subj, y, _ = S["sl16"][p]
            T["nested_selection"][nm] = {"selected": pick, "auc": T["auc"][pick][nm],
                                         "mean_other_two": float(np.mean([T["auc"][pick][o] for o in others]))}
            if task in ("CN_vs_dementia", "CN_vs_AD"):
                for h, a, b in (("H1", "sl16_mc", "sl16"), ("H2", "f16_mc", "f16"), ("H3", pick, "sl12")):
                    if a in S and b in S and p in S[a]:
                        assert np.array_equal(S[a][p][0], S[b][p][0])
                        d, pv = delong_paired(y, S[a][p][2], S[b][p][2])
                        fam[h].append({"task": task, "cohort": nm, "a": a, "b": b, "n": int(len(y)),
                                       "auc_a": CM.auc(y, S[a][p][2]), "auc_b": CM.auc(y, S[b][p][2]),
                                       "delta": d, "ci95_boot": boot_ci(y, S[a][p][2], S[b][p][2], seed=251 + len(fam[h])),
                                       "p_delong": pv})
    for h, rows in fam.items():
        for r, ph in zip(rows, holm([r["p_delong"] for r in rows])):
            r["p_holm"] = ph
        rep["tests"][h] = rows
    OUT.write_text(json.dumps(rep, indent=1))
    for task, T in rep["tasks"].items():
        print(f"\n== {task}  (internal selection: {T['internal_selection']})")
        for c, v in T["auc"].items():
            print(f"   {c:11s} " + "  ".join(f"{k} {v[k]:.3f}" for k in ("CAUEEG", "ds004504", "P-ADIC", "BrainLat") if k in v))
        for nm, s in T["nested_selection"].items():
            print(f"   nested LOCO -> {nm}: {s['selected']} ({s['auc']:.3f})")
    for h, rows in rep["tests"].items():
        print(f"\n{h}")
        for r in rows:
            print(f"   {r['task']:15s} {r['cohort']:9s} {r['a']} vs {r['b']}: {r['auc_a']:.3f} vs {r['auc_b']:.3f} "
                  f"d={r['delta']:+.3f} [{r['ci95_boot'][0]:+.3f},{r['ci95_boot'][1]:+.3f}] p={r['p_delong']:.3f} holm={r['p_holm']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
