"""Learning curves: is "at these sample sizes" true? AUC vs number of training subjects.

Review W2: the paper's thesis is regime-specific ("at the sample sizes the field has, capacity does
not pay") but training size was never varied. Here the CAUEEG task cohort is split once into a fixed
internal test set (20% of subjects, stratified, seed 7) and a training pool (80%). Each model is
trained on a stratified fraction f of the pool, f in {0.1, 0.25, 0.5, 0.75, 1.0}, three repeats
(different subsample AND training seed), with the unchanged fixed recipe (25 epochs, final weights),
as a SINGLE model (no five-fold ensemble -- the curve compares slopes, not the deployed ensemble).
Both the fixed internal test set and the external site are scored.

If high-capacity networks were sample-limited, their curves would still be rising at f=1.0 and
steeper than the spectral models'. If all curves flatten together, the ceiling is information-, not
estimator-limited at this scale.

    python experiments/214_learning_curve.py --tasks CN_vs_dementia,CN_vs_impaired
Out: outputs/results/canonical/learning_curve/<task>.json (+ per-run npz)
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

OUT = C.OUT / "learning_curve"
FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
REPEATS = (0, 1, 2)
if os.environ.get("REV_EPOCHS"):                                         # smoke test only
    FRACS, REPEATS = (0.1, 1.0), (0,)
DEEP = ("spectral75k", "eegnet", "shallow", "deep4", "atcnet", "ncg_v5", "conformer")
CLASSICAL = ("rf123", "rf_node")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--tasks", default="CN_vs_dementia,CN_vs_impaired")
    ap.add_argument("--models", default=",".join(CLASSICAL + DEEP))
    args = ap.parse_args()

    import torch
    C.set_determinism(torch)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    e161 = C.load_161()
    device = "cuda"
    assert torch.cuda.is_available()

    spec2 = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tb)
    T = tb.subject_table(args.tag)
    subs_T = np.asarray(T["subjects"]); Xq = T["X_spectral"]
    F = np.load(C.FEAT / args.tag / "node12.npy", mmap_mode="r")

    Xw = np.load(C.windows_path(args.tag), mmap_mode="r")
    meta = np.load(C.FEAT / args.tag / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    win_cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    Xn_all = {s: np.nanmedian(np.asarray(F[groups == s]).reshape(-1, 228), axis=0) for s in subs_T}
    OUT.mkdir(parents=True, exist_ok=True)

    for task in [t for t in args.tasks.split(",") if t]:
        pos, neg, site = C.TASKS[task]
        keep = np.isin(win_cls, pos + neg); y = np.isin(win_cls, pos).astype(int)
        cau = np.unique(groups[keep & (win_coh == "CAUEEG")])
        ext = np.unique(groups[keep & (win_coh == site)])
        cau_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in cau])
        ext_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in ext])
        pool_i, test_i = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=7).split(cau, cau_y))
        pool, pool_y, itest, itest_y = cau[pool_i], cau_y[pool_i], cau[test_i], cau_y[test_i]
        res_path = OUT / f"{task}.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {
            "task": task, "n_pool": int(len(pool)), "n_internal_test": int(len(itest)),
            "n_external": int(len(ext)), "ext_site": site, "rows": []}
        done = {(r["model"], r["frac"], r["repeat"]) for r in res["rows"]}
        need_deep = any((m, f, r) not in done for m in args.models.split(",") if m in DEEP
                        for f in FRACS for r in REPEATS)
        if need_deep:
            m_eval = np.isin(groups, np.concatenate([itest, ext])) & keep
            idx_eval = np.flatnonzero(m_eval)
            X_eval = np.asarray(Xw[idx_eval]); g_eval = groups[idx_eval]
            is_int = np.isin(g_eval, itest)
            idx_pool = np.flatnonzero(np.isin(groups, pool) & keep)
            X_pool = np.asarray(Xw[idx_pool]); g_pool = groups[idx_pool]; y_pool = y[idx_pool]
        print(f"\n=== {task}: pool {len(pool)} | internal test {len(itest)} | {site} {len(ext)}", flush=True)
        for frac in FRACS:
            for rep in REPEATS:
                if frac >= 1.0:
                    sub = pool
                else:
                    sel, _ = next(StratifiedShuffleSplit(1, train_size=frac, random_state=1000 + rep)
                                  .split(pool, pool_y))
                    sub = pool[sel]
                for name in [m for m in args.models.split(",") if m]:
                    if (name, frac, rep) in done:
                        continue
                    t0 = time.time(); seed = 42 + rep
                    if name in CLASSICAL:
                        Xsrc = (lambda ss: Xq[np.searchsorted(subs_T, ss)]) if name == "rf123" else \
                               (lambda ss: np.stack([Xn_all[s] for s in ss]))
                        ysub = np.array([y[np.flatnonzero(groups == s)[0]] for s in sub])
                        f = RandomForestClassifier(400, random_state=seed, n_jobs=6,
                                                   class_weight="balanced").fit(Xsrc(sub), ysub)
                        s_int = f.predict_proba(Xsrc(itest))[:, 1]; s_ext = f.predict_proba(Xsrc(ext))[:, 1]
                    else:
                        mfit = np.isin(g_pool, sub)
                        model, _ = C.train_one(torch, name, X_pool[mfit], y_pool[mfit], device, seed, e161)
                        p = C.predict(torch, model, X_eval, device, e161)
                        del model; torch.cuda.empty_cache()
                        s_int = C.subject_mean(p[is_int], g_eval[is_int], itest)
                        s_ext = C.subject_mean(p[~is_int], g_eval[~is_int], ext)
                    row = {"model": name, "frac": frac, "repeat": rep, "n_train": int(len(sub)),
                           "auc_internal_test": C.auc(itest_y, s_int), "auc_external": C.auc(ext_y, s_ext),
                           "minutes": round((time.time() - t0) / 60, 2)}
                    res["rows"].append(row); done.add((name, frac, rep))
                    np.savez_compressed(OUT / f"{task}__{name}__f{frac}__r{rep}.npz", itest=itest,
                                        itest_y=itest_y, s_int=s_int, ext=ext, ext_y=ext_y, s_ext=s_ext)
                    res_path.write_text(json.dumps(res, indent=1))
                    print(f"   f={frac:<4} r={rep} n={len(sub):4d} {name:12s} internal "
                          f"{row['auc_internal_test']:.3f} external {row['auc_external']:.3f} "
                          f"({row['minutes']} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
