"""All ten deep architectures and the classical models, trained per task as binary contrasts.

The three-class models cannot answer an Alzheimer's-versus-control question: their dementia class
folds Alzheimer's together with frontotemporal disease, so scoring them on that contrast asks a
model trained to say "dementia" to say "Alzheimer's". Restoring a four-class scheme is not the fix
either, because frontotemporal disease has twelve training subjects and was folded away for that
reason.

Each task is therefore trained as its own binary contrast, on the classes it concerns and no
others, which is what the concept models in 157 already do. Frontotemporal subjects enter only the
tasks whose positive class includes them; the Alzheimer's task excludes them from training and
evaluation entirely, so its 0.853 and its 0.816 are answers to different questions rather than the
same posterior read twice.

    python experiments/161_deep_per_task.py
Out: outputs/results/audit/deep_per_task.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "outputs" / "results" / "audit"
PRED = OUT / "predictions_per_task"
FEAT = ROOT / "outputs" / "harmonised_features"
WIN = ROOT / "outputs" / "harmonised_windows"
WINDOW, SEED, N_BOOT = 1024, 42, 2000

#: Training seed, set per run by --seed. The FOLD SPLIT stays at SEED so that varying this changes
#: initialisation and batch order only, with subjects held fixed. That is the decomposition the
#: retraining-variance estimate needs: one null varies subjects and fixes models, the other varies
#: training and fixes subjects, and mixing them measures neither.
TRAIN_SEED = SEED
EPOCHS, BATCH, LR = 25, 128, 3e-3

CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD",
         "CN": "CN", "C": "CN", "CONTROL": "CN", "MCI": "MCI"}

DEEP = ("shallow", "deep4", "eegnet", "conformer", "atcnet", "eegnex",
        "eegitnet", "ncg_v5", "ncg_v2", "ncg_v3")

TASKS = {
    "CN_vs_AD":       (("AD",),              ("CN",), "ds004504_raw"),
    "CN_vs_dementia": (("AD", "FTD"),        ("CN",), "ds004504_raw"),
    "CN_vs_MCI":      (("MCI",),             ("CN",), "GENEEG"),
    "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",), "ds004504_raw"),
}


def operating(y, s, thr):
    """Counts and rates at one threshold."""
    pred = (np.asarray(s) >= thr).astype(int)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) or 1.0
    mcc = (tp * tn - fp * fn) / den
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "balanced_accuracy": 0.5 * (sens + spec), "f1": f1, "mcc": mcc}


def ece(y, s, bins=10):
    """Expected calibration error over equal-count bins."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    order = np.argsort(s)
    e = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        e += len(chunk) / len(y) * abs(s[chunk].mean() - y[chunk].mean())
    return float(e)


def metrics(y, s, thr_prev, n_boot=N_BOOT, seed=SEED):
    """AUC with interval, average precision, calibration, and rates at two thresholds."""
    from _protocol_v2 import safe_auc
    from sklearn.metrics import average_precision_score
    a = safe_auc(y, s)
    if a is None:
        return None
    y, s, n = np.asarray(y), np.asarray(s), len(y)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v = safe_auc(y[i], s[i])
        if v is not None:
            out.append(v)
    thr_src = float(np.quantile(s, 1 - thr_prev))
    best, thr_j = -1.0, thr_src
    for t in np.unique(s):
        o = operating(y, s, t)
        j = o["sensitivity"] + o["specificity"]
        if j > best:
            best, thr_j = j, float(t)
    return {"auc": a,
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n": int(n), "n_pos": int(y.sum()),
            "average_precision": float(average_precision_score(y, s)),
            "brier": float(np.mean((s - y) ** 2)),
            "ece": ece(y, s),
            "at_source_prevalence": operating(y, s, thr_src),
            "at_youden": operating(y, s, thr_j)}


def make_binary(name, device):
    """Same architectures as the reported benchmark, with a two-way head."""
    import braindecode.models as M
    kw = dict(n_chans=19, n_outputs=2, n_times=WINDOW)
    if name == "shallow":
        m = M.ShallowFBCSPNet(**kw, final_conv_length="auto")
    elif name == "deep4":
        m = M.Deep4Net(**kw, final_conv_length="auto")
    elif name == "eegnet":
        m = M.EEGNetv4(**kw)
    elif name == "conformer":
        m = M.EEGConformer(**kw)
    elif name == "atcnet":
        m = M.ATCNet(**kw)
    elif name == "eegnex":
        m = M.EEGNeX(**kw)
    elif name == "eegitnet":
        m = M.EEGITNet(**kw)
    elif name == "ncg_v5":
        from src.models.v5.neuro_chrono_graph_v5 import NeuroChronoGraphV5
        m = NeuroChronoGraphV5(n_channels=19, n_classes=2)
    elif name == "ncg_v2":
        from src.models.v2.neuro_chrono_graph_v2 import NeuroChronoGraphV2
        m = NeuroChronoGraphV2(n_channels=19, n_classes=2)
    elif name == "ncg_v3":
        from src.models.v3.neuro_chrono_graph_v3 import NeuroChronoGraphV3
        m = NeuroChronoGraphV3(n_channels=19, n_classes=2)
    else:
        raise ValueError(name)
    return m.to(device)


def logits(model, x):
    """The class logits, by name. The hierarchical v2 head exposes logits_screen rather than
    logits, and every model here also returns adjacency and attention tensors; taking the first
    dict entry silently trains on one of those, so the key is looked up explicitly and a model
    exposing none of them raises rather than proceeding."""
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        for k in ("logits", "logits_screen"):
            if k in out:
                return out[k]
        raise KeyError("no logits in model output: %s" % list(out))
    return out


def train_eval(name, X, y, groups, tr_sub, te_sub, device):
    """Subject-grouped five-fold on the training cohort; fold ensemble applied to the test site."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset

    sub_y = np.array([y[groups == s][0] for s in tr_sub])
    te_mask = np.isin(groups, te_sub)
    Xte = torch.from_numpy(np.ascontiguousarray(X[te_mask]))
    preds = []
    for a, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr_sub, sub_y):
        fit_mask = np.isin(groups, tr_sub[a])
        Xtr = torch.from_numpy(np.ascontiguousarray(X[fit_mask]))
        ytr = torch.from_numpy(y[fit_mask].astype(np.int64))
        torch.manual_seed(TRAIN_SEED)
        model = make_binary(name, device)
        cls, cnt = np.unique(y[fit_mask], return_counts=True)
        w = np.zeros(2, dtype=np.float32)
        w[cls] = len(ytr) / (len(cls) * cnt)
        crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=EPOCHS * max(1, len(ytr) // BATCH + 1))
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, drop_last=False)
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(logits(model, xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
                try:
                    sched.step()
                except Exception:                                          # noqa: BLE001
                    pass
        model.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(Xte), 256):
                p.append(torch.softmax(logits(model, Xte[i:i + 256].to(device)), 1)[:, 1].cpu().numpy())
        preds.append(np.concatenate(p))
        del model
        torch.cuda.empty_cache()
    win = np.mean(preds, axis=0)
    g_te = groups[te_mask]
    subs = np.unique(g_te)
    return subs, np.array([win[g_te == s].mean() for s in subs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--models", default=",".join(DEEP))
    ap.add_argument("--merge", action="store_true",
                    help="add these models to the existing artefact instead of replacing it")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training seed: model initialisation and batch order. The subject "
                         "fold split is held at 42 regardless, so repeated runs at different "
                         "seeds estimate retraining variance with subjects fixed. A seed "
                         "other than 42 writes to its own artefact and never overwrites the "
                         "reported one.")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma-separated subset of tasks, so the prodromal column can be "
                         "reseeded without repeating a 17-hour battery")
    args = ap.parse_args()

    global TRAIN_SEED
    TRAIN_SEED = args.seed
    want_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in want_tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; choose from {list(TASKS)}")

    import torch
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.utils.provenance import write_result
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

    PRED.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    d = FEAT / args.tag
    wp = WIN / args.tag / "windows.npy"
    Xw = np.load(wp, mmap_mode="r") if wp.exists() else None
    if Xw is None:
        raise SystemExit("windows.npy not found for tag %s; run 131 once to materialise it" % args.tag)
    meta = np.load(d / "meta.npz", allow_pickle=False)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]
    win_cls = np.array([CLASS[str(x).upper()] for x in labels])
    win_coh = np.array([str(x) for x in cohort])

    T = tb.subject_table(args.tag)          # subject-level concepts and spectral block
    report = {"tag": args.tag, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
              "training": "per-task binary", "tasks": {}}
    # Re-running a single architecture must not discard the others: an earlier rerun would have
    # overwritten a completed nine-hour run with one model.
    stem = "deep_per_task" if args.seed == SEED else f"deep_per_task_seed{args.seed}"
    dest = OUT / f"{stem}.json"
    report["train_seed"] = args.seed
    report["split_seed"] = SEED
    if args.merge and dest.exists():
        import json as _j
        prev = _j.loads(dest.read_text())
        report["tasks"] = prev.get("tasks", {})
        print("merging into existing results (%d tasks)" % len(report["tasks"]), flush=True)

    for task, (pos, neg, site) in TASKS.items():
        if task not in want_tasks:
            continue
        keep = np.isin(win_cls, pos + neg)
        y = np.isin(win_cls, pos).astype(int)
        tr_sub = np.unique(groups[keep & (win_coh == "CAUEEG")])
        te_sub = np.unique(groups[keep & (win_coh == site)])
        if len(tr_sub) < 40 or len(te_sub) < 16:
            continue
        idx = np.flatnonzero(keep)
        Xk, yk, gk = Xw[idx], y[idx], groups[idx]
        y_sub = {s: int(yk[gk == s][0]) for s in te_sub}
        yte = np.array([y_sub[s] for s in te_sub])
        prev = float(np.mean([y[np.flatnonzero(groups == sname)[0]] for sname in tr_sub]))
        trep = report["tasks"].get(task) or {
            "eval_cohort": site, "n_train_subjects": int(len(tr_sub)),
            "n_eval_subjects": int(len(te_sub)), "n_eval_pos": int(yte.sum()), "models": {}}
        print("\n=== %s  train %d  eval %d (%d pos)" % (task, len(tr_sub), len(te_sub), yte.sum()),
              flush=True)

        for name in [m for m in args.models.split(",") if m]:
            t0 = time.time()
            try:
                subs, sc = train_eval(name, Xk, yk, gk, tr_sub, te_sub, device)
                order = {s: i for i, s in enumerate(subs)}
                s_aligned = np.array([sc[order[s]] for s in te_sub])
                r = metrics(yte, s_aligned, prev)
                r["minutes"] = round((time.time() - t0) / 60, 1)
                np.savez_compressed(PRED / ("%s__%s.npz" % (task, name)),
                                    subjects=te_sub, y=yte, score=s_aligned)
                trep["models"][name] = r
                o = r["at_source_prevalence"]
                print("   %-12s AUC %.3f [%.3f-%.3f]  AP %.3f  sens %.3f spec %.3f  "
                      "bal %.3f  ECE %.3f  %.1f min"
                      % (name, r["auc"], r["ci95"][0], r["ci95"][1], r["average_precision"],
                         o["sensitivity"], o["specificity"], o["balanced_accuracy"],
                         r["ece"], r["minutes"]), flush=True)
            except Exception as e:                                          # noqa: BLE001
                trep["models"][name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}
                print("   %-12s FAILED %s" % (name, str(e)[:100]), flush=True)

        # classical models on the same task, subject level
        cls_sub = np.array([CLASS[str(labels[np.flatnonzero(groups == s)[0]]).upper()]
                            for s in T["subjects"]])
        coh_sub = T["cohort"]
        km = np.isin(cls_sub, pos + neg)
        ys = np.isin(cls_sub, pos).astype(int)
        tr = km & (coh_sub == "CAUEEG")
        te = km & (coh_sub == site)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            T["X_concept"][tr], ys[tr]))
        for cname, ctor, feat in (
                ("random_forest", lambda: RandomForestClassifier(400, random_state=SEED,
                                                                 n_jobs=-1, class_weight="balanced"),
                 "X_spectral"),
                ("gradient_boosting", lambda: GradientBoostingClassifier(random_state=SEED),
                 "X_spectral"),
                ("logistic", lambda: LogisticRegression(max_iter=2000), "X_spectral"),
                ("SVM", lambda: SVC(probability=True, random_state=SEED), "X_spectral")):
            Z = T[feat]
            if Z is None:
                continue
            sc_ = StandardScaler().fit(Z[tr])
            fits = [ctor().fit(sc_.transform(Z[tr][a]), ys[tr][a]) for a, _ in folds]
            pr = np.mean([f.predict_proba(sc_.transform(Z[te]))[:, 1] for f in fits], axis=0)
            r = metrics(ys[te], pr, prev)
            if r:
                np.savez_compressed(PRED / ("%s__%s.npz" % (task, cname)),
                                    subjects=np.asarray(T["subjects"])[te], y=ys[te], score=pr)
                trep["models"][cname] = r
                o = r["at_source_prevalence"]
                print("   %-12s AUC %.3f [%.3f-%.3f]  AP %.3f  sens %.3f spec %.3f  bal %.3f"
                      % (cname, r["auc"], r["ci95"][0], r["ci95"][1], r["average_precision"],
                         o["sensitivity"], o["specificity"], o["balanced_accuracy"]), flush=True)

        report["tasks"][task] = trep
        write_result(dest, report, script=Path(__file__).name)

    print("\nwrote %s" % dest, flush=True)


if __name__ == "__main__":
    main()
