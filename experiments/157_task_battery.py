"""Every prediction task the corpus supports, evaluated under one protocol, with attribution.

Three gaps this closes.

**Tasks were implicit.** The manuscript evaluates a screening contrast and a dementia contrast and
mentions others in passing, so a reader cannot tell which model was tested on which question. The
corpus supports an Alzheimer's-versus-control contrast with 221 training and 36 external cases and
that contrast was never reported at all. Tasks are therefore declared here as data, one row each,
and every model is run on every task for which the classes exist.

**Attribution was concept-model only.** Leave-one-out AUC and contribution share were computed for
the sign-constrained head. Neither is available for the comparators, so the manuscript cannot say
whether the forest uses the same information. Permutation importance is computed here for every
model on every task, on the external site, at the subject level, which makes the comparison direct.

**Importance was reported without stability.** A single leave-one-out delta on 87 subjects is not
evidence that a concept matters. Every importance estimate here is repeated and reported with its
spread, so a concept that only looks important once is visible as such.

    python experiments/157_task_battery.py
Out: outputs/results/audit/task_battery.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

OUT = ROOT / "outputs" / "results" / "audit"
FEAT = ROOT / "outputs" / "harmonised_features"
N_BOOT, SEED, N_PERM = 2000, 42, 20

# canonical class per cohort label, so the three labelling conventions collapse to one
CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD",
         "CN": "CN", "C": "CN", "CONTROL": "CN", "MCI": "MCI"}

# task = (positive classes, negative classes, train cohort, evaluation cohort)
TASKS = {
    "CN_vs_AD":        (("AD",),            ("CN",), "CAUEEG", "ds004504_raw"),
    "CN_vs_dementia":  (("AD", "FTD"),      ("CN",), "CAUEEG", "ds004504_raw"),
    "CN_vs_MCI":       (("MCI",),           ("CN",), "CAUEEG", "GENEEG"),
    "CN_vs_impaired":  (("AD", "FTD", "MCI"), ("CN",), "CAUEEG", "ds004504_raw"),
    "MCI_vs_dementia": (("AD", "FTD"),      ("MCI",), "CAUEEG", None),
}


#: Topographic concepts whose relative form is invariant to the amplitudes it is named after,
#: and the amplitude-referenced estimator that replaces each under --substitute. Only the two
#: the battery could adjudicate are substituted; left-right asymmetry is not, because its
#: amplitude form carries no criterion validity and substituting it dilutes the gain.
SUBSTITUTE = {"posterior_maximality": 0, "ap_gradient_alpha": 1}   # -> window_amp column


def subject_table(tag, substitute=False):
    """Subject-level concepts, spectral block, class and cohort.

    With ``substitute=True`` the two recovered amplitude-referenced estimators take the place
    of the relative forms they are named after. The columns keep their original position and
    name, so every downstream consumer -- the sign constraint, the attribution, the report
    panel -- indexes the same slot and only the measurement changes.
    """
    from src.models.v6 import CONCEPT_NAMES, aggregate_subject, impute_from_training
    d = FEAT / tag
    C_win = np.load(d / "window_concepts.npy")
    A_win = None
    if substitute:
        ap = d / "window_amp_concepts.npy"
        if not ap.exists():
            raise SystemExit(
                f"--substitute needs {ap}, which only the global scaling arm produces. "
                f"Extract with 127_extract_harmonised.py --mode global.")
        A_win = np.load(ap)
        if len(A_win) != len(C_win):
            raise SystemExit(f"amp rows {len(A_win)} != concept rows {len(C_win)}")
    S_win = np.load(d / "spectral.npy") if (d / "spectral.npy").exists() else None
    m = np.load(d / "meta.npz", allow_pickle=False)
    g, lb, co = m["groups"], m["labels"], m["cohort"]
    subs = np.unique(g)
    idx = [np.flatnonzero(g == s) for s in subs]
    coh = np.array([str(co[i[0]]) for i in idx])
    cls = np.array([CLASS[str(lb[i[0]]).upper()] for i in idx])
    C = np.stack([aggregate_subject(C_win[i]) for i in idx])
    if A_win is not None:
        with np.errstate(all="ignore"):
            A = np.stack([np.nanmedian(A_win[i], axis=0) for i in idx])
        for nm, col in SUBSTITUTE.items():
            C[:, CONCEPT_NAMES.index(nm)] = A[:, col]
    tr = coh == "CAUEEG"
    C, _ = impute_from_training(C, tr)
    S = None
    if S_win is not None and len(S_win) == len(C_win):
        with np.errstate(all="ignore"):
            S = np.stack([np.nanmedian(S_win[i], axis=0) for i in idx])
        S = np.where(np.isfinite(S), S, np.nanmedian(S[tr], axis=0))
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    return {"subjects": subs, "cohort": coh, "cls": cls,
            "X_concept": C[:, cols], "names": [CONCEPT_NAMES[i] for i in cols],
            "X_spectral": S}


def boot_auc(y, s, n_boot=N_BOOT, seed=SEED):
    from _protocol_v2 import safe_auc
    a = safe_auc(y, s)
    if a is None:
        return None
    rng = np.random.default_rng(seed)
    y, s, n = np.asarray(y), np.asarray(s), len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v = safe_auc(y[i], s[i])
        if v is not None:
            out.append(v)
    return {"auc": a, "ci95": [float(np.percentile(out, 2.5)),
                               float(np.percentile(out, 97.5))], "n": int(n)}


def permutation_importance(fit_predict, X, y, names, n_repeat=N_PERM, seed=SEED):
    """Drop in AUC when one column is shuffled, at the subject level, repeated."""
    from _protocol_v2 import safe_auc
    base = safe_auc(y, fit_predict(X))
    if base is None:
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    for j, nm in enumerate(names):
        drops = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            a = safe_auc(y, fit_predict(Xp))
            if a is not None:
                drops.append(base - a)
        if drops:
            out[nm] = {"mean_drop": float(np.mean(drops)), "sd": float(np.std(drops)),
                       "ci95": [float(np.percentile(drops, 2.5)),
                                float(np.percentile(drops, 97.5))]}
    return {"baseline_auc": base, "importance": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--substitute", action="store_true",
                    help="replace the two recovered topographic estimators with their "
                         "amplitude-referenced forms, and flip the constraint that changes "
                         "polarity with them. Needs a global-scaling extraction. Writes to "
                         "its own artefact so the reported relative arm is never overwritten.")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from src.models.v6 import SignConstrainedLogistic
    from src.utils.provenance import write_result

    DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
           "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
           "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
           "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
           "aperiodic_exponent": 0, "spectral_edge": -1}

    if args.substitute:
        # The amplitude estimators measure the opposite pole of the same clinical statement
        # for occipital maximality: the relative form is a rank where higher means alpha is
        # no longer occipitally dominant, the amplitude form is a power ratio where higher
        # means it still is. Its constraint therefore flips. The anterior-posterior gradient
        # already carried a negative constraint and keeps it.
        DIR = dict(DIR, posterior_maximality=-1, ap_gradient_alpha=-1)

    T = subject_table(args.tag, substitute=args.substitute)
    names = T["names"]
    report = {"tag": args.tag, "n_boot": N_BOOT, "n_permutations": N_PERM,
              "substituted": (sorted(SUBSTITUTE) if args.substitute else []),
              "directions": {n: DIR[n] for n in T["names"]},
              "concepts": names, "tasks": {}}

    for task, (pos, neg, train_coh, eval_coh) in TASKS.items():
        keep = np.isin(T["cls"], pos + neg)
        y = np.isin(T["cls"], pos).astype(int)
        tr = keep & (T["cohort"] == train_coh)
        if eval_coh is None:                       # no external site: report training-side CV only
            te = np.zeros(len(y), bool)
        else:
            te = keep & (T["cohort"] == eval_coh)
        if tr.sum() < 40 or len(np.unique(y[tr])) < 2:
            continue

        trep = {"positive": list(pos), "negative": list(neg),
                "train_cohort": train_coh, "eval_cohort": eval_coh,
                "n_train": int(tr.sum()),
                "n_train_pos": int(y[tr].sum()),
                "n_eval": int(te.sum()),
                "n_eval_pos": int(y[te].sum()) if te.sum() else 0,
                "models": {}}
        print(f"\n=== {task}: train {tr.sum()} ({y[tr].sum()} pos) "
              f"eval {te.sum()} ({y[te].sum() if te.sum() else 0} pos)", flush=True)

        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            T["X_concept"][tr], y[tr]))

        def concept_ensemble(constrained, record=None):
            Xtr, ytr = T["X_concept"][tr], y[tr]
            fits = []
            for a, _ in folds:
                if constrained:
                    fits.append(SignConstrainedLogistic([DIR[n] for n in names],
                                                        C=0.5).fit(Xtr[a], ytr[a]))
                else:
                    fits.append(LogisticRegression(max_iter=2000, C=0.5).fit(
                        StandardScaler().fit(Xtr[a]).transform(Xtr[a]), ytr[a]))
            if constrained:
                # The fitted weights are the interpretability claim, so they are recorded rather
                # than inferred from behaviour. A sign constraint is a boundary at zero, and a
                # coefficient that lands exactly on it means the concept is not used at all --
                # which a contribution plot cannot show, because zero times anything is zero.
                W = np.stack([np.asarray(getattr(f, "coef_", getattr(f, "w", None))).ravel()
                              for f in fits])
                if record is not None:
                    mean = W.mean(0)
                    record["coefficients"] = {
                        n: {"mean": float(mean[i]), "sd": float(W[:, i].std()),
                            "zero_in_folds": int((np.abs(W[:, i]) < 1e-8).sum()),
                            "n_folds": int(W.shape[0])}
                        for i, n in enumerate(names)}
                    record["n_zero_concepts"] = int((np.abs(mean) < 1e-8).sum())
                    record["n_concepts"] = int(len(names))
                return lambda X: np.mean([f.predict_proba(X)[:, 1] for f in fits], axis=0)
            sc = StandardScaler().fit(Xtr)
            return lambda X: np.mean([f.predict_proba(sc.transform(X))[:, 1] for f in fits], axis=0)

        def forest(feat):
            X = T[feat]
            if X is None:
                return None
            Xtr, ytr = X[tr], y[tr]
            fits = [RandomForestClassifier(400, random_state=SEED, n_jobs=-1,
                                           class_weight="balanced").fit(Xtr[a], ytr[a])
                    for a, _ in folds]
            return lambda Z: np.mean([f.predict_proba(Z)[:, 1] for f in fits], axis=0)

        weights: dict = {}
        MODELS = {
            "concept_sign_constrained": (concept_ensemble(True, weights), "X_concept"),
            "concept_unconstrained":    (concept_ensemble(False), "X_concept"),
            "concept_forest":           (forest("X_concept"), "X_concept"),
            "spectral_forest":          (forest("X_spectral"), "X_spectral"),
        }

        for mname, (pred, feat) in MODELS.items():
            if pred is None:
                continue
            entry = {}
            if te.sum() >= 16 and len(np.unique(y[te])) > 1:
                entry["external"] = boot_auc(y[te], pred(T[feat][te]))
            oof = np.full(tr.sum(), np.nan)
            Xtr, ytr = T[feat][tr], y[tr]
            for a, b in folds:
                if feat == "X_concept" and mname == "concept_sign_constrained":
                    f = SignConstrainedLogistic([DIR[n] for n in names], C=0.5).fit(Xtr[a], ytr[a])
                    oof[b] = f.predict_proba(Xtr[b])[:, 1]
                elif mname == "concept_unconstrained":
                    sc = StandardScaler().fit(Xtr[a])
                    f = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr[a]), ytr[a])
                    oof[b] = f.predict_proba(sc.transform(Xtr[b]))[:, 1]
                else:
                    f = RandomForestClassifier(400, random_state=SEED, n_jobs=-1,
                                               class_weight="balanced").fit(Xtr[a], ytr[a])
                    oof[b] = f.predict_proba(Xtr[b])[:, 1]
            entry["internal_cv"] = boot_auc(ytr, oof)
            if feat == "X_concept" and te.sum() >= 16 and len(np.unique(y[te])) > 1:
                entry["permutation_importance"] = permutation_importance(
                    pred, T[feat][te].copy(), y[te], names)
            trep["models"][mname] = entry
            ex = entry.get("external", {})
            print("  %-26s internal %.3f   external %s" % (
                mname, entry["internal_cv"]["auc"],
                ("%.3f [%.3f-%.3f]" % (ex["auc"], *ex["ci95"])) if ex else "n/a"), flush=True)

        trep["sign_constrained_weights"] = weights
        report["tasks"][task] = trep

    stem = "task_battery_repaired" if args.substitute else "task_battery"
    dest = write_result(OUT / f"{stem}.json", report, script=Path(__file__).name)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
