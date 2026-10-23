"""What does the recording add to age, for EVERY model -- and how do models do when age cannot help?

Review C2: the published increment-over-age endpoint was computed only for the companion paper's
concept model, externally the EEG+age model scored BELOW EEG alone, and the large external increment
was driven by age being below chance at ds004504 (controls older than cases). Here, for each canonical
model and task:

  * substitution margin  D_sub = AUC(EEG) - AUC(age)
  * incremental margin   D_inc = AUC(EEG+age) - AUC(age), with the joint model a logistic regression on
    [logit(EEG score), age] fitted on CAUEEG out-of-fold EEG scores (internal: nested in the same five
    folds; external: fitted on all CAUEEG OOF scores and applied unchanged)
  * EEG+age minus EEG alone (does adding age help or hurt at each site?)
  * permutation-refit p for D_inc (EEG score permuted among CAUEEG subjects, age and label fixed,
    joint model refitted; B=1000, so the smallest attainable p is 1/1001)
  * AGE-MATCHED external AUC: every control greedily matched to its nearest-age case within 3 years
    without replacement (then cases matched back only if unused), so AUC(age) ~ 0.5 in the matched set
    and an EEG AUC there cannot be an age effect.

    python experiments/222_increment_age_matched.py
Out: outputs/results/canonical/increment_age.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CAN = ROOT / "outputs" / "results" / "canonical"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
MODELS = ("rf123", "rf_node", "concept", "spectral75k", "ncg_v5", "ncg_v3", "atcnet", "eegnet", "conformer")
N_PERM, CALIPER = 1000, 3.0


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def joint_fit(s, a, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Z = np.column_stack([logit(s), a]); sc = StandardScaler().fit(Z)
    f = LogisticRegression(max_iter=2000).fit(sc.transform(Z), y)
    return lambda s2, a2: f.predict_proba(sc.transform(np.column_stack([logit(s2), a2])))[:, 1]


def sex_gap(y, s, sel_a, sel_b, n=2000, seed=7):
    """AUC(stratum a) - AUC(stratum b) with each stratum resampled independently: the two groups are
    disjoint subjects, so this is an unpaired difference and its interval must be built that way."""
    if sel_a.sum() <= 5 or sel_b.sum() <= 5 or len(np.unique(y[sel_a])) < 2 or len(np.unique(y[sel_b])) < 2:
        return None
    ya, sa, yb, sb = y[sel_a], s[sel_a], y[sel_b], s[sel_b]
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        ia = rng.integers(0, len(ya), len(ya)); ib = rng.integers(0, len(yb), len(yb))
        if 0 < ya[ia].sum() < len(ya) and 0 < yb[ib].sum() < len(yb):
            out.append(auc(ya[ia], sa[ia]) - auc(yb[ib], sb[ib]))
    d = auc(ya, sa) - auc(yb, sb)
    return {"delta": float(d), "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "p_boot": float(min(1, 2 * min(np.mean(np.array(out) <= 0), np.mean(np.array(out) >= 0))))}


def boot_ci(y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(fn(i))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def age_match(subs, y, ages, caliper=CALIPER):
    ctrl = [i for i in range(len(y)) if y[i] == 0]; case = [i for i in range(len(y)) if y[i] == 1]
    used, pairs = set(), []
    for c in sorted(ctrl, key=lambda i: ages[i]):
        best = None
        for k in case:
            if k in used:
                continue
            d = abs(ages[k] - ages[c])
            if d <= caliper and (best is None or d < best[0]):
                best = (d, k)
        if best:
            used.add(best[1]); pairs.append((c, best[1]))
    idx = np.array(sorted([i for p in pairs for i in p]))
    return idx, len(pairs)


def seed_ensemble(task, model):
    fs = sorted((CAN / task).glob(f"{model}__seed*.npz"))
    if not fs:
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]; z0 = zs[0]
    def al(z, ks, kv):
        o = {s: i for i, s in enumerate(z[ks])}; return np.array([z[kv][o[s]] for s in z0[ks]], float)
    out = dict(tr=z0["train_subjects"], ytr=z0["train_y"], fold=z0["fold_of_subject"],
               oof=np.mean([al(z, "train_subjects", "oof_score") for z in zs], 0),
               te=z0["ext_subjects"], yte=z0["ext_y"],
               ext=np.mean([al(z, "ext_subjects", "ext_score") for z in zs], 0), n_seeds=len(zs))
    if all("padic_score" in z for z in zs):
        out.update(padic_subjects=z0["padic_subjects"], padic_y=z0["padic_y"],
                   padic=np.mean([al(z, "padic_subjects", "padic_score") for z in zs], 0))
    return out


def main():
    A = _load("a204", "204_age_conditioned.py")
    amap = A.load_ages()
    # P-ADIC ages come from the dataset's own per-subject fields (written by 224 --stage inspect)
    pmeta_path = CAN / "padic_subject_meta.json"
    pmeta = json.loads(pmeta_path.read_text()) if pmeta_path.exists() else {}
    padic_age = {s: v["age"] for s, v in pmeta.items() if v.get("age")}
    padic_sex = {s: v.get("sex") for s, v in pmeta.items()}
    rep = {"n_perm": N_PERM, "caliper_years": CALIPER, "tasks": {}}
    rng = np.random.default_rng(20260917)
    for task in TASKS:
        tr_out = {}
        for model in MODELS:
            E = seed_ensemble(task, model)
            if E is None:
                continue
            a_tr = np.array([amap.get(A.keyf(s, "CAUEEG"), np.nan) for s in E["tr"]], float)
            a_te = np.array([amap.get(A.keyf(s, "ds004504_raw"), np.nan) for s in E["te"]], float)
            ok_tr, ok_te = np.isfinite(a_tr), np.isfinite(a_te)
            ytr, s_tr, f_tr, atr = E["ytr"][ok_tr], E["oof"][ok_tr], E["fold"][ok_tr], a_tr[ok_tr]
            yte, s_te, ate = E["yte"][ok_te], E["ext"][ok_te], a_te[ok_te]
            sites = [("external", E["te"][ok_te], yte, s_te, ate)]
            if E.get("padic") is not None:
                ap = np.array([padic_age.get(str(s), np.nan) for s in E["padic_subjects"]], float)
                okp = np.isfinite(ap)
                sites.append(("padic", E["padic_subjects"][okp], E["padic_y"][okp], E["padic"][okp], ap[okp]))

            def internal_joint(s):
                out = np.zeros(len(ytr))
                for k in np.unique(f_tr):
                    tr = f_tr != k
                    out[f_tr == k] = joint_fit(s[tr], atr[tr], ytr[tr])(s[f_tr == k], atr[f_tr == k])
                return out
            j_int = internal_joint(s_tr)
            res = {"n_seeds": E["n_seeds"]}
            arms = [("internal", ytr, s_tr, j_int, atr)]
            for name, subs, ys, ss, aa in sites:
                arms.append((name, ys, ss, joint_fit(s_tr, atr, ytr)(ss, aa), aa))
            for side, y, s, j, a in arms:
                A_age, A_eeg, A_joint = auc(y, a), auc(y, s), auc(y, j)
                res[side] = {"n": int(len(y)), "auc_age": A_age, "auc_eeg": A_eeg, "auc_eeg_age": A_joint,
                             "d_sub": A_eeg - A_age, "d_inc": A_joint - A_age, "joint_minus_eeg": A_joint - A_eeg,
                             "d_inc_ci": boot_ci(y, lambda i: auc(y[i], j[i]) - auc(y[i], a[i])),
                             "joint_minus_eeg_ci": boot_ci(y, lambda i: auc(y[i], j[i]) - auc(y[i], s[i]))}
            ge = {side: 1 for side, *_ in arms}
            for _ in range(N_PERM):
                sp = s_tr[rng.permutation(len(s_tr))]
                if auc(ytr, internal_joint(sp)) - res["internal"]["auc_age"] >= res["internal"]["d_inc"]:
                    ge["internal"] += 1
                for name, subs, ys, ss, aa in sites:
                    if auc(ys, joint_fit(sp, atr, ytr)(ss, aa)) - res[name]["auc_age"] >= res[name]["d_inc"]:
                        ge[name] += 1
            for side in ge:
                res[side]["p_perm"] = ge[side] / (N_PERM + 1)
            # P-ADIC records sex, which the development cohort does not, so the one cohort that can
            # answer "is this a sex effect?" is checked here: sex alone, and the model within each sex.
            if "padic" in res and padic_sex:
                sx = np.array([padic_sex.get(str(s)) for s in sites[-1][1]])
                ysp, ssp = sites[-1][2], sites[-1][3]
                male = sx == "M"; female = sx == "F"
                def within(sel):
                    if sel.sum() <= 5 or len(np.unique(ysp[sel])) < 2:
                        return None
                    y_, s_ = ysp[sel], ssp[sel]
                    return {"n": int(sel.sum()), "n_pos": int(y_.sum()), "auc": auc(y_, s_),
                            "ci95": boot_ci(y_, lambda i: auc(y_[i], s_[i]))}
                res["padic_sex"] = {
                    "n_male": int(male.sum()), "n_female": int(female.sum()),
                    "auc_sex_alone": auc(ysp[male | female], male[male | female].astype(float))
                    if len(np.unique(ysp[male | female])) > 1 else None,
                    "male": within(male), "female": within(female),
                    "female_minus_male": sex_gap(ysp, ssp, female, male),
                    "auc_within_male": auc(ysp[male], ssp[male]) if male.sum() > 5 and len(np.unique(ysp[male])) > 1 else None,
                    "auc_within_female": auc(ysp[female], ssp[female]) if female.sum() > 5 and len(np.unique(ysp[female])) > 1 else None}
            for name, subs, ys, ss, aa in sites:
                idx, n_pairs = age_match(subs, ys, aa)
                if n_pairs >= 8:
                    ym, sm, am = ys[idx], ss[idx], aa[idx]
                    res[f"age_matched_{name}"] = {
                        "n_pairs": int(n_pairs), "auc_age": auc(ym, am), "auc_eeg": auc(ym, sm),
                        "auc_eeg_ci": boot_ci(ym, lambda i: auc(ym[i], sm[i])),
                        "mean_age_ctrl": float(am[ym == 0].mean()), "mean_age_case": float(am[ym == 1].mean())}
            tr_out[model] = res
            am_e = res.get("age_matched_external", {})
            print(f"{task:15s} {model:12s} int: age {res['internal']['auc_age']:.3f} eeg {res['internal']['auc_eeg']:.3f} "
                  f"joint {res['internal']['auc_eeg_age']:.3f} | ext: age {res['external']['auc_age']:.3f} eeg "
                  f"{res['external']['auc_eeg']:.3f} joint {res['external']['auc_eeg_age']:.3f}"
                  + (f" | matched {am_e.get('n_pairs')} pairs age {am_e.get('auc_age', float('nan')):.3f} "
                     f"eeg {am_e.get('auc_eeg', float('nan')):.3f}" if am_e else "")
                  + (f" | P-ADIC age {res['padic']['auc_age']:.3f} eeg {res['padic']['auc_eeg']:.3f}"
                     if "padic" in res else ""), flush=True)
        rep["tasks"][task] = tr_out
        (CAN / "increment_age.json").write_text(json.dumps(rep, indent=1))
    print("wrote", CAN / "increment_age.json")


if __name__ == "__main__":
    main()
