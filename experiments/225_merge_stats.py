"""Merge every canonical shard into the tables the revised paper reports, with the statistics done once.

Inputs: outputs/results/canonical/<task>/<model>__seed<s>.npz from 211 (networks, ladder), 212
(classical), 216 (foundation models) and 224 (P-ADIC scoring, extra `padic_*` arrays when present).

For every task and every evaluation set (internal CAUEEG out-of-fold; each external site):
  * AUC with DeLong 95% CI (Sun & Xu fast DeLong), per seed and for the seed ensemble
    (mean of subject scores over seeds);
  * paired subject bootstrap (B=4000, identical resamples for every model) of the AUC difference
    against the reference (random forest on the 123-d QEEG block, seed ensemble) and against
    the released SpectraLite model; two-sided p, Holm-adjusted within each (task, evaluation set) family;
  * two one-sided tests at margins 0.05 / 0.075 / 0.10 via the 90% bootstrap interval;
  * DeLong paired test as a secondary check.
Plus: parameter counts, and the Spearman correlation of parameter count with external AUC over
EXACTLY the networks plotted (review C9).

    python experiments/225_merge_stats.py
Out: outputs/results/canonical/summary.json, summary_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAN = ROOT / "outputs" / "results" / "canonical"
B, SEED = 4000, 20260917
MARGINS = (0.05, 0.075, 0.10)
TASKS = ("CN_vs_AD", "CN_vs_dementia", "CN_vs_impaired", "CN_vs_MCI")

DISPLAY = {
    "spectralite16": "16-feature variant", "spectral75k": "SpectraLite",
    "ncg_v5": "NCG-v5", "ncg_v2": "NCG-v2", "ncg_v3": "NCG-v3",
    "eegnet": "EEGNetv4", "shallow": "ShallowFBCSPNet", "deep4": "Deep4Net", "eegitnet": "EEG-ITNet",
    "atcnet": "ATCNet", "conformer": "EEGConformer", "eegnex": "EEGNeX",
    "rf123": "Forest (QEEG-123)", "rf_node": "Forest (node features)", "gb": "Gradient boosting",
    "svm": "SVM", "logistic": "Logistic regression", "concept": "Concept model",
    "labram_ft": "LaBraM (fine-tuned)", "cbramod_ft": "CBraMod (fine-tuned)", "lead_ft": "LEAD (fine-tuned)",
    "ladder_spectral+raw": "NCG ladder +raw", "ladder_spectral+graph": "NCG ladder +graph",
    "ladder_spectral+raw+graph": "NCG ladder +raw+graph", "ladder_spectral+raw+graph+lobe": "NCG ladder +raw+graph+lobe",
}
FAMILY = {"spectralite16": "ours", "spectral75k": "ours", "ncg_v5": "graph", "ncg_v2": "graph", "ncg_v3": "graph",
          "eegnet": "deep", "shallow": "deep", "deep4": "deep", "eegitnet": "deep", "atcnet": "deep",
          "conformer": "deep", "eegnex": "deep", "rf123": "classical", "rf_node": "classical", "gb": "classical",
          "svm": "classical", "logistic": "classical", "concept": "concept"}
def _params():
    """Trainable-parameter counts written by 227 (counted from the instantiated models)."""
    p = CAN / "param_budget.json"
    if p.exists():
        d = json.loads(p.read_text()).get("all_models")
        if d:
            return d
    return {"ncg_v5": 964_000}


PARAMS = _params()
PARAMS.setdefault("spectralite16", 75_971)
PARAMS.setdefault("spectral75k", 75_459)
# The capacity correlation is computed over the 11 distinct architectures plotted in the paper, with SpectraLite
# (12 features); the 16-feature variant, chosen after external scoring, is swapped in by 257 (capacity, variant_swapped).
CAPACITY_EXCLUDE = {"spectralite16"}


def display(m):
    if m in DISPLAY:
        return DISPLAY[m]
    for fm, nm in (("labram", "LaBraM"), ("eegpt", "EEGPT"), ("cbramod", "CBraMod"), ("lead", "LEAD")):
        if m.startswith(fm + "_probe"):
            return f"{nm} (frozen probe)"
        if m == fm + "_ft":
            return f"{nm} (fine-tuned)"
    return m


def family(m):
    if m in FAMILY:
        return FAMILY[m]
    if m.startswith(("labram", "eegpt", "cbramod", "lead")):
        return "foundation"
    if m.startswith("ladder"):
        return "ladder"
    return "other"


# ----------------------------------------------------------------------------- statistics
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan
    r = rankdata(s, method="average"); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _midrank(x):
    from scipy.stats import rankdata
    return rankdata(x, method="average")


def delong(y, scores):
    """Fast DeLong (Sun & Xu 2014). scores: [k, n]. Returns aucs [k], covariance [k,k]."""
    y = np.asarray(y).astype(bool); X = np.atleast_2d(scores)
    pos, neg = X[:, y], X[:, ~y]; m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in X])
    aucs = tz[:, y].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, y] - tx) / n; v10 = 1 - (tz[:, ~y] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_ci(y, s):
    from scipy.stats import norm
    a, c = delong(y, np.asarray(s)[None]); se = np.sqrt(c[0, 0])
    return [float(max(0, a[0] - 1.96 * se)), float(min(1, a[0] + 1.96 * se))]


def delong_paired_p(y, s1, s2):
    from scipy.stats import norm
    a, c = delong(y, np.vstack([s1, s2]))
    var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    if var <= 0:
        return 1.0
    return float(2 * norm.sf(abs(a[0] - a[1]) / np.sqrt(var)))


def boot_indices(y, seed, n=B):
    rng = np.random.default_rng(seed); out = []
    y = np.asarray(y)
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(i)
    return np.array(out)


def boot_aucs(y, s, idx):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    out = np.empty(len(idx))
    for b, i in enumerate(idx):
        yy, ss = y[i], s[i]
        r = rankdata(ss, method="average"); n1 = yy.sum(); n0 = len(yy) - n1
        out[b] = (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


# ----------------------------------------------------------------------------- loading
def fm_dropped():
    """Foundation-model shards from the normalisation that was NOT selected on internal out-of-fold
    AUC. Both are computed by 216; reporting both would enter the same encoder twice and quietly give
    it two chances at the external comparison, so only the selected one is merged."""
    p = CAN / "fm_norm_selection.json"
    if not p.exists():
        return set()
    sel = json.loads(p.read_text())["selected"]
    out = set()
    for fm, keep in sel.items():
        out |= {f"{fm}_probe_{n}" for n in ("physical", "zscore") if n != keep}
    return out


DROP = fm_dropped()


def load_task(task):
    d = CAN / task
    models = {}
    for f in sorted(d.glob("*__seed*.npz")):
        name, seed = f.stem.rsplit("__seed", 1)
        if name in DROP:
            continue
        z = np.load(f, allow_pickle=True)
        rec = models.setdefault(name, {"seeds": {}})
        rec["seeds"][int(seed)] = {k: z[k] for k in z.files if not k.endswith("_win") and "win_" not in k
                                   and k != "ext_win_folds"}
    return models


def ensemble(rec, key_sub, key_score):
    seeds = [s for s in sorted(rec["seeds"]) if key_sub in rec["seeds"][s]]
    if len(seeds) < len(rec["seeds"]):
        print(f"   WARNING: {key_sub}: only seeds {seeds} of {sorted(rec['seeds'])} have this set (not yet scored?)",
              flush=True)
    base = rec["seeds"][seeds[0]]
    subs = base[key_sub]
    S = []
    for s in seeds:
        r = rec["seeds"][s]
        order = {x: i for i, x in enumerate(r[key_sub])}
        S.append(np.array([r[key_score][order[x]] for x in subs], float))
    return subs, np.mean(S, 0), [float(auc(base[key_sub.replace("subjects", "y")], x)) for x in S]


def main():
    summary = {"reference": "rf123", "bootstrap": B, "margins": MARGINS, "tasks": {}}
    md = ["# Canonical benchmark summary (auto-generated by 225)\n"]
    for task in TASKS:
        models = load_task(task)
        if "rf123" not in models:
            continue
        tsum = {"models": {}, "evaluation_sets": {}}
        sets = [("internal", "train_subjects", "oof_score", "train_y")]
        sets.append(("external", "ext_subjects", "ext_score", "ext_y"))
        if any("padic_subjects" in r["seeds"][s] for r in models.values() for s in r["seeds"]):
            sets.append(("padic", "padic_subjects", "padic_score", "padic_y"))
        if any("brainlat_subjects" in r["seeds"][s] for r in models.values() for s in r["seeds"]):
            sets.append(("brainlat", "brainlat_subjects", "brainlat_score", "brainlat_y"))
        for set_name, ks, kscore, ky in sets:
            ref = models["rf123"]
            if ks not in ref["seeds"][sorted(ref["seeds"])[0]]:
                continue
            subs_ref, s_ref, _ = ensemble(ref, ks, kscore)
            y_ref = ref["seeds"][sorted(ref["seeds"])[0]][ky]
            idx = boot_indices(y_ref, SEED)
            b_ref = boot_aucs(y_ref, s_ref, idx)
            # The released reference is the 16-feature SpectraLite.
            spec = models.get("spectralite16")
            s_spec = None
            if spec and ks in spec["seeds"][sorted(spec["seeds"])[0]]:
                subs_sp, s_sp, _ = ensemble(spec, ks, kscore)
                if set(subs_sp) == set(subs_ref):
                    o = {x: i for i, x in enumerate(subs_sp)}
                    s_spec = np.array([s_sp[o[x]] for x in subs_ref]); b_spec = boot_aucs(y_ref, s_spec, idx)
            # The locked SpectraLite-12 (fixed before any external scoring) is a second reference.
            spec12 = models.get("spectral75k"); s_spec12 = None
            if spec12 and ks in spec12["seeds"][sorted(spec12["seeds"])[0]]:
                subs12, s12, _ = ensemble(spec12, ks, kscore)
                if set(subs12) == set(subs_ref):
                    o = {x: i for i, x in enumerate(subs12)}
                    s_spec12 = np.array([s12[o[x]] for x in subs_ref]); b_spec12 = boot_aucs(y_ref, s_spec12, idx)
            rows, p_ref, p_spec, names = {}, [], [], []
            for name, rec in models.items():
                first = rec["seeds"][sorted(rec["seeds"])[0]]
                if ks not in first:
                    continue
                subs, s, per_seed = ensemble(rec, ks, kscore)
                if set(subs) != set(subs_ref):
                    continue
                o = {x: i for i, x in enumerate(subs)}
                s = np.array([s[o[x]] for x in subs_ref])
                a = auc(y_ref, s); b = boot_aucs(y_ref, s, idx)
                row = {"display": display(name), "family": family(name), "n_seeds": len(per_seed),
                       "auc": a, "auc_ci_delong": delong_ci(y_ref, s),
                       "auc_per_seed": per_seed,
                       "auc_seed_mean": float(np.mean(per_seed)),
                       "auc_seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                       "n": int(len(y_ref)), "n_pos": int(np.sum(y_ref))}
                d = b - b_ref
                row["vs_rf"] = {"delta": a - auc(y_ref, s_ref),
                                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                                "p_boot": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                                "p_delong": delong_paired_p(y_ref, s, s_ref) if name != "rf123" else 1.0}
                row["vs_rf"]["equivalent_within"] = {str(mg): bool(-mg < row["vs_rf"]["ci90"][0] and row["vs_rf"]["ci90"][1] < mg)
                                                     for mg in MARGINS}
                if s_spec is not None:
                    d2 = b - b_spec
                    row["vs_spectralite"] = {"delta": a - auc(y_ref, s_spec),
                                             "ci95": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))],
                                             "ci90": [float(np.percentile(d2, 5)), float(np.percentile(d2, 95))],
                                             "p_boot": float(min(1, 2 * min((d2 <= 0).mean(), (d2 >= 0).mean())))}
                    row["vs_spectralite"]["equivalent_within"] = {
                        str(mg): bool(-mg < row["vs_spectralite"]["ci90"][0] and row["vs_spectralite"]["ci90"][1] < mg)
                        for mg in MARGINS}
                if s_spec12 is not None:
                    d3 = b - b_spec12
                    row["vs_spectralite12"] = {"delta": a - auc(y_ref, s_spec12),
                                               "ci95": [float(np.percentile(d3, 2.5)), float(np.percentile(d3, 97.5))],
                                               "ci90": [float(np.percentile(d3, 5)), float(np.percentile(d3, 95))],
                                               "p_boot": float(min(1, 2 * min((d3 <= 0).mean(), (d3 >= 0).mean())))}
                    row["vs_spectralite12"]["equivalent_within"] = {
                        str(mg): bool(-mg < row["vs_spectralite12"]["ci90"][0] and row["vs_spectralite12"]["ci90"][1] < mg)
                        for mg in MARGINS}
                if name in PARAMS:
                    row["params"] = PARAMS[name]
                rows[name] = row
                # Every comparator, the graph ladder included, enters the Holm family of its (task, set).
                if name != "rf123":
                    names.append(name); p_ref.append(row["vs_rf"]["p_boot"])
            for nm, adj in zip(names, holm(p_ref)):
                rows[nm]["vs_rf"]["p_holm"] = float(adj)
            rows["rf123"]["vs_rf"]["holm_family_size"] = len(names)
            fam2 = [n for n in rows if n != "spectralite16" and "vs_spectralite" in rows[n]]
            for nm, adj in zip(fam2, holm([rows[n]["vs_spectralite"]["p_boot"] for n in fam2])):
                rows[nm]["vs_spectralite"]["p_holm"] = float(adj)
            fam3 = [n for n in rows if n != "spectral75k" and "vs_spectralite12" in rows[n]]
            for nm, adj in zip(fam3, holm([rows[n]["vs_spectralite12"]["p_boot"] for n in fam3])):
                rows[nm]["vs_spectralite12"]["p_holm"] = float(adj)
            tsum["evaluation_sets"][set_name] = rows
            # Capacity versus discrimination over the distinct trainable architectures in this table.
            nets = [m for m in rows if m in PARAMS and not m.startswith("ladder") and m not in CAPACITY_EXCLUDE]
            if len(nets) >= 5:
                r, p = spearman([PARAMS[m] for m in nets], [rows[m]["auc"] for m in nets])
                tsum.setdefault("capacity_correlation", {})[set_name] = {
                    "spearman_rho": r, "p": p, "n_networks": len(nets), "models": nets}
            md.append(f"\n## {task} -- {set_name} (n={len(y_ref)}, {int(np.sum(y_ref))} positive)\n")
            md.append("| Model | seeds | AUC [DeLong 95% CI] | seed SD | dAUC vs RF [95% CI] | p_Holm | equiv .05/.075/.10 |")
            md.append("|---|---|---|---|---|---|---|")
            for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["auc"]):
                e = r["vs_rf"]["equivalent_within"]
                md.append(f"| {r['display']} | {r['n_seeds']} | {r['auc']:.3f} [{r['auc_ci_delong'][0]:.3f}, "
                          f"{r['auc_ci_delong'][1]:.3f}] | {'' if r['auc_seed_sd'] is None else f'{r['auc_seed_sd']:.3f}'} | "
                          f"{r['vs_rf']['delta']:+.3f} [{r['vs_rf']['ci95'][0]:+.3f}, {r['vs_rf']['ci95'][1]:+.3f}] | "
                          f"{r['vs_rf'].get('p_holm', float('nan')):.3f} | "
                          f"{'Y' if e['0.05'] else '-'}/{'Y' if e['0.075'] else '-'}/{'Y' if e['0.1'] else '-'} |")
        # internal-vs-external dissociation, over every model present in both
        both = [m for m in tsum["evaluation_sets"].get("internal", {})
                if m in tsum["evaluation_sets"].get("external", {}) and not m.startswith("ladder")
                and m not in CAPACITY_EXCLUDE]
        if len(both) >= 5:
            xi = [tsum["evaluation_sets"]["internal"][m]["auc"] for m in both]
            xe = [tsum["evaluation_sets"]["external"][m]["auc"] for m in both]
            r, p = spearman(xi, xe)
            ref_i = tsum["evaluation_sets"]["internal"].get("rf123", {}).get("auc")
            ref_e = tsum["evaluation_sets"]["external"].get("rf123", {}).get("auc")
            tsum["dissociation"] = {
                "spearman_internal_external": r, "p": p, "n_models": len(both),
                "internal_range": [float(min(xi)), float(max(xi))],
                "external_range": [float(min(xe)), float(max(xe))],
                "best_internal_model": both[int(np.argmax(xi))],
                "gain_over_forest_of_best_internal": {
                    "internal": float(max(xi) - ref_i) if ref_i else None,
                    "external": float(tsum["evaluation_sets"]["external"][both[int(np.argmax(xi))]]["auc"] - ref_e)
                    if ref_e else None}}
            md.append(f"\n**Dissociation ({task}):** Spearman(internal, external) = {r:+.2f} "
                      f"(p={p:.2f}, n={len(both)} models); internal range "
                      f"{min(xi):.3f}-{max(xi):.3f}, external range {min(xe):.3f}-{max(xe):.3f}.\n")
            # With a second external site the comparison that matters is site against architecture:
            # how much of the AUC a model loses is explained by which clinic it is read at, next to
            # how much separates the architectures within a clinic.
            pset = tsum["evaluation_sets"].get("padic", {})
            three = [m for m in both if m in pset]
            if len(three) >= 5:
                yi = [tsum["evaluation_sets"]["internal"][m]["auc"] for m in three]
                ye = [tsum["evaluation_sets"]["external"][m]["auc"] for m in three]
                yp = [pset[m]["auc"] for m in three]
                r_ip, p_ip = spearman(yi, yp); r_ep, p_ep = spearman(ye, yp)
                drop = [a - b for a, b in zip(ye, yp)]
                tsum["dissociation"]["second_site"] = {
                    "n_models": len(three),
                    "spearman_internal_padic": r_ip, "p_internal_padic": p_ip,
                    "spearman_external_padic": r_ep, "p_external_padic": p_ep,
                    "padic_range": [float(min(yp)), float(max(yp))],
                    "spread_within_site": {"internal": float(max(yi) - min(yi)),
                                           "external": float(max(ye) - min(ye)),
                                           "padic": float(max(yp) - min(yp))},
                    "drop_external_to_padic": {"mean": float(np.mean(drop)), "min": float(min(drop)),
                                               "max": float(max(drop))}}
                sd = tsum["dissociation"]["second_site"]
                md.append(f"**Second site ({task}):** across {len(three)} models the spread within a site is "
                          f"{sd['spread_within_site']['internal']:.3f} (internal), "
                          f"{sd['spread_within_site']['external']:.3f} (ds004504), "
                          f"{sd['spread_within_site']['padic']:.3f} (P-ADIC), while moving from ds004504 to "
                          f"P-ADIC costs every model {sd['drop_external_to_padic']['min']:.3f}-"
                          f"{sd['drop_external_to_padic']['max']:.3f} AUC (mean "
                          f"{sd['drop_external_to_padic']['mean']:.3f}). Spearman(internal, P-ADIC) = "
                          f"{r_ip:+.2f} (p={p_ip:.2f}); Spearman(ds004504, P-ADIC) = {r_ep:+.2f} (p={p_ep:.2f}).\n")
            # Three external sites: the single set of models scored at every evaluation, so the spread
            # within a site and the drop between sites are read off one consistent set (used in Results).
            blset = tsum["evaluation_sets"].get("brainlat", {})
            allsites = [m for m in three if m in blset]
            if len(allsites) >= 5:
                a_i = [tsum["evaluation_sets"]["internal"][m]["auc"] for m in allsites]
                a_e = [tsum["evaluation_sets"]["external"][m]["auc"] for m in allsites]
                a_p = [pset[m]["auc"] for m in allsites]
                a_b = [blset[m]["auc"] for m in allsites]
                dpa = [a - b for a, b in zip(a_e, a_p)]; dbl = [a - b for a, b in zip(a_e, a_b)]
                tsum["dissociation"]["all_sites"] = {
                    "n_models": len(allsites),
                    "range": {"internal": [float(min(a_i)), float(max(a_i))],
                              "external": [float(min(a_e)), float(max(a_e))],
                              "padic": [float(min(a_p)), float(max(a_p))],
                              "brainlat": [float(min(a_b)), float(max(a_b))]},
                    "spread_within_site": {"internal": float(max(a_i) - min(a_i)),
                                           "external": float(max(a_e) - min(a_e)),
                                           "padic": float(max(a_p) - min(a_p)),
                                           "brainlat": float(max(a_b) - min(a_b))},
                    "drop_external_to_padic": {"mean": float(np.mean(dpa)), "min": float(min(dpa)), "max": float(max(dpa))},
                    "drop_external_to_brainlat": {"mean": float(np.mean(dbl)), "min": float(min(dbl)), "max": float(max(dbl))}}
                sa = tsum["dissociation"]["all_sites"]
                md.append(f"**Three external sites ({task}):** across {len(allsites)} models scored at every "
                          f"evaluation, spread within a site is {sa['spread_within_site']['external']:.3f} "
                          f"(ds004504), {sa['spread_within_site']['padic']:.3f} (P-ADIC), "
                          f"{sa['spread_within_site']['brainlat']:.3f} (BrainLat); the drop ds004504->P-ADIC is "
                          f"{sa['drop_external_to_padic']['min']:.3f}-{sa['drop_external_to_padic']['max']:.3f} "
                          f"(mean {sa['drop_external_to_padic']['mean']:.3f}), ds004504->BrainLat "
                          f"{sa['drop_external_to_brainlat']['min']:.3f}-{sa['drop_external_to_brainlat']['max']:.3f} "
                          f"(mean {sa['drop_external_to_brainlat']['mean']:.3f}).\n")
        # Learning curves: at 55 training subjects the draw-to-draw spread swamps any single model's
        # external gain, so the testable statement is the pattern across model families -- does each
        # family gain more internally than externally? That is a sign test on paired gains.
        lc_path = CAN / "learning_curve" / f"{task}.json"
        if lc_path.exists():
            lc = json.loads(lc_path.read_text())
            per = {}
            for r in lc["rows"]:
                per.setdefault(r["model"], {}).setdefault(r["n_train"], []).append(
                    (r["auc_internal_test"], r["auc_external"]))
            gains = {}
            for m, byn in per.items():
                ns = sorted(byn)
                if len(ns) < 2:
                    continue
                lo, hi = ns[0], ns[-1]
                mean = lambda n, i: float(np.mean([v[i] for v in byn[n]]))
                sd = lambda n, i: float(np.std([v[i] for v in byn[n]], ddof=1)) if len(byn[n]) > 1 else 0.0
                gains[m] = {"n_from": lo, "n_to": hi,
                            "internal_gain": mean(hi, 0) - mean(lo, 0),
                            "external_gain": mean(hi, 1) - mean(lo, 1),
                            "internal_sd_at_min_n": sd(lo, 0), "external_sd_at_min_n": sd(lo, 1)}
            if len(gains) >= 5:
                diffs = [g["internal_gain"] - g["external_gain"] for g in gains.values()]
                k = int(sum(d > 0 for d in diffs)); n = len(diffs)
                # exact two-sided sign test, written out so the merge keeps no SciPy dependency
                from math import comb
                tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
                tsum["learning_curve_pattern"] = {
                    "n_families": n, "n_internal_gt_external": k,
                    "median_internal_minus_external": float(np.median(diffs)),
                    "sign_test_p_two_sided": float(min(1.0, 2 * tail)), "per_model": gains}
                md.append(f"**Learning curves ({task}):** {k}/{n} families gain more internally than "
                          f"externally, median difference {np.median(diffs):+.3f}, sign test "
                          f"p={min(1.0, 2 * tail):.4f}.\n")
        summary["tasks"][task] = tsum
    (CAN / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (CAN / "summary_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
