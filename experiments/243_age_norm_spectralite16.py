"""Fold-safe age-normative ablation for the canonical 16-feature SpectraLite.

Factorial comparison requested for the manuscript:
  * raw 16-feature SpectraLite (existing canonical shards; used by the audit below)
  * age-normative 16-feature SpectraLite (trained anew)
  * each score model with a transferred source threshold
  * each score model with local empirical or split-conformal thresholds

The quadratic age reference and residual-scale model are fitted on training-fold CAUEEG
controls only. They transform subject windows before model training and inference. The
receiving-site controls used for score-threshold calibration do not enter the feature
transform, so the conformal threshold is evaluated on held-out controls under the usual
exchangeability condition. All score/threshold methods share the splits in _calib_splits.py.

Training shards are written to outputs/results/canonical/age_norm_spectralite16/.
After all task runs finish, use --summarize for paired discrimination, age-matched,
probability-calibration, decision-curve, and local-threshold summaries.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load_module("canonical211_age16", "211_canonical_benchmark.py")
E237 = load_module("spectralite237_age16", "237_canonical_spectralite16.py")
A204 = load_module("age_reference204_age16", "204_age_conditioned.py")
SPLITS = load_module("calibration_splits_age16", "_calib_splits.py")

CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "age_norm_spectralite16"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
SEEDS = (42, 43, 44)
TARGET_SPEC = 0.80
BOOTSTRAPS = 4000


def _as_str(values):
    return np.asarray([str(x) for x in values])


def _load_ages():
    """Age maps for the development cohort and the two primary external cohorts."""
    development_external = A204.load_ages()  # CAUEEG and ds004504 participant metadata
    pmeta_path = CAN / "padic_subject_meta.json"
    pmeta = json.loads(pmeta_path.read_text(encoding="utf-8"))
    padic = {str(subject): float(row["age"]) for subject, row in pmeta.items()
             if row.get("age") is not None}
    bmeta = json.loads((CAN / "brainlat_subject_meta.json").read_text(encoding="utf-8"))
    brainlat = {str(subject): float(row["age"]) for subject, row in bmeta.items()
                if row.get("age") not in (None, "", "nan")}
    return development_external, padic, brainlat


def _age_for(subject: str, cohort: str, maps):
    dev_ds, padic, brainlat = maps
    if cohort == "CAUEEG":
        value = dev_ds.get(A204.keyf(subject, "CAUEEG"))
    elif cohort == "ds004504_raw":
        value = dev_ds.get(A204.keyf(subject, "ds004504_raw"))
    elif cohort == "P-ADIC":
        value = padic.get(subject)
    elif cohort == "BrainLat":
        value = brainlat.get(subject)
    else:
        raise ValueError(f"Age map not configured for cohort {cohort!r}")
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Missing age for {cohort} subject {subject}")
    return float(value)


def _stack(subjects, labels, features, ages, norm=None):
    xs, ys, groups = [], [], []
    for subject, label in zip(subjects, labels):
        sid = str(subject)
        if sid not in features:
            raise KeyError(f"Feature cache has no windows for subject {sid}")
        x = np.asarray(features[sid], dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (19, 16):
            raise ValueError(f"Expected (windows, 19, 16) features, got {x.shape} for {sid}")
        if norm is not None:
            age = ages[sid]
            z = A204.norm_dev(norm, x.reshape(len(x), -1), np.full(len(x), age, float))
            x = z.reshape(len(x), 19, 16).astype(np.float32)
        xs.append(x)
        ys.append(np.full(len(x), int(label), dtype=np.int64))
        groups.append(np.full(len(x), sid))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(groups))


def _subject_mean(window_scores, window_groups, subjects):
    return np.asarray([window_scores[window_groups == str(subject)].mean()
                       for subject in subjects], dtype=np.float32)


def _train_fold(torch, xtrain, ytrain, seed, epochs=25, batch_size=128):
    from src.models.spectral75k import SpectraLite

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpectraLite(n_classes=2, n_features=16).to("cuda")
    weights = torch.tensor([1.0 / max(int((ytrain == c).sum()), 1) for c in (0, 1)],
                           dtype=torch.float32, device="cuda")
    criterion = torch.nn.CrossEntropyLoss(weight=weights / weights.sum() * 2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    xt = torch.tensor(np.ascontiguousarray(xtrain), dtype=torch.float32, device="cuda")
    yt = torch.tensor(ytrain, dtype=torch.long, device="cuda")
    steps_per_epoch = (len(xt) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=steps_per_epoch)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(xt), device="cuda")
        for start in range(0, len(xt), batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(node_features=xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


def _predict(torch, model, x, batch_size=512):
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(np.ascontiguousarray(x[start:start + batch_size]),
                              dtype=torch.float32, device="cuda")
            scores.append(torch.softmax(model(node_features=xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def train_task(task: str, seeds=SEEDS, epochs=25):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to silently train on CPU")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)

    OUT.mkdir(parents=True, exist_ok=True)
    reference_path = CAN / task / "spectralite16__seed42.npz"
    if not reference_path.exists():
        raise FileNotFoundError(f"Canonical 16-feature reference shard missing: {reference_path}")
    ref = np.load(reference_path, allow_pickle=True)
    train_subjects = _as_str(ref["train_subjects"])
    train_y = np.asarray(ref["train_y"], dtype=np.int64)
    fold_of_subject = np.asarray(ref["fold_of_subject"], dtype=np.int64)
    if not (len(train_subjects) == len(train_y) == len(fold_of_subject)):
        raise ValueError("Canonical subject, label, and fold arrays do not align")

    sites = {
        "ext": (_as_str(ref["ext_subjects"]), np.asarray(ref["ext_y"], dtype=np.int64), "ds004504_raw"),
        "padic": (_as_str(ref["padic_subjects"]), np.asarray(ref["padic_y"], dtype=np.int64), "P-ADIC"),
    }
    maps = _load_ages()
    if "brainlat_subjects" in ref:
        bs, by = _as_str(ref["brainlat_subjects"]), np.asarray(ref["brainlat_y"], dtype=np.int64)
        has_age = np.array([s in maps[2] for s in bs])
        sites["brainlat"] = (bs[has_age], by[has_age], "BrainLat")
    ages = {sid: _age_for(sid, "CAUEEG", maps) for sid in train_subjects}
    for _, (subjects, _, cohort) in sites.items():
        ages.update({sid: _age_for(sid, cohort, maps) for sid in subjects})

    print(f"[{task}] loading cached 16-feature windows", flush=True)
    features = E237.load_feats()
    required = set(train_subjects)
    for subjects, _, _ in sites.values():
        required.update(subjects)
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"{len(missing)} required subjects are missing feature windows; first={missing[:3]}")
    print(f"[{task}] CUDA {torch.cuda.get_device_name(0)} | {len(train_subjects)} train subjects | "
          f"external sizes {[(k, len(v[0])) for k, v in sites.items()]}", flush=True)

    for seed in seeds:
        dest = OUT / f"{task}__seed{seed}.npz"
        if dest.exists():
            print(f"[{task}] seed {seed} artifact exists; keeping it and skipping", flush=True)
            continue
        t0 = time.time()
        oof = np.full(len(train_subjects), np.nan, dtype=np.float32)
        ext_fold_scores = {name: [] for name in sites}
        fold_beta, fold_gamma, fold_usable, fold_control_n = [], [], [], []

        for fold in range(5):
            fit_idx = np.flatnonzero(fold_of_subject != fold)
            held_idx = np.flatnonzero(fold_of_subject == fold)
            fit_subjects, fit_y = train_subjects[fit_idx], train_y[fit_idx]
            held_subjects, held_y = train_subjects[held_idx], train_y[held_idx]
            fit_controls = fit_subjects[fit_y == 0]
            if len(fit_controls) < 30:
                raise ValueError(f"Fold {fold} has only {len(fit_controls)} reference controls")

            control_medians = np.stack([
                np.nanmedian(np.asarray(features[sid], dtype=np.float32), axis=0).reshape(-1)
                for sid in fit_controls
            ])
            reference_ages = np.asarray([ages[sid] for sid in fit_controls], dtype=float)
            norm = A204.norm_fit(control_medians, reference_ages, min_n=30)
            fold_beta.append(norm["beta"])
            fold_gamma.append(norm["gamma"])
            fold_usable.append(norm["usable"])
            fold_control_n.append(len(fit_controls))

            xfit, yfit, _ = _stack(fit_subjects, fit_y, features, ages, norm)
            model = _train_fold(torch, xfit, yfit, seed=seed, epochs=epochs)
            xheld, _, gheld = _stack(held_subjects, held_y, features, ages, norm)
            oof[held_idx] = _subject_mean(_predict(torch, model, xheld), gheld, held_subjects)

            for name, (subjects, labels, _) in sites.items():
                xsite, _, gsite = _stack(subjects, labels, features, ages, norm)
                window_score = _predict(torch, model, xsite)
                ext_fold_scores[name].append(_subject_mean(window_score, gsite, subjects))
            print(f"[{task}] seed {seed} fold {fold + 1}/5 done", flush=True)
            del model
            torch.cuda.empty_cache()

        artifact = {
            "train_subjects": train_subjects, "train_y": train_y,
            "fold_of_subject": fold_of_subject, "oof_score": oof,
            "ext_subjects": sites["ext"][0], "ext_y": sites["ext"][1],
            "ext_score": np.mean(ext_fold_scores["ext"], axis=0).astype(np.float32),
            "padic_subjects": sites["padic"][0], "padic_y": sites["padic"][1],
            "padic_score": np.mean(ext_fold_scores["padic"], axis=0).astype(np.float32),
            "train_age": np.asarray([ages[sid] for sid in train_subjects], dtype=np.float32),
            "ext_age": np.asarray([ages[sid] for sid in sites["ext"][0]], dtype=np.float32),
            "padic_age": np.asarray([ages[sid] for sid in sites["padic"][0]], dtype=np.float32),
        }
        if "brainlat" in sites:
            artifact.update(brainlat_subjects=sites["brainlat"][0], brainlat_y=sites["brainlat"][1],
                            brainlat_score=np.mean(ext_fold_scores["brainlat"], axis=0).astype(np.float32),
                            brainlat_age=np.asarray([ages[sid] for sid in sites["brainlat"][0]], dtype=np.float32))
        np.savez_compressed(dest, **artifact)
        norm_dest = OUT / f"{task}__normparams_seed{seed}.npz"
        np.savez_compressed(norm_dest, beta=np.asarray(fold_beta), gamma=np.asarray(fold_gamma),
                            usable=np.asarray(fold_usable), reference_control_n=np.asarray(fold_control_n))
        elapsed = (time.time() - t0) / 60
        info = {
            "task": task, "model": "SpectraLite", "node_features": 16, "parameters": 75971,
            "seed": seed, "epochs": epochs, "folds": 5, "fold_assignment_source": str(reference_path),
            "normative_reference": "quadratic age mean and age-dependent residual scale; training-fold CAUEEG controls only",
            "normative_min_reference_n": 30, "local_thresholds_use_external_controls_only": True,
            "external_sites": {name: {"n": int(len(row[0])), "n_cases": int(row[1].sum()),
                                      "age_complete": True} for name, row in sites.items()},
            "fold_reference_control_n": fold_control_n, "minutes": round(elapsed, 2),
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "checkpoint_saved": False,
        }
        (OUT / f"{task}__seed{seed}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"[{task}] seed {seed} complete in {elapsed:.1f} min | "
              f"age-model AUC ds={_auc(sites['ext'][1], artifact['ext_score']):.3f} "
              f"P-ADIC={_auc(sites['padic'][1], artifact['padic_score']):.3f}", flush=True)
    print(f"[{task}] age-normative training artifacts: {OUT}", flush=True)


def _auc(y, score):
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _paired_bootstrap_delta(y, raw, aged, draws=BOOTSTRAPS, seed=7):
    y = np.asarray(y, dtype=int)
    raw = np.asarray(raw, dtype=float)
    aged = np.asarray(aged, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return {"delta_age_minus_raw": _auc(y, aged) - _auc(y, raw), "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws, dtype=float)
    for b in range(draws):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        diffs[b] = _auc(y[idx], aged[idx]) - _auc(y[idx], raw[idx])
    return {
        "delta_age_minus_raw": _auc(y, aged) - _auc(y, raw),
        "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
        "bootstrap_draws": int(draws), "bootstrap_type": "stratified paired subject bootstrap",
    }


def _delong_paired(y, score_a, score_b):
    """Paired DeLong p-value and normal interval for AUC(score_a)-AUC(score_b)."""
    from scipy.stats import norm
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=bool)
    scores = np.vstack([np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)])
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        return {"p_delong": None, "ci95_delong": [None, None]}
    tx = np.asarray([rankdata(row, method="average") for row in pos])
    ty = np.asarray([rankdata(row, method="average") for row in neg])
    tz = np.asarray([rankdata(row, method="average") for row in scores])
    aucs = tz[:, y].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, y] - tx) / n
    v10 = 1 - (tz[:, ~y] - ty) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    delta = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {"p_delong": 1.0, "ci95_delong": [delta, delta]}
    se = float(np.sqrt(variance))
    return {"p_delong": float(2 * norm.sf(abs(delta / se))),
            "ci95_delong": [float(delta - 1.96 * se), float(delta + 1.96 * se)]}


def _holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _ensemble(task, age_norm: bool):
    if age_norm:
        paths = [OUT / f"{task}__seed{seed}.npz" for seed in SEEDS]
    else:
        paths = [CAN / task / f"spectralite16__seed{seed}.npz" for seed in SEEDS]
    absent = [str(p) for p in paths if not p.exists()]
    if absent:
        raise FileNotFoundError(f"Missing {'age-normalized' if age_norm else 'raw'} shards: {absent}")
    zs = [np.load(path, allow_pickle=True) for path in paths]
    z0 = zs[0]

    def aligned(key_subjects, key_score):
        base_subjects = _as_str(z0[key_subjects])
        score_runs = []
        for z in zs:
            source_subjects = _as_str(z[key_subjects])
            lookup = {sid: i for i, sid in enumerate(source_subjects)}
            score_runs.append(np.asarray([z[key_score][lookup[sid]] for sid in base_subjects], dtype=float))
        return base_subjects, np.mean(score_runs, axis=0)

    train_subjects, oof = aligned("train_subjects", "oof_score")
    ext_subjects, ext = aligned("ext_subjects", "ext_score")
    padic_subjects, padic = aligned("padic_subjects", "padic_score")
    extra = {}
    if all("brainlat_subjects" in z for z in zs):
        bl_subjects, bl = aligned("brainlat_subjects", "brainlat_score")
        y_of = dict(zip(_as_str(z0["brainlat_subjects"]), np.asarray(z0["brainlat_y"], dtype=int)))
        aged_ref = OUT / f"{task}__seed{SEEDS[0]}.npz"
        if aged_ref.exists():
            keep_set = set(_as_str(np.load(aged_ref, allow_pickle=True)["brainlat_subjects"]))
            m = np.array([s in keep_set for s in bl_subjects]); bl_subjects, bl = bl_subjects[m], bl[m]
        extra = {"brainlat_subjects": bl_subjects, "brainlat_y": np.array([y_of[s] for s in bl_subjects]), "brainlat": bl}
    return extra | {
        "train_subjects": train_subjects, "train_y": np.asarray(z0["train_y"], dtype=int), "oof": oof,
        "ext_subjects": ext_subjects, "ext_y": np.asarray(z0["ext_y"], dtype=int), "ext": ext,
        "padic_subjects": padic_subjects, "padic_y": np.asarray(z0["padic_y"], dtype=int), "padic": padic,
    }


def _age_maps_for_eval():
    return _load_ages()


def _ages_for_site(subjects, site, maps):
    dev_ds, padic, brainlat = maps
    if site == "ds004504":
        result = np.asarray([dev_ds.get(A204.keyf(s, "ds004504_raw"), np.nan) for s in subjects], dtype=float)
    elif site == "P-ADIC":
        result = np.asarray([padic.get(str(s), np.nan) for s in subjects], dtype=float)
    elif site == "BrainLat":
        result = np.asarray([brainlat.get(str(s), np.nan) for s in subjects], dtype=float)
    else:
        raise ValueError(site)
    if not np.isfinite(result).all():
        raise ValueError(f"Age missing for {np.sum(~np.isfinite(result))} {site} subjects")
    return result


def _age_match(subjects, y, ages, caliper=3.0):
    controls = [i for i in range(len(y)) if y[i] == 0]
    cases = [i for i in range(len(y)) if y[i] == 1]
    used_cases, pairs = set(), []
    for ci in sorted(controls, key=lambda i: ages[i]):
        available = [j for j in cases if j not in used_cases]
        if not available:
            break
        distance = np.abs(ages[available] - ages[ci])
        nearest = int(np.argmin(distance))
        if distance[nearest] <= caliper:
            case_idx = available[nearest]
            used_cases.add(case_idx)
            pairs.append((ci, case_idx))
    return pairs


def _paired_pair_bootstrap(y_pair, raw_pair, age_pair, draws=BOOTSTRAPS, seed=17):
    y_pair = np.asarray(y_pair, dtype=int)
    raw_pair = np.asarray(raw_pair, dtype=float)
    age_pair = np.asarray(age_pair, dtype=float)
    n_pairs = len(y_pair) // 2
    if n_pairs < 8:
        return {"delta_age_minus_raw": _auc(y_pair, age_pair) - _auc(y_pair, raw_pair), "ci95": [None, None]}
    # Pair bootstrap keeps the age-matched control/case unit intact.
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        pair_ids = rng.integers(0, n_pairs, n_pairs)
        idx = np.ravel(np.column_stack([2 * pair_ids, 2 * pair_ids + 1]))
        diffs.append(_auc(y_pair[idx], age_pair[idx]) - _auc(y_pair[idx], raw_pair[idx]))
    return {"delta_age_minus_raw": _auc(y_pair, age_pair) - _auc(y_pair, raw_pair),
            "ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
            "bootstrap_draws": int(draws), "bootstrap_unit": "matched age pair"}


def _ece(y, score, bins=10):
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    order = np.argsort(score)
    value = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            value += len(indices) / len(y) * abs(score[indices].mean() - y[indices].mean())
    return float(value)


def _probability_summary(y, score):
    from sklearn.linear_model import LogisticRegression
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    clipped = np.clip(score, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(logit, y)
    return {
        "brier": float(np.mean((score - y) ** 2)), "ece_10_equal_count_bins": _ece(y, score),
        "calibration_intercept": float(fit.intercept_[0]), "calibration_slope": float(fit.coef_[0, 0]),
        "mean_score": float(score.mean()), "observed_prevalence": float(y.mean()),
    }


def _decision_curve_summary(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    observed = float(y.mean())
    thresholds = np.round(np.arange(0.02, 0.51, 0.01), 3)
    out = {}
    if observed <= 0 or observed >= 1:
        return out
    for prevalence in (0.05, 0.10, 0.20):
        positive_weight = prevalence / observed
        negative_weight = (1 - prevalence) / (1 - observed)
        effective_n = positive_weight * y.sum() + negative_weight * (1 - y).sum()
        net, treat_all = [], []
        for threshold in thresholds:
            refer = score >= threshold
            tp = np.sum((y == 1) & refer) * positive_weight
            fp = np.sum((y == 0) & refer) * negative_weight
            net.append(tp / effective_n - fp / effective_n * threshold / (1 - threshold))
            treat_all.append(prevalence - (1 - prevalence) * threshold / (1 - threshold))
        advantage = np.asarray(net) - np.maximum(np.asarray(treat_all), 0)
        at_prev = int(np.argmin(np.abs(thresholds - prevalence)))
        out[str(prevalence)] = {
            "max_advantage_per_1000": float(1000 * advantage.max()),
            "advantage_at_threshold_equal_prevalence_per_1000": float(1000 * advantage[at_prev]),
        }
    return out


def _conformal_threshold(anchor_scores, target=TARGET_SPEC):
    ordered = np.sort(np.asarray(anchor_scores, dtype=float))
    k = len(ordered)
    j = int(np.ceil((k + 1) * target))
    return float(np.nextafter(ordered[min(j, k) - 1], np.inf))


def _split_summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p2_5": float(np.percentile(values, 2.5)), "p97_5": float(np.percentile(values, 97.5))}


def _threshold_metrics(model_eval, model_source, site_name):
    prefix = {"ds004504": "ext", "P-ADIC": "padic", "BrainLat": "brainlat"}[site_name]
    subjects, y, score = model_eval[f"{prefix}_subjects"], model_eval[f"{prefix}_y"], model_eval[prefix]
    source_oof = model_source["oof"]
    source_controls = source_oof[model_source["train_y"] == 0]
    source_thr = float(np.quantile(source_controls, TARGET_SPEC))
    controls = subjects[y == 0]
    all_control_mask = y == 0
    cases = y == 1
    result = {"n_subjects": int(len(y)), "n_controls": int(all_control_mask.sum()),
              "n_cases": int(cases.sum()), "source_threshold": {
                  "specificity": float(np.mean(score[all_control_mask] < source_thr)),
                  "sensitivity": float(np.mean(score[cases] >= source_thr)),
              }, "by_k": {}}
    for k in SPLITS.K_VALUES:
        if k >= len(controls):
            continue
        split_values = {rule: {"specificity": [], "sensitivity": []}
                        for rule in ("local_quantile", "local_conformal")}
        for anchor, held in SPLITS.anchor_splits(controls, k):
            is_anchor = np.isin(subjects, anchor)
            is_held = np.isin(subjects, held)
            local_scores = score[is_anchor]
            thresholds = {
                "local_quantile": float(np.quantile(local_scores, TARGET_SPEC)),
                "local_conformal": _conformal_threshold(local_scores),
            }
            for rule, threshold in thresholds.items():
                split_values[rule]["specificity"].append(float(np.mean(score[is_held] < threshold)))
                split_values[rule]["sensitivity"].append(float(np.mean(score[cases] >= threshold)))
        result["by_k"][str(k)] = {
            rule: {metric: _split_summary(vals) for metric, vals in quantities.items()}
            for rule, quantities in split_values.items()
        }
        result["by_k"][str(k)]["split_note"] = (
            "percentiles describe variability over the 200 repeated anchor splits; they are not population CIs"
        )
    return result


def summarize_all():
    OUT.mkdir(parents=True, exist_ok=True)
    per_task = {}
    p_values, p_labels = [], []
    bl_p, bl_labels = [], []
    maps = _age_maps_for_eval()
    for task in TASKS:
        raw = _ensemble(task, age_norm=False)
        aged = _ensemble(task, age_norm=True)
        entry = {"subjects_identical": True, "seeds": list(SEEDS), "sites": {}}
        if not np.array_equal(raw["train_subjects"], aged["train_subjects"]) or not np.array_equal(
                raw["train_y"], aged["train_y"]):
            raise ValueError(f"Raw and age-normalized internal subject arrays differ: {task}")
        internal_delta = _paired_bootstrap_delta(raw["train_y"], raw["oof"], aged["oof"], seed=500 + len(p_values))
        entry["internal_oof"] = {
            "n": int(len(raw["train_y"])), "auc_raw16": _auc(raw["train_y"], raw["oof"]),
            "auc_age_norm16": _auc(aged["train_y"], aged["oof"]),
            "paired_delta_age_minus_raw": internal_delta,
            "delong_ci_and_p": _delong_paired(raw["train_y"], aged["oof"], raw["oof"]),
        }
        for site, subj_key, y_key, score_key, raw_score_key in (
            ("ds004504", "ext_subjects", "ext_y", "ext", "ext"),
            ("P-ADIC", "padic_subjects", "padic_y", "padic", "padic"),
            ("BrainLat", "brainlat_subjects", "brainlat_y", "brainlat", "brainlat"),
        ):
            subjects = raw[subj_key]
            y = raw[y_key]
            age_subjects = aged[subj_key]
            if not np.array_equal(subjects, age_subjects) or not np.array_equal(y, aged[y_key]):
                raise ValueError(f"Raw and age-normalized subject/label order differs: {task} {site}")
            raw_score, age_score = raw[raw_score_key], aged[score_key]
            site_ages = _ages_for_site(subjects, site, maps)
            paired = _paired_bootstrap_delta(y, raw_score, age_score, seed=700 + len(p_values))
            delong = _delong_paired(y, age_score, raw_score)
            auc_entry = {
                "n": int(len(y)), "n_controls": int(np.sum(y == 0)), "n_cases": int(np.sum(y == 1)),
                "auc_raw16": _auc(y, raw_score), "auc_age_norm16": _auc(y, age_score),
                "paired_delta_age_minus_raw": paired,
                "delong_ci_and_p": delong,
            }
            if site == "BrainLat":   # added after the prespecified six-test family; own Holm family
                bl_p.append(delong["p_delong"] if delong["p_delong"] is not None else 1.0); bl_labels.append(task)
            else:
                p_values.append(delong["p_delong"] if delong["p_delong"] is not None else 1.0)
                p_labels.append((task, site))

            match_pairs = _age_match(subjects, y, site_ages, caliper=3.0)
            if len(match_pairs) >= 8:
                ids = np.asarray([idx for pair in match_pairs for idx in pair], dtype=int)
                y_match = y[ids]
                raw_match, age_match = raw_score[ids], age_score[ids]
                match_delta = _paired_pair_bootstrap(y_match, raw_match, age_match,
                                                     seed=1700 + len(p_values))
                auc_entry["age_matched_within_3_years"] = {
                    "pairs": int(len(match_pairs)), "n": int(len(ids)),
                    "mean_age_control": float(site_ages[ids][y_match == 0].mean()),
                    "mean_age_case": float(site_ages[ids][y_match == 1].mean()),
                    "auc_age_alone": _auc(y_match, site_ages[ids]),
                    "auc_raw16": _auc(y_match, raw_match), "auc_age_norm16": _auc(y_match, age_match),
                    "paired_delta_age_minus_raw": match_delta,
                }
            auc_entry["auc_age_alone_full_site"] = _auc(y, site_ages)
            entry["sites"][site] = {
                "discrimination": auc_entry,
                "probability_calibration_raw16": _probability_summary(y, raw_score),
                "probability_calibration_age_norm16": _probability_summary(y, age_score),
                "decision_curve_raw16": _decision_curve_summary(y, raw_score),
                "decision_curve_age_norm16": _decision_curve_summary(y, age_score),
                "local_threshold_raw16": _threshold_metrics(raw, raw, site),
                "local_threshold_age_norm16": _threshold_metrics(aged, aged, site),
            }

        # Add shared-split paired operating-point deltas. These are split-distribution summaries,
        # not inferential CIs, because the 200 anchor partitions overlap.
        for site in ("ds004504", "P-ADIC", "BrainLat"):
            for k in SPLITS.K_VALUES:
                kstr = str(k)
                r = entry["sites"][site]["local_threshold_raw16"]["by_k"].get(kstr)
                a = entry["sites"][site]["local_threshold_age_norm16"]["by_k"].get(kstr)
                if r is None or a is None:
                    continue
                # Exact paired split differences use the same anchor partitions.
                entry["sites"][site].setdefault("paired_threshold_deltas", {})[kstr] = _paired_threshold_deltas(
                    raw, aged, site, k)

        per_task[task] = entry

    adjusted = _holm(p_values)
    for (task, site), p_adj in zip(p_labels, adjusted):
        per_task[task]["sites"][site]["discrimination"]["delong_ci_and_p"]["p_holm_across_6_task_site_tests"] = float(p_adj)
    for task, p_adj in zip(bl_labels, _holm(bl_p)):
        per_task[task]["sites"]["BrainLat"]["discrimination"]["delong_ci_and_p"]["p_holm_across_3_brainlat_tests"] = float(p_adj)
    report = {
        "model": "SpectraLite", "parameters": 75971, "node_features": 16,
        "age_normative_fit": "fold-local CAUEEG controls only; quadratic age mean and age-varying residual scale",
        "target_specificity": TARGET_SPEC, "seeds": list(SEEDS), "anchor_splits": SPLITS.N_SPLITS,
        "k_values": list(SPLITS.K_VALUES), "paired_auc_bootstraps": BOOTSTRAPS,
        "holm_family": "six task-by-site age-normalized versus raw AUC tests",
        "limitations": [
            "P-ADIC and ds004504 form the prespecified six-test family; BrainLat (79 of 80 with age) was added later and is Holm-corrected separately over its three tasks.",
            "Anchor-split percentiles quantify calibration-split variability, not population confidence intervals.",
            "The split-conformal interpretation requires exchangeability of local calibration and future healthy subjects.",
            "External probability calibration and decision curves are secondary, exploratory summaries.",
        ],
        "tasks": per_task,
    }
    dest = OUT / "summary.json"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {dest}")
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    _print_headline(report)


def _paired_threshold_deltas(raw, aged, site, k):
    """Paired differences over exactly the same _calib_splits anchor partitions."""
    prefix = {"ds004504": "ext", "P-ADIC": "padic", "BrainLat": "brainlat"}[site]
    raw_subjects, raw_y, raw_score = raw[f"{prefix}_subjects"], raw[f"{prefix}_y"], raw[prefix]
    age_subjects, age_y, age_score = aged[f"{prefix}_subjects"], aged[f"{prefix}_y"], aged[prefix]
    if not np.array_equal(raw_subjects, age_subjects) or not np.array_equal(raw_y, age_y):
        raise ValueError(f"Cannot pair local thresholds: {site} arrays differ")
    controls = raw_subjects[raw_y == 0]
    cases = raw_y == 1
    paired = {rule: {metric: [] for metric in ("specificity", "sensitivity")}
              for rule in ("local_quantile", "local_conformal")}
    for anchor, held in SPLITS.anchor_splits(controls, k):
        ia = np.isin(raw_subjects, anchor)
        ih = np.isin(raw_subjects, held)
        for rule in paired:
            if rule == "local_quantile":
                tr = float(np.quantile(raw_score[ia], TARGET_SPEC))
                ta = float(np.quantile(age_score[ia], TARGET_SPEC))
            else:
                tr = _conformal_threshold(raw_score[ia])
                ta = _conformal_threshold(age_score[ia])
            paired[rule]["specificity"].append(float(np.mean(age_score[ih] < ta)) -
                                                float(np.mean(raw_score[ih] < tr)))
            paired[rule]["sensitivity"].append(float(np.mean(age_score[cases] >= ta)) -
                                                float(np.mean(raw_score[cases] >= tr)))
    return {rule: {metric: _split_summary(values) for metric, values in metrics.items()}
            for rule, metrics in paired.items()}


def _print_headline(report):
    print("\nPaired external AUC: age-normalized minus raw SpectraLite-16")
    print("task              site       raw    age-norm    delta [paired 95% bootstrap]   p(Holm)")
    for task, row in report["tasks"].items():
        for site, data in row["sites"].items():
            d = data["discrimination"]
            delta = d["paired_delta_age_minus_raw"]
            ci = delta["ci95"]
            p = d["delong_ci_and_p"].get("p_holm_across_6_task_site_tests", d["delong_ci_and_p"].get("p_holm_across_3_brainlat_tests"))
            print(f"{task:17s} {site:10s} {d['auc_raw16']:.3f}  {d['auc_age_norm16']:.3f}  "
                  f"{delta['delta_age_minus_raw']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {p:.4f}")
    print("\nAt k=20: mean held-out-control specificity / case sensitivity")
    print("task              site       model/rule                     specificity  sensitivity")
    for task, row in report["tasks"].items():
        for site, data in row["sites"].items():
            for model_key, model_label in (("local_threshold_raw16", "raw"),
                                           ("local_threshold_age_norm16", "age-norm")):
                krow = data[model_key]["by_k"].get("20")
                if not krow:
                    continue
                for rule in ("local_quantile", "local_conformal"):
                    sp = krow[rule]["specificity"]["mean"]
                    se = krow[rule]["sensitivity"]["mean"]
                    print(f"{task:17s} {site:10s} {model_label + '/' + rule:30s} {sp:.3f}        {se:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, help="train one task; run tasks concurrently for GPU saturation")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--summarize", action="store_true", help="summarize completed age and raw shards")
    args = parser.parse_args()
    if args.summarize:
        if args.task:
            parser.error("--summarize cannot be combined with --task")
        summarize_all()
        return
    if not args.task:
        parser.error("Specify --task or --summarize")
    seeds = tuple(int(v) for v in args.seeds.split(",") if v)
    if not seeds:
        parser.error("At least one seed is required")
    train_task(args.task, seeds=seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
