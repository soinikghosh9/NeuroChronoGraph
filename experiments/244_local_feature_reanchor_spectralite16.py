"""Leakage-controlled local feature re-anchoring for canonical SpectraLite-16.

Fits source age references within CAUEEG training folds, then tests receiving-site healthy-control
location-only and location+scale corrections. Feature-reference anchors, score-threshold anchors,
held-out controls, and cases have separate roles. Cases never enter either calibration fit.

Primary analysis: 20 local controls per split (10 feature-reference + 10 threshold anchors), 200
repeated splits, paired external AUC and sensitivity/specificity. A 10-control total budget is
secondary (5 + 5). Results go to outputs/results/canonical/local_reanchor_spectralite16/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E243 = _load_module("age_norm243_reanchor", ROOT / "experiments" / "243_age_norm_spectralite16.py")
E237 = E243.E237
A204 = E243.A204

CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "local_reanchor_spectralite16"
CHECKPOINTS = OUT / "checkpoints"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
SEEDS = (42, 43, 44)
FOLDS = 5
TOTAL_BUDGETS = (10, 20)
REPEATS = 200
PRIMARY_BUDGET = 20
TARGET_SPEC = 0.80
BOOTSTRAPS = 4000
METHODS = ("source_reference", "local_location", "local_location_scale")
FLOOR = float(A204.FLOOR)


def _prepare_task(task: str):
    ref_path = CAN / task / "spectralite16__seed42.npz"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    ref = np.load(ref_path, allow_pickle=True)
    train_subjects = E243._as_str(ref["train_subjects"])
    train_y = np.asarray(ref["train_y"], dtype=np.int64)
    fold_ids = np.asarray(ref["fold_of_subject"], dtype=np.int64)
    if not (len(train_subjects) == len(train_y) == len(fold_ids)):
        raise ValueError(f"Canonical train arrays do not align for {task}")
    sites = {
        "ds004504": (_as_str(ref["ext_subjects"]), np.asarray(ref["ext_y"], dtype=np.int64), "ds004504_raw"),
        "P-ADIC": (_as_str(ref["padic_subjects"]), np.asarray(ref["padic_y"], dtype=np.int64), "P-ADIC"),
    }
    maps = E243._load_ages()
    ages = {sid: E243._age_for(sid, "CAUEEG", maps) for sid in train_subjects}
    for subjects, _, cohort in sites.values():
        ages.update({sid: E243._age_for(sid, cohort, maps) for sid in subjects})
    return ref, train_subjects, train_y, fold_ids, sites, ages


def _as_str(values):
    return np.asarray([str(x) for x in values])


def _fit_norms_and_site_cache(task, train_subjects, train_y, fold_ids, sites, ages, features):
    norms = []
    site_cache = {site: [] for site in sites}
    for fold in range(FOLDS):
        fit_idx = np.flatnonzero(fold_ids != fold)
        fit_subjects, fit_y = train_subjects[fit_idx], train_y[fit_idx]
        controls = fit_subjects[fit_y == 0]
        if len(controls) < 30:
            raise ValueError(f"{task} fold {fold}: only {len(controls)} source controls")
        subject_medians = np.stack([
            np.nanmedian(np.asarray(features[sid], dtype=np.float32), axis=0).reshape(-1)
            for sid in controls
        ])
        norm = A204.norm_fit(subject_medians, np.asarray([ages[sid] for sid in controls]), min_n=30)
        if int(np.asarray(norm["usable"]).sum()) != 19 * 16:
            raise ValueError(f"{task} fold {fold}: not all 304 feature dimensions have a source reference")
        norms.append(norm)
        for site, (subjects, _, _) in sites.items():
            per_subject, medians = {}, {}
            for sid in subjects:
                raw = np.asarray(features[sid], dtype=np.float32)
                z = A204.norm_dev(
                    norm, raw.reshape(len(raw), -1), np.full(len(raw), ages[sid], dtype=float)
                ).reshape(-1, 19, 16).astype(np.float32)
                per_subject[sid] = z
                medians[sid] = np.nanmedian(z, axis=0).reshape(-1).astype(np.float32)
            site_cache[site].append({"windows": per_subject, "medians": medians})
    return norms, site_cache


def _site_groups(subjects, cache_fold):
    x = np.concatenate([cache_fold["windows"][sid] for sid in subjects], axis=0)
    groups = np.concatenate([np.full(len(cache_fold["windows"][sid]), sid) for sid in subjects])
    return x, groups


def _predict_one_site(torch, model, subjects, cache_fold):
    x, groups = _site_groups(subjects, cache_fold)
    window_scores = E243._predict(torch, model, x)
    return E243._subject_mean(window_scores, groups, subjects)


def _fit_local_transform(cache_fold, subjects, mode):
    if mode == "local_location":
        anchor = np.stack([cache_fold["medians"][sid] for sid in subjects]).astype(float)
        shift = np.median(anchor, axis=0)
        scale = np.ones_like(shift)
    elif mode == "local_location_scale":
        anchor = np.stack([cache_fold["medians"][sid] for sid in subjects]).astype(float)
        shift = np.median(anchor, axis=0)
        mad = 1.4826 * np.median(np.abs(anchor - shift), axis=0)
        # Match the established age-anchor helper: preserve a neutral scale for near-zero MAD.
        scale = np.where(mad > FLOOR, mad, 1.0)
    elif mode == "source_reference":
        shift = np.zeros(19 * 16, dtype=float)
        scale = np.ones_like(shift)
    else:
        raise ValueError(mode)
    return shift.astype(np.float32), scale.astype(np.float32)


def _alignment_error(medians, shift, scale):
    values = (np.asarray(medians, dtype=float) - shift) / scale
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    location_error = float(np.median(np.abs(center)))
    scale_error = float(np.median(np.abs(np.log(np.maximum(mad, FLOOR)))))
    return location_error, scale_error


def _predict_ensemble(torch, models, subjects, cache_by_fold, adapt_subjects, held_control_subjects, mode):
    total = np.zeros(len(subjects), dtype=np.float64)
    alignment_errors = {
        name: [] for name in (
            "before_location", "before_log_scale", "after_location", "after_log_scale"
        )
    }
    for fold in range(FOLDS):
        cache = cache_by_fold[fold]
        shift, scale = _fit_local_transform(cache, adapt_subjects, mode)
        x, groups = _site_groups(subjects, cache)
        x = ((x.reshape(len(x), -1) - shift) / scale).reshape(-1, 19, 16).astype(np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError(f"Non-finite adapted features in fold {fold}")
        # This alignment diagnostic is evaluated on controls that did not fit the transform.
        if held_control_subjects:
            held_medians = np.stack([cache["medians"][sid] for sid in held_control_subjects])
            before_location, before_scale = _alignment_error(
                held_medians, np.zeros_like(shift), np.ones_like(scale)
            )
            after_location, after_scale = _alignment_error(held_medians, shift, scale)
            alignment_errors["before_location"].append(before_location)
            alignment_errors["before_log_scale"].append(before_scale)
            alignment_errors["after_location"].append(after_location)
            alignment_errors["after_log_scale"].append(after_scale)
        for seed in SEEDS:
            model = models[fold][seed]
            window_scores = E243._predict(torch, model, x)
            total += E243._subject_mean(window_scores, groups, subjects) / (FOLDS * len(SEEDS))
    return total.astype(np.float32), {
        key: float(np.mean(values)) if values else None
        for key, values in alignment_errors.items()
    }


def _split_indices(control_subjects, total_budget, repetition):
    controls = np.asarray(sorted(control_subjects))
    rng = np.random.default_rng(10_000 * total_budget + repetition)
    permuted = controls[rng.permutation(len(controls))]
    n_feature = total_budget // 2
    adapt = permuted[:n_feature]
    threshold = permuted[n_feature:total_budget]
    held = permuted[total_budget:]
    if len(adapt) < 2 or len(threshold) < 2 or len(held) < 2:
        raise ValueError(f"Insufficient controls for total anchor budget {total_budget}")
    return adapt.tolist(), threshold.tolist(), held.tolist()


def _operating(scores, y, held_ids, case_ids, threshold):
    y = np.asarray(y, dtype=int)
    held = np.isin(np.arange(len(y)), held_ids) & (y == 0)
    cases = y == 1
    return {
        "specificity": float(np.mean(scores[held] < threshold)),
        "sensitivity": float(np.mean(scores[cases] >= threshold)),
    }


def _dist_summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p2_5": float(np.percentile(x, 2.5)),
        "p97_5": float(np.percentile(x, 97.5)),
        "interpretation": "repeated anchor-split variability; not a population confidence interval",
    }


def _paired_auc_summary(y, baseline, adapted, seed):
    boot = E243._paired_bootstrap_delta(y, baseline, adapted, draws=BOOTSTRAPS, seed=seed)
    delong = E243._delong_paired(y, adapted, baseline)
    return {
        "auc_reference": E243._auc(y, baseline),
        "auc_adapted": E243._auc(y, adapted),
        "delta_adapted_minus_reference": boot["delta_age_minus_raw"],
        "paired_subject_bootstrap_ci95": boot["ci95"],
        "bootstrap_draws": BOOTSTRAPS,
        "p_delong": delong["p_delong"],
        "delong_ci95": delong["ci95_delong"],
        "ci_note": "conditional on the fitted three-seed model ensemble and repeated anchor-averaged scores",
    }


def run_smoke(task):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    ref, train_subjects, train_y, fold_ids, sites, ages = _prepare_task(task)
    features = E237.load_feats()
    fit_idx = np.flatnonzero(fold_ids != 0)
    controls = train_subjects[fit_idx][train_y[fit_idx] == 0]
    med = np.stack([np.nanmedian(np.asarray(features[s], dtype=np.float32), axis=0).reshape(-1) for s in controls])
    norm = A204.norm_fit(med, np.asarray([ages[s] for s in controls]), min_n=30)
    site = "ds004504"
    subjects, labels, _ = sites[site]
    cache = {"windows": {}, "medians": {}}
    for sid in subjects:
        raw = np.asarray(features[sid], dtype=np.float32)
        z = A204.norm_dev(norm, raw.reshape(len(raw), -1), np.full(len(raw), ages[sid])).reshape(-1, 19, 16)
        cache["windows"][sid] = z.astype(np.float32)
        cache["medians"][sid] = np.nanmedian(z, axis=0).reshape(-1)
    control_ids = subjects[labels == 0].tolist()
    adapt, threshold, held = _split_indices(control_ids, 20, 0)
    xfit, yfit, _ = E243._stack(train_subjects[fit_idx], train_y[fit_idx], features, ages, norm)
    model = E243._train_fold(torch, xfit, yfit, seed=42, epochs=1)
    scores = {}
    for method in METHODS:
        shift, scale = _fit_local_transform(cache, adapt if method != "source_reference" else [], method)
        x, groups = _site_groups(subjects, cache)
        xt = ((x.reshape(len(x), -1) - shift) / scale).reshape(-1, 19, 16).astype(np.float32)
        score = E243._subject_mean(E243._predict(torch, model, xt), groups, subjects)
        scores[method] = score
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError("Smoke-test prediction contains non-finite scores")
    if set(adapt) & set(threshold) or set(adapt) & set(held) or set(threshold) & set(held):
        raise AssertionError("Smoke-test control roles overlap")
    print(f"GPU smoke passed: {torch.cuda.get_device_name(0)}; {len(controls)} source controls; "
          f"local split sizes {len(adapt)}/{len(threshold)}/{len(held)}; "
          f"finite scores for {list(scores)}", flush=True)


def run_task(task: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run on CPU")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)

    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f"{task}__summary.json"
    scores_path = OUT / f"{task}__split_scores.npz"
    if result_path.exists() or scores_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing task output for {task}")
    started = time.time()
    ref, train_subjects, train_y, fold_ids, sites, ages = _prepare_task(task)
    features = E237.load_feats()
    norms, site_cache = _fit_norms_and_site_cache(
        task, train_subjects, train_y, fold_ids, sites, ages, features
    )
    models = {fold: {} for fold in range(FOLDS)}
    oof_by_seed = {seed: np.full(len(train_subjects), np.nan, dtype=np.float32) for seed in SEEDS}
    external_by_seed = {
        site: {seed: np.zeros(len(values[0]), dtype=np.float32) for seed in SEEDS}
        for site, values in sites.items()
    }

    for fold in range(FOLDS):
        fit_idx = np.flatnonzero(fold_ids != fold)
        held_idx = np.flatnonzero(fold_ids == fold)
        xfit, yfit, _ = E243._stack(
            train_subjects[fit_idx], train_y[fit_idx], features, ages, norms[fold]
        )
        xheld, _, gheld = E243._stack(
            train_subjects[held_idx], train_y[held_idx], features, ages, norms[fold]
        )
        for seed in SEEDS:
            checkpoint = CHECKPOINTS / f"{task}__seed{seed}__fold{fold}__v1.pt"
            from src.models.spectral75k import SpectraLite
            model = SpectraLite(n_classes=2, n_features=16).to("cuda")
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cuda", weights_only=True)
                model.load_state_dict(state)
            else:
                model = E243._train_fold(torch, xfit, yfit, seed=seed, epochs=25)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
            model.eval()
            models[fold][seed] = model
            held_scores = E243._predict(torch, model, xheld)
            oof_by_seed[seed][held_idx] = E243._subject_mean(held_scores, gheld, train_subjects[held_idx])
            for site, (subjects, _, _) in sites.items():
                external_by_seed[site][seed] += _predict_one_site(
                    torch, model, subjects, site_cache[site][fold]
                ) / FOLDS
        print(f"[{task}] trained/loaded fold {fold + 1}/{FOLDS}", flush=True)
        del xfit, yfit, xheld, gheld

    oof = np.mean(np.stack([oof_by_seed[s] for s in SEEDS]), axis=0)
    source_controls = oof[train_y == 0]
    source_threshold = float(np.quantile(source_controls, TARGET_SPEC))
    base_scores = {}
    baseline_reference = E243._ensemble(task, age_norm=True)
    for site, (subjects, labels, _) in sites.items():
        base_scores[site] = np.mean(
            np.stack([external_by_seed[site][seed] for seed in SEEDS]), axis=0
        ).astype(np.float32)
        key = "ext" if site == "ds004504" else "padic"
        if not np.array_equal(subjects, baseline_reference[f"{key}_subjects"]):
            raise ValueError(f"{task}/{site}: site subjects differ from canonical age-norm run")
        if not np.allclose(base_scores[site], baseline_reference[key], rtol=0, atol=2e-4):
            max_diff = float(np.max(np.abs(base_scores[site] - baseline_reference[key])))
            raise ValueError(f"{task}/{site}: reproduced age-norm scores differ (max {max_diff:.6g})")
    if not np.allclose(oof, baseline_reference["oof"], rtol=0, atol=2e-4):
        raise ValueError(f"{task}: reproduced OOF scores differ from experiment 243")

    site_reports = {}
    split_artifacts = {
        "train_subjects": train_subjects,
        "train_y": train_y,
        "source_oof": oof,
        "source_threshold": np.asarray(source_threshold),
    }
    norm_beta = np.stack([n["beta"] for n in norms])
    norm_gamma = np.stack([n["gamma"] for n in norms])

    for site, (subjects, labels, _) in sites.items():
        subjects = _as_str(subjects)
        labels = np.asarray(labels, dtype=np.int64)
        control_subjects = subjects[labels == 0].tolist()
        case_mask = labels == 1
        base = base_scores[site]
        if len(control_subjects) <= max(TOTAL_BUDGETS) + 2:
            raise ValueError(f"{task}/{site}: insufficient controls for planned held-out evaluation")
        site_report = {
            "n_subjects": int(len(subjects)),
            "n_controls": int(np.sum(labels == 0)),
            "n_cases": int(np.sum(labels == 1)),
            "source_threshold": source_threshold,
            "budgets": {},
        }
        site_split_scores = {}
        for total_budget in TOTAL_BUDGETS:
            budget_key = str(total_budget)
            threshold_half = total_budget - total_budget // 2
            metric_values = {
                method: {
                    "auc": [],
                    "operating": {
                        "transferred_source": {"specificity": [], "sensitivity": []},
                        "local_quantile_separate_subset": {"specificity": [], "sensitivity": []},
                        "local_conformal_separate_subset": {"specificity": [], "sensitivity": []},
                    }
                } for method in METHODS
            }
            metric_values["source_reference"]["operating"].update({
                "local_quantile_full_budget": {"specificity": [], "sensitivity": []},
                "local_conformal_full_budget": {"specificity": [], "sensitivity": []},
            })
            split_auc_deltas = {
                method: [] for method in ("local_location", "local_location_scale")
            }
            split_operating_deltas = {
                method: {
                    rule: {"specificity": [], "sensitivity": []}
                    for rule in ("local_quantile_separate_subset", "local_conformal_separate_subset")
                }
                for method in ("local_location", "local_location_scale")
            }
            alignment = {
                method: {
                    "before_location": [], "before_log_scale": [],
                    "after_location": [], "after_log_scale": [],
                }
                for method in METHODS
            }
            score_sum = {method: np.zeros(len(subjects), dtype=np.float64) for method in METHODS}
            score_count = {method: np.zeros(len(subjects), dtype=np.int32) for method in METHODS}
            split_scores = np.full((REPEATS, len(METHODS), len(subjects)), np.nan, dtype=np.float32)
            adapt_indices = np.zeros((REPEATS, total_budget // 2), dtype=np.int32)
            threshold_indices = np.zeros((REPEATS, threshold_half), dtype=np.int32)
            held_indices = np.zeros((REPEATS, len(control_subjects) - total_budget), dtype=np.int32)

            id_to_idx = {sid: i for i, sid in enumerate(subjects)}
            for repetition in range(REPEATS):
                adapt_ids, threshold_ids, held_ids = _split_indices(
                    control_subjects, total_budget, repetition
                )
                adapt_idx = np.asarray([id_to_idx[s] for s in adapt_ids], dtype=int)
                threshold_idx = np.asarray([id_to_idx[s] for s in threshold_ids], dtype=int)
                all_anchor_idx = np.concatenate([adapt_idx, threshold_idx])
                held_idx = np.asarray([id_to_idx[s] for s in held_ids], dtype=int)
                adapt_indices[repetition] = adapt_idx
                threshold_indices[repetition] = threshold_idx
                held_indices[repetition] = held_idx

                predictions = {"source_reference": base}
                align = {}
                for method in ("local_location", "local_location_scale"):
                    predictions[method], align[method] = _predict_ensemble(
                        torch, models, subjects, site_cache[site], adapt_ids, held_ids, method
                    )
                align["source_reference"] = {
                    "before_location": None,
                    "before_log_scale": None,
                    "after_location": None,
                    "after_log_scale": None,
                }
                eval_idx = np.concatenate([np.flatnonzero(case_mask), held_idx])
                eval_y = labels[eval_idx]
                case_idx = np.flatnonzero(case_mask)

                thresholds_by_method = {}
                for method, scores in predictions.items():
                    metric_values[method]["auc"].append(E243._auc(eval_y, scores[eval_idx]))
                    thresholds = {
                        "transferred_source": source_threshold,
                        "local_quantile_separate_subset": float(np.quantile(scores[threshold_idx], TARGET_SPEC)),
                        "local_conformal_separate_subset": E243._conformal_threshold(scores[threshold_idx], TARGET_SPEC),
                    }
                    if method == "source_reference":
                        thresholds["local_quantile_full_budget"] = float(
                            np.quantile(scores[all_anchor_idx], TARGET_SPEC)
                        )
                        thresholds["local_conformal_full_budget"] = E243._conformal_threshold(
                            scores[all_anchor_idx], TARGET_SPEC
                        )
                    thresholds_by_method[method] = thresholds
                    for rule, threshold in thresholds.items():
                        metric_values[method]["operating"][rule]["specificity"].append(
                            float(np.mean(scores[held_idx] < threshold))
                        )
                        metric_values[method]["operating"][rule]["sensitivity"].append(
                            float(np.mean(scores[case_idx] >= threshold))
                        )
                    for method_delta in ("local_location", "local_location_scale"):
                        if method == method_delta:
                            split_auc_deltas[method_delta].append(
                                float(E243._auc(eval_y, scores[eval_idx]) -
                                      E243._auc(eval_y, predictions["source_reference"][eval_idx]))
                            )
                    for key in alignment[method]:
                        alignment[method][key].append(align[method][key])
                    for rule in ("local_quantile_separate_subset", "local_conformal_separate_subset"):
                        if method in split_operating_deltas:
                            for metric in ("specificity", "sensitivity"):
                                split_operating_deltas[method][rule][metric].append(
                                    metric_values[method]["operating"][rule][metric][-1] -
                                    metric_values["source_reference"]["operating"][rule][metric][-1]
                                )
                    split_scores[repetition, METHODS.index(method)] = scores
                    score_sum[method][eval_idx] += scores[eval_idx]
                    score_count[method][eval_idx] += 1

                if repetition == 0 or (repetition + 1) % 20 == 0:
                    print(f"[{task}/{site}/k{total_budget}] anchor split {repetition + 1}/{REPEATS}",
                          flush=True)

            averaged_scores = {
                method: (score_sum[method] / np.maximum(score_count[method], 1)).astype(np.float32)
                for method in METHODS
            }
            all_eval = np.flatnonzero(score_count["source_reference"] > 0)
            y_eval = labels[all_eval]
            primary_cmp = _paired_auc_summary(
                y_eval, averaged_scores["source_reference"][all_eval],
                averaged_scores["local_location_scale"][all_eval],
                seed=6000 + TASKS.index(task) * 100 + (0 if site == "ds004504" else 10) + total_budget,
            )
            location_cmp = _paired_auc_summary(
                y_eval, averaged_scores["source_reference"][all_eval],
                averaged_scores["local_location"][all_eval],
                seed=7000 + TASKS.index(task) * 100 + (0 if site == "ds004504" else 10) + total_budget,
            )
            budget_report = {
                "feature_anchor_n": total_budget // 2,
                "threshold_anchor_n": threshold_half,
                "repeats": REPEATS,
                "primary_location_scale_auc": primary_cmp,
                "secondary_location_only_auc": location_cmp,
                "auc_split_distribution": {
                    method: _dist_summary(metric_values[method]["auc"]) for method in METHODS
                },
                "paired_auc_delta_split_distribution": {
                    method: _dist_summary(values) for method, values in split_auc_deltas.items()
                },
                "operating_split_distribution": {
                    method: {
                        rule: {
                            metric: _dist_summary(values)
                            for metric, values in metrics.items()
                        }
                        for rule, metrics in metric_values[method]["operating"].items()
                    } for method in METHODS
                },
                "paired_operating_delta_split_distribution": {
                    method: {
                        rule: {
                            metric: _dist_summary(values)
                            for metric, values in metrics.items()
                        } for rule, metrics in rules.items()
                    } for method, rules in split_operating_deltas.items()
                },
                "heldout_control_feature_alignment": {
                    method: {
                        "before": {
                            "median_abs_feature_location": _dist_summary(values["before_location"]),
                            "median_abs_log_robust_scale": _dist_summary(values["before_log_scale"]),
                        },
                        "after": {
                            "median_abs_feature_location": _dist_summary(values["after_location"]),
                            "median_abs_log_robust_scale": _dist_summary(values["after_log_scale"]),
                        },
                    } for method, values in alignment.items()
                    if method != "source_reference"
                },
            }
            site_report["budgets"][budget_key] = budget_report
            site_split_scores[budget_key] = {
                "scores": split_scores,
                "adapt_indices": adapt_indices,
                "threshold_indices": threshold_indices,
                "held_indices": held_indices,
                "averaged_scores": averaged_scores,
                "score_count": score_count,
            }
        site_reports[site] = site_report
        split_artifacts[f"{site}__subjects"] = subjects
        split_artifacts[f"{site}__y"] = labels
        split_artifacts[f"{site}__base_scores"] = base
        for budget_key, arrays in site_split_scores.items():
            for name, value in arrays.items():
                if name == "averaged_scores":
                    for method, scores in value.items():
                        split_artifacts[f"{site}__k{budget_key}__mean_{method}"] = scores
                elif name == "score_count":
                    for method, counts in value.items():
                        split_artifacts[f"{site}__k{budget_key}__count_{method}"] = counts
                else:
                    split_artifacts[f"{site}__k{budget_key}__{name}"] = value

    report = {
        "model": "SpectraLite",
        "parameters": 75971,
        "node_features": 16,
        "tasks": task,
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "external_sites": ["ds004504", "P-ADIC"],
        "anchor_budgets_total": list(TOTAL_BUDGETS),
        "feature_reference_and_threshold_anchors_disjoint": True,
        "heldout_controls_excluded_from_both_fits": True,
        "local_scale": "1.4826*MAD; fallback scale 1 when MAD <= 1e-9",
        "threshold_target_specificity": TARGET_SPEC,
        "primary_budget": PRIMARY_BUDGET,
        "primary_contrast": "age-normalized source reference + local location/scale vs age-normalized source reference alone",
        "secondary_contrast": "age-normalized source reference + local location only vs source reference alone",
        "limitations": [
            "Only two age-complete external sites are evaluated.",
            "Local anchor-split percentiles describe calibration-split variability, not population confidence intervals.",
            "Paired subject-bootstrap intervals condition on a fixed three-seed prediction ensemble.",
            "The 10-control total budget leaves only five controls for each distinct calibration role.",
            "The earlier 12-feature anchor experiment included feature-reference controls in evaluation; this run excludes them.",
        ],
        "sites": site_reports,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    np.savez_compressed(
        scores_path,
        **split_artifacts,
        norm_beta=norm_beta,
        norm_gamma=norm_gamma,
        norm_usable=np.stack([n["usable"] for n in norms]),
    )
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{task}] wrote {result_path}; elapsed {report['elapsed_minutes']:.1f} min", flush=True)


def summarize():
    reports = {}
    for task in TASKS:
        path = OUT / f"{task}__summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed task report: {path}")
        reports[task] = json.loads(path.read_text(encoding="utf-8"))
    p_values = []
    labels = []
    for task, report in reports.items():
        for site in ("ds004504", "P-ADIC"):
            test = report["sites"][site]["budgets"][str(PRIMARY_BUDGET)]["primary_location_scale_auc"]
            p_values.append(1.0 if test["p_delong"] is None else float(test["p_delong"]))
            labels.append((task, site))
    adjusted = E243._holm(p_values)
    for (task, site), p_adjusted in zip(labels, adjusted):
        reports[task]["sites"][site]["budgets"][str(PRIMARY_BUDGET)][
            "primary_location_scale_auc"
        ]["p_holm_across_6_primary_tests"] = float(p_adjusted)
    summary = {
        "model": "SpectraLite",
        "parameters": 75971,
        "node_features": 16,
        "primary_family": "six task-by-site location-scale versus no-local-feature-reanchoring AUC tests at total anchor budget 20",
        "tasks": reports,
        "interpretation_note": "AUC intervals are conditional on the fixed three-seed ensemble; split percentiles are descriptive.",
    }
    target = OUT / "summary.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPrimary paired external AUC: location+scale re-anchor minus source age-reference only")
    print("task                 site       base AUC  reanchored  delta [paired 95% CI]    p(Holm)")
    for task, report in reports.items():
        for site in ("ds004504", "P-ADIC"):
            r = report["sites"][site]["budgets"][str(PRIMARY_BUDGET)]["primary_location_scale_auc"]
            ci = r["paired_subject_bootstrap_ci95"]
            ph = r["p_holm_across_6_primary_tests"]
            print(f"{task:20s} {site:10s} {r['auc_reference']:.3f}     {r['auc_adapted']:.3f}     "
                  f"{r['delta_adapted_minus_reference']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {ph:.4f}")
    print(f"\nWrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--smoke", action="store_true", help="one-fold GPU pipeline smoke test; writes no results")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        if args.summarize:
            parser.error("--smoke and --summarize cannot be combined")
        run_smoke(args.task or TASKS[0])
    elif args.summarize:
        if args.task:
            parser.error("--summarize and --task cannot be combined")
        summarize()
    elif args.task:
        run_task(args.task)
    else:
        parser.error("Specify --task, --smoke, or --summarize")


if __name__ == "__main__":
    main()
