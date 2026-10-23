"""Sensitivity analysis for the manuscript's receiving-site MAD scale floor.

Reuses the exact age-normalized SpectraLite-16 checkpoints and repeated, disjoint
feature-anchor / score-threshold / held-control splits from experiment 244. This
tests the manuscript formulation scale=max(1, 1.4826*MAD) without retraining.
Results are written to a separate folder and never overwrite experiment 244.
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


E244 = _load_module("local_reanchor244_floor1", ROOT / "experiments" / "244_local_feature_reanchor_spectralite16.py")
E243 = E244.E243
E237 = E244.E237
A204 = E244.A204

OUT = E244.OUT / "manuscript_scale_floor1_sensitivity"
TASKS = E244.TASKS
SITES = ("ds004504", "P-ADIC")
TARGET_SPEC = E244.TARGET_SPEC
REPEATS = E244.REPEATS
BOOTSTRAPS = E244.BOOTSTRAPS


def _load_models(torch, task):
    from src.models.spectral75k import SpectraLite
    models = {fold: {} for fold in range(E244.FOLDS)}
    for fold in range(E244.FOLDS):
        for seed in E244.SEEDS:
            path = E244.CHECKPOINTS / f"{task}__seed{seed}__fold{fold}__v1.pt"
            if not path.exists():
                raise FileNotFoundError(f"Expected experiment-244 checkpoint: {path}")
            model = SpectraLite(n_classes=2, n_features=16).to("cuda")
            model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            model.eval()
            models[fold][seed] = model
    return models


def _predict_floor1_ensemble(torch, models, scoring_subjects, cache_by_fold, adapt_subjects):
    """Apply manuscript scale=max(1, 1.4826*MAD), then average the fixed ensemble."""
    total = np.zeros(len(scoring_subjects), dtype=np.float64)
    for fold in range(E244.FOLDS):
        cache = cache_by_fold[fold]
        anchor = np.stack([cache["medians"][sid] for sid in adapt_subjects]).astype(np.float64)
        shift = np.median(anchor, axis=0)
        mad = 1.4826 * np.median(np.abs(anchor - shift), axis=0)
        scale = np.maximum(1.0, mad)
        x, groups = E244._site_groups(scoring_subjects, cache)
        x = ((x.reshape(len(x), -1) - shift) / scale).reshape(-1, 19, 16).astype(np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError(f"Non-finite scale-floor-1 features in fold {fold}")
        for seed in E244.SEEDS:
            window_scores = E243._predict(torch, models[fold][seed], x)
            total += E243._subject_mean(window_scores, groups, scoring_subjects) / (
                E244.FOLDS * len(E244.SEEDS)
            )
    return total.astype(np.float32)


def _run_site(torch, task, site, subjects, labels, cache, models, split_data, budget):
    subjects = E244._as_str(subjects)
    labels = np.asarray(labels, dtype=np.int64)
    id_to_idx = {sid: i for i, sid in enumerate(subjects)}
    cases_idx = np.flatnonzero(labels == 1)
    cases = subjects[cases_idx].tolist()
    anchor_indices = split_data[f"{site}__k{budget}__adapt_indices"]
    threshold_indices = split_data[f"{site}__k{budget}__threshold_indices"]
    held_indices = split_data[f"{site}__k{budget}__held_indices"]
    base_mean = split_data[f"{site}__k{budget}__mean_source_reference"]
    base_count = split_data[f"{site}__k{budget}__count_source_reference"]
    source_threshold = float(split_data["source_threshold"])
    adapted_sum = np.zeros(len(subjects), dtype=np.float64)
    adapted_count = np.zeros(len(subjects), dtype=np.int32)
    auc_by_split = []
    operating = {
        "transferred_source": {"specificity": [], "sensitivity": []},
        "local_quantile_separate_subset": {"specificity": [], "sensitivity": []},
        "local_conformal_separate_subset": {"specificity": [], "sensitivity": []},
    }
    split_predictions = np.full((REPEATS, len(subjects)), np.nan, dtype=np.float32)

    for rep in range(REPEATS):
        adapt_ids = subjects[anchor_indices[rep]].tolist()
        threshold_idx = threshold_indices[rep]
        held_idx = held_indices[rep]
        # Score cases plus the disjoint threshold and held-control sets. Feature anchors
        # are used only to fit shift/scale and are never part of evaluation.
        scoring_idx = np.unique(np.concatenate([cases_idx, threshold_idx, held_idx]))
        scoring_subjects = subjects[scoring_idx].tolist()
        scores = _predict_floor1_ensemble(
            torch, models, scoring_subjects, cache, adapt_ids
        )
        score_by_idx = dict(zip(scoring_idx.tolist(), scores.tolist()))
        split_predictions[rep, scoring_idx] = scores

        eval_idx = np.concatenate([cases_idx, held_idx])
        y_eval = labels[eval_idx]
        score_eval = np.asarray([score_by_idx[i] for i in eval_idx], dtype=float)
        auc_by_split.append(E243._auc(y_eval, score_eval))

        threshold_scores = np.asarray([score_by_idx[i] for i in threshold_idx], dtype=float)
        thresholds = {
            "transferred_source": source_threshold,
            "local_quantile_separate_subset": float(np.quantile(threshold_scores, TARGET_SPEC)),
            "local_conformal_separate_subset": E243._conformal_threshold(threshold_scores, TARGET_SPEC),
        }
        held_scores = np.asarray([score_by_idx[i] for i in held_idx], dtype=float)
        case_scores = np.asarray([score_by_idx[i] for i in cases_idx], dtype=float)
        for rule, threshold in thresholds.items():
            operating[rule]["specificity"].append(float(np.mean(held_scores < threshold)))
            operating[rule]["sensitivity"].append(float(np.mean(case_scores >= threshold)))
        adapted_sum[eval_idx] += np.asarray([score_by_idx[i] for i in eval_idx])
        adapted_count[eval_idx] += 1
        if rep == 0 or (rep + 1) % 50 == 0:
            print(f"[{task}/{site}/k{budget}/floor1] split {rep + 1}/{REPEATS}", flush=True)

    valid = adapted_count > 0
    y_valid = labels[valid]
    adapted_mean = (adapted_sum[valid] / adapted_count[valid]).astype(np.float32)
    base_valid = np.asarray(base_mean, dtype=np.float32)[valid]
    auc = E244._paired_auc_summary(
        y_valid, base_valid, adapted_mean,
        seed=8500 + TASKS.index(task) * 100 + SITES.index(site) * 10 + budget,
    )
    return {
        "total_anchor_budget": int(budget),
        "feature_anchor_n": int(anchor_indices.shape[1]),
        "separate_threshold_anchor_n": int(threshold_indices.shape[1]),
        "repeats": REPEATS,
        "primary_paired_auc": auc,
        "split_auc_distribution": E244._dist_summary(auc_by_split),
        "operating_point_split_distributions": {
            rule: {metric: E244._dist_summary(values) for metric, values in metrics.items()}
            for rule, metrics in operating.items()
        },
        "evaluated_subjects": int(valid.sum()),
        "case_subjects": int(np.sum(y_valid == 1)),
        "held_control_subjects": int(np.sum(y_valid == 0)),
        "adapted_subject_mean_scores": (adapted_sum / np.maximum(adapted_count, 1)).astype(np.float32),
        "adapted_subject_score_counts": adapted_count,
        "split_predictions": split_predictions,
    }


def run_task(task):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run this sensitivity analysis on CPU")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    OUT.mkdir(parents=True, exist_ok=True)
    task_json = OUT / f"{task}__summary.json"
    task_npz = OUT / f"{task}__scores.npz"
    if task_json.exists() or task_npz.exists():
        raise FileExistsError(f"Refusing to overwrite existing sensitivity output: {task}")
    source_npz = E244.OUT / f"{task}__split_scores.npz"
    if not source_npz.exists():
        raise FileNotFoundError(f"Complete experiment 244 first: {source_npz}")
    started = time.time()
    split_data = np.load(source_npz, allow_pickle=True)
    _, train_subjects, train_y, fold_ids, sites, ages = E244._prepare_task(task)
    features = E237.load_feats()
    _, cache = E244._fit_norms_and_site_cache(
        task, train_subjects, train_y, fold_ids, sites, ages, features
    )
    models = _load_models(torch, task)

    reports, saved = {}, {}
    for site in SITES:
        subjects, labels, _ = sites[site]
        reports[site] = {"budgets": {}}
        for budget in E244.TOTAL_BUDGETS:
            result = _run_site(
                torch, task, site, subjects, labels, cache[site], models,
                split_data, budget,
            )
            reports[site]["budgets"][str(budget)] = {
                key: value for key, value in result.items()
                if key not in ("adapted_subject_mean_scores", "adapted_subject_score_counts", "split_predictions")
            }
            saved[f"{site}__k{budget}__mean_scores"] = result["adapted_subject_mean_scores"]
            saved[f"{site}__k{budget}__score_counts"] = result["adapted_subject_score_counts"]
            saved[f"{site}__k{budget}__split_predictions"] = result["split_predictions"]
            saved[f"{site}__k{budget}__subjects"] = E244._as_str(subjects)
            saved[f"{site}__k{budget}__y"] = np.asarray(labels, dtype=np.int64)

    report = {
        "model": "SpectraLite",
        "parameters": 75971,
        "node_features": 16,
        "sensitivity_arm": "scale=max(1, 1.4826*MAD), using feature-reference controls only",
        "reused_checkpoints_and_splits_from": str(E244.OUT),
        "source_age_normative_reference_retrained": False,
        "threshold_controls_disjoint_from_feature_anchors": True,
        "held_controls_excluded_from_both_calibration_fits": True,
        "evaluated_only_at_two_external_sites": True,
        "sites": reports,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    np.savez_compressed(task_npz, **saved)
    task_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{task}] wrote manuscript-formula sensitivity results in {report['elapsed_minutes']:.1f} min", flush=True)


def summarize():
    reports = {task: json.loads((OUT / f"{task}__summary.json").read_text(encoding="utf-8"))
               for task in TASKS}
    p_values, labels = [], []
    for task in TASKS:
        for site in SITES:
            item = reports[task]["sites"][site]["budgets"]["20"]["primary_paired_auc"]
            p_values.append(1.0 if item["p_delong"] is None else float(item["p_delong"]))
            labels.append((task, site))
    adjusted = E243._holm(p_values)
    for (task, site), p_holm in zip(labels, adjusted):
        reports[task]["sites"][site]["budgets"]["20"]["primary_paired_auc"][
            "p_holm_across_6_scale_floor_sensitivity_tests"
        ] = float(p_holm)
    summary = {
        "model": "SpectraLite",
        "parameters": 75971,
        "node_features": 16,
        "scale_formula": "max(1, 1.4826 * median absolute deviation)",
        "interpretation": "Sensitivity analysis for the manuscript-described local scale floor; same fixed checkpoints and repeated splits as the primary experiment.",
        "tasks": reports,
    }
    path = OUT / "summary.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nManuscript max-one MAD scale sensitivity: re-anchor minus source age-reference only")
    print("task                 site       base AUC  floor-1 AUC  delta [paired 95% CI]    p(Holm)")
    for task in TASKS:
        for site in SITES:
            item = reports[task]["sites"][site]["budgets"]["20"]["primary_paired_auc"]
            ci = item["paired_subject_bootstrap_ci95"]
            ph = item["p_holm_across_6_scale_floor_sensitivity_tests"]
            print(f"{task:20s} {site:10s} {item['auc_reference']:.3f}     {item['auc_adapted']:.3f}       "
                  f"{item['delta_adapted_minus_reference']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {ph:.4f}")
    print(f"\nWrote {path}")


def summarize_operating_comparison():
    """Compare threshold-only calibration with feature re-anchoring plus the same rule.

    This is split-level descriptive inference: the repeated partitions overlap, so
    their percentiles are not population confidence intervals.
    """
    result = {
        "scale_formula": "max(1, 1.4826 * MAD)",
        "comparison": "age-normalized SpectraLite source reference vs the same scores after local feature re-centering/rescaling; each arm gets its own threshold from the same disjoint threshold anchors",
        "split_distribution_note": "Percentiles describe repeated anchor-split variability, not population confidence intervals.",
        "tasks": {},
    }
    for task in TASKS:
        adapted = np.load(OUT / f"{task}__scores.npz", allow_pickle=True)
        original = np.load(E244.OUT / f"{task}__split_scores.npz", allow_pickle=True)
        task_result = {}
        for site in SITES:
            labels = np.asarray(original[f"{site}__y"], dtype=int)
            case_idx = np.flatnonzero(labels == 1)
            site_result = {}
            for budget in E244.TOTAL_BUDGETS:
                key = f"{site}__k{budget}"
                thresholds_idx = original[f"{key}__threshold_indices"]
                held_idx = original[f"{key}__held_indices"]
                base_split = original[f"{key}__scores"][:, 0, :]
                transformed_split = adapted[f"{key}__split_predictions"]
                arms = {
                    "threshold_only": {"quantile": {"specificity": [], "sensitivity": []},
                                       "conformal": {"specificity": [], "sensitivity": []}},
                    "feature_reanchor_plus_threshold": {
                        "quantile": {"specificity": [], "sensitivity": []},
                        "conformal": {"specificity": [], "sensitivity": []},
                    },
                }
                deltas = {
                    rule: {metric: [] for metric in ("specificity", "sensitivity")}
                    for rule in ("quantile", "conformal")
                }
                for rep in range(REPEATS):
                    held = held_idx[rep]
                    threshold_anchors = thresholds_idx[rep]
                    for arm_name, prediction in (
                        ("threshold_only", base_split[rep]),
                        ("feature_reanchor_plus_threshold", transformed_split[rep]),
                    ):
                        anchor_scores = prediction[threshold_anchors]
                        rules = {
                            "quantile": float(np.quantile(anchor_scores, TARGET_SPEC)),
                            "conformal": E243._conformal_threshold(anchor_scores, TARGET_SPEC),
                        }
                        for rule, threshold in rules.items():
                            spec = float(np.mean(prediction[held] < threshold))
                            sens = float(np.mean(prediction[case_idx] >= threshold))
                            arms[arm_name][rule]["specificity"].append(spec)
                            arms[arm_name][rule]["sensitivity"].append(sens)
                    for rule in ("quantile", "conformal"):
                        for metric in ("specificity", "sensitivity"):
                            deltas[rule][metric].append(
                                arms["feature_reanchor_plus_threshold"][rule][metric][-1]
                                - arms["threshold_only"][rule][metric][-1]
                            )

                site_result[str(budget)] = {
                    "n_feature_reference_anchors": budget // 2,
                    "n_separate_threshold_anchors": budget - budget // 2,
                    "threshold_only": {
                        rule: {metric: E244._dist_summary(values) for metric, values in metrics.items()}
                        for rule, metrics in arms["threshold_only"].items()
                    },
                    "feature_reanchor_plus_threshold": {
                        rule: {metric: E244._dist_summary(values) for metric, values in metrics.items()}
                        for rule, metrics in arms["feature_reanchor_plus_threshold"].items()
                    },
                    "paired_split_delta_reanchor_minus_threshold_only": {
                        rule: {metric: E244._dist_summary(values) for metric, values in metrics.items()}
                        for rule, metrics in deltas.items()
                    },
                }
            task_result[site] = site_result
        result["tasks"][task] = task_result
    path = OUT / "operating_comparison.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote paired operating-point comparison to {path}")
    for task in TASKS:
        for site in SITES:
            d = result["tasks"][task][site]["20"][
                "paired_split_delta_reanchor_minus_threshold_only"]["conformal"]
            print(f"{task:20s} {site:10s} conformal delta specificity={d['specificity']['mean']:+.3f}; "
                  f"delta sensitivity={d['sensitivity']['mean']:+.3f} "
                  f"(split-variability 2.5-97.5%: "
                  f"{d['sensitivity']['p2_5']:+.3f} to {d['sensitivity']['p97_5']:+.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summarize-operating", action="store_true")
    args = parser.parse_args()
    if args.summarize and args.summarize_operating:
        parser.error("Choose only one summary mode")
    if args.summarize_operating:
        if args.task:
            parser.error("--summarize-operating and --task cannot be combined")
        summarize_operating_comparison()
    elif args.summarize:
        if args.task:
            parser.error("--summarize and --task cannot be combined")
        summarize()
    elif args.task:
        run_task(args.task)
    else:
        parser.error("Specify --task or --summarize")


if __name__ == "__main__":
    main()
