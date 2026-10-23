"""External validation of candidate electrode placements for SpectraLite-16.

This follow-up tests whether the prespecified temporo-parietal positions carry useful
case-control information outside CAUEEG. It uses the canonical 75,971-parameter,
16-feature checkpoints without retraining. For each candidate montage, it re-references
the already harmonized/ICA-cleaned continuous signal to the candidate channels, recomputes
all sixteen per-channel features, and evaluates the same five source-fold models.

The primary test is the four-channel set T5/T6/P3/P4 versus the full 19-channel input.
Matched-size frontal, frontotemporal, frontocentral, temporal, and posterior layouts test
whether that set is specifically informative. The F7/F8/T3/T4 frontotemporal set is a
literature-motivated secondary hypothesis. T5/T6 alone and an eight-channel posterior strip
are also secondary.
Candidate availability is checked against each subject's harmonization QC sidecar. GENEEG
is reported as ineligible for temporo-parietal validation because T5/T6 are flat there.

This isolates the effect of re-referencing and removing channels after the common upstream
filtering/ICA pipeline. It is not a prospective hardware-acquisition validation and it does
not estimate uncertainty from retraining the source models.

Run: python experiments/247_external_montage_validation_spectralite16.py --bootstraps 2000
Output: outputs/results/canonical/external_montage_validation_spectralite16.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E246 = load_module("electrode_validation_246_for_montage247", "246_electrode_validation_spectralite16.py")
E242 = load_module("montage_features_242_for_montage247", "242_montage_spectralite16.py")
C = E246.C
from src.config.config import CHANNEL_NAMES
from src.models.spectral75k import SpectraLite
from src.models.v5.spectral_nodes import SpectralNodeFeatures

CH = list(CHANNEL_NAMES)
SEEDS = (42, 43, 44)
WINDOW = 1024
BATCH = 256
PARAMETERS = 75_971
FEATURES = 16
CONT = C.ALZ / "harmonised"
OUT = C.OUT / "external_montage_validation_spectralite16.json"

MONTAGES = {
    "tp4": ("T5", "T6", "P3", "P4"),
    "posterior4": ("T5", "T6", "O1", "O2"),
    "temporal4": ("T3", "T4", "T5", "T6"),
    "frontal4": ("Fp1", "Fp2", "F3", "F4"),
    "frontotemporal4": ("F7", "F8", "T3", "T4"),
    "frontocentral4": ("F3", "F4", "C3", "C4"),
    "temporal2": ("T5", "T6"),
    "posterior8": ("F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"),
}
SITE_CACHE = {
    "ds004504": ("channel_clean", "ds004504_raw"),
    "P-ADIC": ("padic_channel_clean", "P-ADIC"),
    "P-ADIC_all_controls": ("padic_channel_clean", "P-ADIC"),
    "BrainLat": ("brainlat_channel_clean", "BrainLat"),
    "GENEEG": ("channel_clean", "GENEEG"),
}
SCORE_KEY = {
    "ds004504": "ext_score", "GENEEG": "ext_score", "P-ADIC": "padic_score",
    "P-ADIC_all_controls": "padic_all_score", "BrainLat": "brainlat_score",
}
SUBJECT_KEY = {
    "ds004504": "ext_subjects", "GENEEG": "ext_subjects", "P-ADIC": "padic_subjects",
    "P-ADIC_all_controls": "padic_all_subjects", "BrainLat": "brainlat_subjects",
}


def _holm_into(rows, pkey, outkey):
    ps = [row[pkey] for row in rows]
    adj = E246.holm(ps)
    for row, value in zip(rows, adj):
        row[outkey] = float(value)


def _load_site_cache(tag, cohort):
    """Load metadata, cached full-montage tensors, raw windows, and sensor QC."""
    base = C.FEAT / tag
    meta = np.load(base / "meta.npz", allow_pickle=False)
    groups = meta["groups"].astype(str)
    starts = meta["start"].astype(int)
    cohorts = meta["cohort"].astype(str)
    idx = np.flatnonzero(cohorts == cohort)
    ids = np.unique(groups[idx])
    if not len(ids):
        raise ValueError(f"No subjects for {tag}/{cohort}")
    node12 = np.load(base / "node12.npy", mmap_mode="r")
    ext4 = np.load(base / "node_ext4.npy", mmap_mode="r")
    feature_map = {}
    raw_map = {}
    flat_map = {}
    rows_by_id = {sid: np.flatnonzero((groups == sid) & (cohorts == cohort)) for sid in ids}
    print(f"  reading {cohort}: {len(ids)} subjects", flush=True)
    for n, sid in enumerate(ids, 1):
        rows = rows_by_id[sid]
        feature_map[sid] = np.concatenate((np.asarray(node12[rows]), np.asarray(ext4[rows])), axis=-1).astype(np.float32)
        starts_sid = starts[rows]
        raw = np.load(CONT / f"{sid}.npy", mmap_mode="r")
        wins = np.stack([np.asarray(raw[:, s:s + WINDOW], dtype=np.float32) for s in starts_sid])
        if wins.shape[-1] != WINDOW:
            raise ValueError(f"Short/missing cached window for {sid}")
        raw_map[sid] = wins * 1e6
        sidecar = CONT / f"{sid}.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing QC sidecar: {sidecar}")
        qc = json.loads(sidecar.read_text(encoding="utf-8")).get("qc", {})
        flat_map[sid] = set(qc.get("flat_channels", []))
        if n % 40 == 0:
            print(f"    loaded {n}/{len(ids)}", flush=True)
    return {"ids": ids, "rows": rows_by_id, "features": feature_map,
            "raw": raw_map, "flat": flat_map}


def _montage_features(site_cache, device, ex, montage_name, channels):
    """Re-reference candidate channels and recompute the 16 input descriptors."""
    eligible = [sid for sid in site_cache["ids"]
                if not (set(channels) & site_cache["flat"][sid])]
    if not eligible:
        return {}, eligible
    chunks = [site_cache["raw"][sid][:, [CH.index(ch) for ch in channels], :]
              for sid in eligible]
    X = np.concatenate(chunks, axis=0)
    del chunks
    # The stored recordings have already received full-montage average reference. Removing
    # their selected-channel mean is algebraically equivalent to average-reference on S.
    X = (X - X.mean(axis=1, keepdims=True)).astype(np.float32, copy=False)
    nwin = np.asarray([len(site_cache["raw"][sid]) for sid in eligible], dtype=int)
    feats = []
    with torch.inference_mode():
        for start in range(0, len(X), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + BATCH])).to(device)
            f12 = ex(xb)
            f4 = E242.extended_features_torch(xb, ex, list(channels))
            feats.append(torch.cat((f12, f4), dim=-1).float().cpu().numpy())
    F = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    if F.shape[1:] != (len(channels), FEATURES):
        raise AssertionError(f"Unexpected features for {montage_name}: {F.shape}")
    out = {}
    cursor = 0
    for sid, count in zip(eligible, nwin):
        out[sid] = F[cursor:cursor + count]
        cursor += count
    if cursor != len(F):
        raise AssertionError("Window-to-subject feature indexing mismatch")
    return out, eligible


@torch.inference_mode()
def _score_subjects(models, subjects, features, channels, device):
    blocks = [features[str(sid)] for sid in subjects]
    ends = np.cumsum([len(x) for x in blocks])
    starts = np.r_[0, ends[:-1]]
    X = np.concatenate(blocks).astype(np.float32, copy=False)
    folds = []
    for model in models:
        model.eval()
        p = []
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH])).to(device)
            p.append(torch.softmax(model(node_features=xb, channel_names=channels), dim=1)[:, 1]
                     .float().cpu().numpy())
        win = np.concatenate(p)
        folds.append(np.asarray([win[a:b].mean() for a, b in zip(starts, ends)], dtype=np.float32))
    return np.mean(folds, axis=0)


def _load_models(task, seed, device):
    models = []
    for fold in range(5):
        path = C.CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{fold}.pt"
        model = SpectraLite(n_classes=2, n_features=FEATURES).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    return models


def _site_report(task, site, data, site_cache, feature_sets, baseline_scores, bootstraps):
    subjects = data["subjects"]
    y = data["y"]
    outputs = {"n_subjects_canonical": int(len(subjects)),
               "n_controls": int((y == 0).sum()), "n_cases": int((y == 1).sum()),
               "montages": {}, "comparisons": {},
               "signal_processing_scope": "subset re-reference after harmonized filtering/ICA; fixed model weights"}
    full = np.asarray(baseline_scores, dtype=np.float32)
    outputs["montages"]["all19_canonical"] = {"electrodes": 19,
                                                "auc": E246.auc(y, full)}
    eligible_by_montage = {}
    candidate_scores = {}
    for montage, channels in MONTAGES.items():
        fmap, eligible_all = feature_sets[montage]
        eligible = np.asarray([sid for sid in subjects if str(sid) in fmap], dtype=str)
        if len(eligible) < 12:
            outputs["montages"][montage] = {"status": "insufficient eligible subjects",
                                            "n_eligible": int(len(eligible)),
                                            "required_channels": list(channels)}
            continue
        ixmap = {str(sid): i for i, sid in enumerate(subjects)}
        ix = np.asarray([ixmap[str(sid)] for sid in eligible], dtype=int)
        yy = y[ix]
        if len(np.unique(yy)) != 2:
            outputs["montages"][montage] = {"status": "single-class after channel QC",
                                            "n_eligible": int(len(eligible))}
            continue
        pred_seeds = []
        for seed in SEEDS:
            pred_seeds.append(_score_subjects(data["models"][seed], eligible, fmap,
                                              list(channels), data["device"]))
        pred = np.mean(pred_seeds, axis=0)
        base = full[ix]
        outputs["montages"][montage] = {
            "electrodes": len(channels), "required_channels": list(channels),
            "n_eligible": int(len(eligible)), "n_controls": int((yy == 0).sum()),
            "n_cases": int((yy == 1).sum()), "auc": E246.auc(yy, pred),
            "auc_full_matched_subjects": E246.auc(yy, base),
        }
        eligible_by_montage[montage] = (eligible, yy, pred, base)
        candidate_scores[montage] = pred

    for montage, item in eligible_by_montage.items():
        eligible, yy, pred, base = item
        seed = zlib.crc32(f"{task}:{site}:{montage}".encode())
        delta, ci = E246.bootstrap_delta(yy, pred, base, bootstraps, seed)
        p = E246.delong_paired(yy, pred, base)
        outputs["comparisons"][f"{montage}_minus_all19"] = {
            "delta_auc_candidate_minus_all19": float(delta), "ci95_paired_subject_bootstrap": ci,
            "p_delong": float(p), "n_subjects": int(len(yy)),
        }

    # Compare every equal-size four-channel layout on the same subjects. This permits an
    # exploratory ranking without selecting a region from the outcomes and then testing it
    # against a narrower comparator family.
    four_channel_layouts = (
        "tp4", "posterior4", "temporal4", "frontal4", "frontocentral4", "frontotemporal4"
    )
    available_groups = [name for name in four_channel_layouts if name in eligible_by_montage]
    if len(available_groups) >= 2:
        common_ids = set(map(str, eligible_by_montage[available_groups[0]][0]))
        for name in available_groups[1:]:
            common_ids.intersection_update(map(str, eligible_by_montage[name][0]))
        first_subjects = eligible_by_montage[available_groups[0]][0]
        common = [str(sid) for sid in first_subjects if str(sid) in common_ids]
        first_y = eligible_by_montage[available_groups[0]][1]
        first_pos = {str(sid): i for i, sid in enumerate(first_subjects)}
        common_y = first_y[np.asarray([first_pos[sid] for sid in common], dtype=int)]
        common_predictions = {}
        for name in available_groups:
            subject_ids, yy_group, pred_group, _ = eligible_by_montage[name]
            positions = {str(sid): i for i, sid in enumerate(subject_ids)}
            indices = np.asarray([positions[sid] for sid in common], dtype=int)
            common_predictions[name] = pred_group[indices]
        outputs["four_channel_common_subjects"] = {
            "n_subjects": int(len(common)),
            "n_controls": int((common_y == 0).sum()),
            "n_cases": int((common_y == 1).sum()),
            "layouts": available_groups,
            "auc_by_layout": {name: float(E246.auc(common_y, common_predictions[name]))
                              for name in available_groups},
        }
    else:
        common = []

    for left_i, left in enumerate(available_groups):
        for right in available_groups[left_i + 1:]:
            if len(common) < 12:
                continue
            left_sub, left_y, left_pred, _ = eligible_by_montage[left]
            right_sub, right_y, right_pred, _ = eligible_by_montage[right]
            left_pos = {str(sid): i for i, sid in enumerate(left_sub)}
            right_pos = {str(sid): i for i, sid in enumerate(right_sub)}
            idx_left = np.asarray([left_pos[sid] for sid in common])
            idx_right = np.asarray([right_pos[sid] for sid in common])
            yy = common_y
            key = f"group_pair_{left}_minus_{right}"
            seed = zlib.crc32(f"{task}:{site}:{key}".encode())
            delta, ci = E246.bootstrap_delta(
                yy, left_pred[idx_left], right_pred[idx_right], bootstraps, seed
            )
            p = E246.delong_paired(yy, left_pred[idx_left], right_pred[idx_right])
            outputs["comparisons"][key] = {
                "left_layout": left, "right_layout": right,
                "auc_left": float(outputs["four_channel_common_subjects"]["auc_by_layout"][left]),
                "auc_right": float(outputs["four_channel_common_subjects"]["auc_by_layout"][right]),
                "delta_auc_left_minus_right": float(delta),
                "ci95_paired_subject_bootstrap": ci, "p_delong": float(p),
                "n_subjects": int(len(yy)),
            }
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(C.TASKS))
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This fixed-checkpoint montage analysis expects the available CUDA GPU")
    torch.use_deterministic_algorithms(False)
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()} | params={PARAMETERS:,} | features={FEATURES}", flush=True)
    ex = SpectralNodeFeatures(sfreq=256.0).to(device).eval()
    tasks = [t for t in args.tasks.split(",") if t]
    report = {
        "model": "SpectraLite", "parameters": PARAMETERS, "features_per_electrode": FEATURES,
        "seeds": list(SEEDS), "primary_montage": list(MONTAGES["tp4"]),
        "primary_contrast": "TP4 fixed-weight subset-CAR vs canonical full-19 score, paired on eligible subjects",
        "primary_p_adjustment": "Holm over all eligible task-site contrasts",
        "secondary_equal_size_contrasts": "All pairwise contrasts among TP4, posterior4, temporal4, frontal4, frontocentral4, and literature-motivated frontotemporal4; Holm over all eligible contrasts",
        "bootstrap_draws": int(args.bootstraps),
        "protocol": "recompute 16 descriptors after subset re-reference of harmonized/ICA-cleaned signals; reuse all five source folds and three seeds; no external training or sensor selection",
        "limitations": [
            "This is a fixed-weight, post-harmonization re-reference analysis; it does not rerun acquisition, filtering, or ICA for each physical montage.",
            "Subject bootstrap intervals condition on the fitted source checkpoints and do not include training-seed or model-refit uncertainty.",
            "GENEEG T5/T6 are flat according to the QC sidecars, so it cannot validate TP4; other layouts are evaluated where their required channels pass QC."
        ],
        "tasks": {},
    }
    site_cache_by_key = {}
    feature_sets_by_key = {}
    for task in tasks:
        if task not in C.TASKS:
            raise ValueError(f"Unknown task {task}")
        shard42 = np.load(C.OUT / task / "spectralite16__seed42.npz", allow_pickle=True)
        sets = E246.get_sets(shard42, task)
        report["tasks"][task] = {}
        print(f"\n=== {task} ===", flush=True)
        # Load each external site's signals/features only once. All candidate windows are
        # reused across task endpoints and fixed source checkpoints.
        sites_to_run = [site for site in sets if site in SITE_CACHE]
        for site in sites_to_run:
            cache_key = SITE_CACHE[site]
            if cache_key not in site_cache_by_key:
                tag, cohort = cache_key
                print(f"Preparing {site} from {tag}/{cohort} ...", flush=True)
                cache = _load_site_cache(tag, cohort)
                site_cache_by_key[cache_key] = cache
                feature_sets_by_key[cache_key] = {}
                for montage, channels in MONTAGES.items():
                    fm, eligible_all = _montage_features(cache, device, ex, montage, channels)
                    feature_sets_by_key[cache_key][montage] = (fm, eligible_all)
                    print(f"  {montage:12s} {len(eligible_all):3d}/{len(cache['ids'])} eligible", flush=True)

        for site, data in sets.items():
            if site not in SITE_CACHE:
                continue
            tag, cohort = SITE_CACHE[site]
            cache_key = (tag, cohort)
            cache = site_cache_by_key[cache_key]
            feature_sets = feature_sets_by_key[cache_key]
            # Restrict canonical sample rows to subjects with a recorded metadata/raw window.
            available = set(cache["ids"])
            missing = [str(s) for s in data["subjects"] if str(s) not in available]
            if missing:
                raise KeyError(f"{task}/{site}: canonical subjects missing from feature cache: {missing[:5]}")
            # Keep subject order and y exactly as in the canonical shard.
            model_store = {}
            for seed in SEEDS:
                model_store[seed] = _load_models(task, seed, device)
            data_for_eval = {**data, "models": model_store, "device": device}
            baseline = np.mean(np.stack([
                np.load(C.OUT / task / f"spectralite16__seed{seed}.npz", allow_pickle=True)[SCORE_KEY[site]]
                for seed in SEEDS
            ]), axis=0)
            # Assert canonical shard order/subject count aligns with the selected feature cache.
            if len(baseline) != len(data["subjects"]):
                raise AssertionError(f"{task}/{site}: baseline score order mismatch")
            for seed in SEEDS:
                shard = np.load(C.OUT / task / f"spectralite16__seed{seed}.npz", allow_pickle=True)
                saved_subjects = np.asarray([str(s) for s in shard[SUBJECT_KEY[site]]])
                if not np.array_equal(saved_subjects, data["subjects"]):
                    raise AssertionError(f"{task}/{site}/seed{seed}: canonical subject order differs")
            result = _site_report(task, site, data_for_eval, cache, feature_sets,
                                  baseline, args.bootstraps)
            report["tasks"][task][site] = result
            print(f"  {site:22s} n={result['n_subjects_canonical']:3d} | "
                  f"AUC full={result['montages']['all19_canonical']['auc']:.3f} | "
                  f"TP4={result['montages'].get('tp4', {}).get('auc', float('nan')):.3f}", flush=True)
            del model_store
            if device == "cuda":
                torch.cuda.empty_cache()

    # Correct the prespecified TP4-vs-full test across all task/site evaluations.
    primary_rows = []
    equal_size_rows = []
    for task, site_rows in report["tasks"].items():
        for site, result in site_rows.items():
            if "comparisons" not in result:
                continue
            reduced = [(name, row) for name, row in result["comparisons"].items()
                       if name.endswith("_minus_all19")]
            if reduced:
                adjusted = E246.holm([row["p_delong"] for _, row in reduced])
                for (name, row), p_adj in zip(reduced, adjusted):
                    row["p_holm_reduced_layouts_within_task_site"] = float(p_adj)
            key = "tp4_minus_all19"
            if key in result["comparisons"]:
                primary_rows.append(result["comparisons"][key])
            for name, row in result["comparisons"].items():
                if name.startswith("group_pair_"):
                    equal_size_rows.append(row)
    _holm_into(primary_rows, "p_delong", "p_holm_primary_all_task_sites")
    _holm_into(equal_size_rows, "p_delong", "p_holm_equal_size_all_task_sites")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
