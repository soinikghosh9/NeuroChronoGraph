"""Subject-level localization and cross-site validation for SpectraLite-16.

Question: are the model's apparently useful posterior temporal/parietal sensors and the
associated case-control spectral patterns reproducible outside CAUEEG?

Design:
* Reuse, without retraining, the canonical 75,971-parameter 16-feature checkpoints from 237.
* For each sensor, mask its attention-pooling contribution and measure paired subject-level
  AUC loss. Also test the prespecified T5/T6 pair and T5/T6/P3/P4 posterior-temporal/parietal
  set. These are feature-pooling ablations: raw signals are not re-referenced or reprocessed.
* Train fixed C=1 logistic probes on one electrode's 16 features using only the canonical
  CAUEEG folds. On age-complete sites, compare age+sensor with an age-only source-trained
  model. GENEEG has no subject-level age; its sensor-only results are explicitly unadjusted.
* Test each electrode-feature's subject-level case-control difference with age and age^2
  covariates when individual ages are available. Use HC3 standard errors, BH across the
  19 x 16 feature family, and conservative Bonferroni-within-electrode then Holm-across-
  electrodes for electrode-level tests.
* External labels are used only for evaluation/association tests, never to select sensors,
  fit the logistic probes, or alter SpectraLite. ds004504 montage comparisons already exist;
  P-ADIC, BrainLat, and GENEEG provide the additional-site checks where their tasks permit.

GENEEG lacks T5/T6 and individual age. Those positions are excluded from GENEEG sensor tests.
The report distinguishes predictive reliance from anatomical/causal significance.

Run: python experiments/246_electrode_validation_spectralite16.py
Output: outputs/results/canonical/electrode_validation_spectralite16.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, rankdata, t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load_module("canonical_211_electrode246", "211_canonical_benchmark.py")
E237 = load_module("spectralite_237_electrode246", "237_canonical_spectralite16.py")
A204 = load_module("age_204_electrode246", "204_age_conditioned.py")
from src.models.spectral75k import SpectraLite
from src.config.config import CHANNEL_NAMES

CAN = C.OUT
CH = list(CHANNEL_NAMES)
TASKS = tuple(C.TASKS)
SEEDS = (42, 43, 44)
TP_PAIR = ("T5", "T6")
TP_SET = ("T5", "T6", "P3", "P4")
BOOTSTRAPS = 1000
BATCH = 512


def auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if len(np.unique(y)) != 2:
        return float("nan")
    r = rankdata(score, method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong_paired(y, a, b):
    """Asymptotic paired DeLong test for two correlated subject-level AUCs."""
    y = np.asarray(y, dtype=int); a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    pos, neg = y == 1, y == 0
    if pos.sum() < 2 or neg.sum() < 2:
        return float("nan")
    p = np.column_stack([a[pos], b[pos]])
    n = np.column_stack([a[neg], b[neg]])
    cmp = p[:, None, :] - n[None, :, :]
    phi = (cmp > 0).astype(float) + 0.5 * (cmp == 0)
    v10 = phi.mean(axis=1)
    v01 = phi.mean(axis=0)
    cov = np.cov(v10, rowvar=False, ddof=1) / len(p)
    cov += np.cov(v01, rowvar=False, ddof=1) / len(n)
    delta = auc(y, a) - auc(y, b)
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not np.isfinite(var) or var <= 1e-15:
        return 1.0 if abs(delta) < 1e-12 else 0.0
    return float(2 * norm.sf(abs(delta) / np.sqrt(var)))


def bootstrap_delta(y, a, b, n_boot, seed):
    """Paired subject bootstrap CI for AUC(a)-AUC(b)."""
    y = np.asarray(y, dtype=int); a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    point = auc(y, a) - auc(y, b)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(auc(y[ix], a[ix]) - auc(y[ix], b[ix]))
    if not vals:
        return point, [float("nan"), float("nan")]
    return point, [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def bootstrap_auc(y, score, n_boot, seed):
    point = auc(y, score)
    rng = np.random.default_rng(seed); y = np.asarray(y, int); score = np.asarray(score, float)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) == 2:
            vals.append(auc(y[ix], score[ix]))
    return point, [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def holm(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def bh(pvalues):
    p = np.asarray(pvalues, float)
    q = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if not len(good):
        return q
    order = good[np.argsort(p[good])]
    vals = p[order] * len(order) / np.arange(1, len(order) + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.minimum(vals, 1.0)
    return q


def load_ages():
    dev_ds = A204.load_ages()
    ppath = CAN / "padic_subject_meta.json"
    bpath = CAN / "brainlat_subject_meta.json"
    padic = json.loads(ppath.read_text(encoding="utf-8")) if ppath.exists() else {}
    brain = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else {}
    return dev_ds, padic, brain


def age_for(sid, cohort, maps):
    dev_ds, padic, brain = maps
    sid = str(sid)
    if cohort in ("CAUEEG", "ds004504_raw"):
        key = A204.keyf(sid, cohort)
        value = dev_ds.get(key) if key is not None else None
    elif cohort == "P-ADIC":
        row = padic.get(sid, {})
        value = row.get("age")
    elif cohort == "BrainLat":
        row = brain.get(sid, {})
        value = row.get("age")
    else:  # GENEEG publishes only a cohort mean; do not treat it as subject-level age.
        value = None
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_sets(z, task):
    rows = {
        "CAUEEG_OOF": ("train_subjects", "train_y", "CAUEEG", "fold_of_subject"),
        "ds004504": ("ext_subjects", "ext_y", "ds004504_raw", None),
        "P-ADIC": ("padic_subjects", "padic_y", "P-ADIC", None),
        "P-ADIC_all_controls": ("padic_all_subjects", "padic_all_y", "P-ADIC", None),
        "BrainLat": ("brainlat_subjects", "brainlat_y", "BrainLat", None),
    }
    if task == "CN_vs_MCI":
        rows["GENEEG"] = rows.pop("ds004504")
        rows["GENEEG"] = ("ext_subjects", "ext_y", "GENEEG", None)
    out = {}
    for name, (ks, ky, cohort, kfold) in rows.items():
        if ks not in z or ky not in z:
            continue
        out[name] = {
            "subjects": np.asarray([str(x) for x in z[ks]]),
            "y": np.asarray(z[ky], dtype=int),
            "cohort": cohort,
            "fold": np.asarray(z[kfold], dtype=int) if kfold else None,
        }
    return out


def subject_matrix(subjects, feature_map):
    miss = [str(s) for s in subjects if str(s) not in feature_map]
    if miss:
        raise KeyError(f"Missing cached 16-feature tensors for {len(miss)} subjects; first={miss[:5]}")
    return np.stack([np.nan_to_num(np.nanmedian(feature_map[str(s)], axis=0), nan=0.0,
                                   posinf=0.0, neginf=0.0) for s in subjects]).astype(np.float32)


def make_masks(available):
    full = np.ones(len(CH), dtype=bool)
    masks = [("full", full.copy())]
    present = set(available)
    for ch in CH:
        if ch in present:
            m = full.copy(); m[CH.index(ch)] = False
            masks.append((f"drop_{ch}", m))
    for label, group in (("only_T5_T6", TP_PAIR), ("drop_T5_T6", TP_PAIR),
                         ("only_T5_T6_P3_P4", TP_SET), ("drop_T5_T6_P3_P4", TP_SET)):
        if set(group).issubset(present):
            m = full.copy()
            if label.startswith("only_"):
                m[:] = False
            for ch in group:
                m[CH.index(ch)] = label.startswith("only_")
            masks.append((label, m))
    return masks


@torch.inference_mode()
def predict_masks(model, X, masks, device, check=False):
    model.eval()
    mask_t = torch.as_tensor(np.stack([m for _, m in masks]), device=device, dtype=torch.bool)
    outputs = []
    for start in range(0, len(X), BATCH):
        xb = torch.as_tensor(np.ascontiguousarray(X[start:start + BATCH]), device=device,
                              dtype=torch.float32)
        node, default_mask, _ = model.node_embed(node_features=xb)
        pool = model.pool
        att_logits = pool.w(torch.tanh(pool.V(node)) * torch.sigmoid(pool.U(node))).squeeze(-1)
        effective = mask_t & default_mask[None, :]
        expanded = att_logits[:, None, :].expand(-1, len(masks), -1)
        expanded = expanded.masked_fill(~effective[None, :, :], float("-inf"))
        weights = torch.softmax(expanded, dim=-1)
        pooled = torch.einsum("bmc,bcd->bmd", weights, node)
        logits = model.head(pooled.reshape(-1, pooled.shape[-1])).reshape(len(xb), len(masks), 2)
        probs = torch.softmax(logits, dim=-1)[..., 1]
        if check:
            direct = torch.softmax(model(node_features=xb), dim=1)[:, 1]
            err = float(torch.max(torch.abs(probs[:, 0] - direct)).cpu())
            if err > 2e-6:
                raise AssertionError(f"manual masked-pooling full path differs from forward by {err}")
        outputs.append(probs.cpu().numpy())
        check = False
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(masks)), np.float32)


def score_subjects(model, subjects, feature_map, masks, device, check=False):
    blocks = [feature_map[str(s)] for s in subjects]
    X = np.concatenate(blocks).astype(np.float32, copy=False)
    ends = np.cumsum([len(x) for x in blocks])
    starts = np.r_[0, ends[:-1]]
    win = predict_masks(model, X, masks, device, check=check)
    return np.stack([win[a:b].mean(axis=0) for a, b in zip(starts, ends)]).astype(np.float32)


def model_masking(task, sets, shard_by_seed, feature_map, device, n_boot):
    names = {}
    for site, data in sets.items():
        # GENEEG has only 17 native channels; never interpret its zero-filled T5/T6 as evidence.
        available = tuple(c for c in CH if not (data["cohort"] == "GENEEG" and c in ("T5", "T6")))
        masks = make_masks(available)
        names[site] = (available, masks)

    pred_by_site = {site: [] for site in sets}
    checks = {}
    for seed in SEEDS:
        z = shard_by_seed[seed]
        for site, data in sets.items():
            _, masks = names[site]
            n = len(data["subjects"])
            by_fold = []
            oof = np.full((n, len(masks)), np.nan, dtype=np.float32) if site == "CAUEEG_OOF" else None
            for fold in range(5):
                ck = C.CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{fold}.pt"
                if not ck.exists():
                    raise FileNotFoundError(f"Missing canonical SpectraLite checkpoint: {ck}")
                model = SpectraLite(n_classes=2, n_features=16).to(device)
                model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
                model.eval()
                if site == "CAUEEG_OOF":
                    idx = np.flatnonzero(data["fold"] == fold)
                    vals = score_subjects(model, data["subjects"][idx], feature_map, masks,
                                          device, check=(fold == 0 and seed == 42))
                    oof[idx] = vals
                else:
                    vals = score_subjects(model, data["subjects"], feature_map, masks, device,
                                          check=(fold == 0 and seed == 42))
                    by_fold.append(vals)
                del model
            raw = oof if site == "CAUEEG_OOF" else np.mean(by_fold, axis=0)
            # Confirm the unmasked path reproduces the published subject-level shard.
            expected_key = {"CAUEEG_OOF": "oof_score", "ds004504": "ext_score", "GENEEG": "ext_score",
                            "P-ADIC": "padic_score", "P-ADIC_all_controls": "padic_all_score",
                            "BrainLat": "brainlat_score"}[site]
            expected = np.asarray(z[expected_key], dtype=float)
            err = float(np.max(np.abs(raw[:, 0] - expected))) if len(expected) else 0.0
            checks[f"{site}_seed{seed}"] = err
            if err > 5e-4:
                raise AssertionError(f"{task} {site} seed{seed}: full-mask score mismatch {err:.3g}")
            pred_by_site[site].append(raw)
            if device == "cuda":
                torch.cuda.empty_cache()

    result = {}
    for site, data in sets.items():
        _, masks = names[site]
        scores = np.mean(pred_by_site[site], axis=0)
        y = data["y"]
        full_auc = auc(y, scores[:, 0])
        row = {"n": int(len(y)), "n_cn": int((y == 0).sum()), "n_positive": int((y == 1).sum()),
               "auc_full": full_auc, "mask_tests": {}, "available_channels": list(names[site][0])}
        loo_keys = [f"drop_{ch}" for ch in names[site][0]]
        loo_ps = []
        rngbase = zlib.crc32(f"{task}:{site}".encode())
        for j, (condition, _) in enumerate(masks):
            if condition == "full":
                continue
            masked_auc = auc(y, scores[:, j])
            delta, ci = bootstrap_delta(y, scores[:, 0], scores[:, j], n_boot,
                                        rngbase + 101 * j)
            p = delong_paired(y, scores[:, 0], scores[:, j])
            row["mask_tests"][condition] = {"auc_masked": masked_auc,
                                             "delta_full_minus_masked": delta,
                                             "ci95": ci, "p_delong": p}
            if condition.startswith("drop_") and condition in loo_keys:
                loo_ps.append((condition, p))
        q = holm([p for _, p in loo_ps])
        for (condition, _), adj in zip(loo_ps, q):
            row["mask_tests"][condition]["p_holm_19_sensors"] = float(adj)
        result[site] = row
    return result, checks


def feature_associations(site, subjects, y, feature_map, age_maps, n_boot, seed):
    Y = subject_matrix(subjects, feature_map).reshape(len(subjects), -1).astype(float)
    ages = np.asarray([age_for(s, site, age_maps) for s in subjects], float)
    age_adjusted = int(np.isfinite(ages).sum()) >= max(40, int(0.80 * len(subjects)))
    keep = np.isfinite(ages) if age_adjusted else np.ones(len(subjects), bool)
    Y, yy, ages = Y[keep], np.asarray(y, int)[keep], ages[keep]
    if len(np.unique(yy)) < 2 or min((yy == 0).sum(), (yy == 1).sum()) < 8:
        return {"skipped": "insufficient complete subjects"}
    if age_adjusted:
        ac = ages - np.mean(ages)
        D = np.column_stack([np.ones(len(yy)), yy, ac / 10.0, (ac / 10.0) ** 2])
        adjustment = "individual age and age squared"
    else:
        D = np.column_stack([np.ones(len(yy)), yy])
        adjustment = "unadjusted; individual age unavailable or <80% complete"
    inv = np.linalg.pinv(D.T @ D)
    coef = inv @ D.T @ Y
    resid = Y - D @ coef
    h = np.clip(np.einsum("ij,ji->i", D @ inv, D.T), 0.0, 0.999999)
    c = D @ inv[:, 1]
    se = np.sqrt(np.sum((c[:, None] * resid / (1.0 - h[:, None])) ** 2, axis=0))
    df = max(1, len(yy) - np.linalg.matrix_rank(D))
    tstat = np.divide(coef[1], se, out=np.zeros_like(se), where=se > 0)
    p = 2.0 * student_t.sf(np.abs(tstat), df)
    scale = np.std(Y, axis=0, ddof=1)
    effect = np.divide(coef[1], scale, out=np.zeros_like(scale), where=scale > 0)
    present = set(CH)
    if site == "GENEEG":
        present -= {"T5", "T6"}
    valid_feature = np.asarray([i for ci, ch in enumerate(CH) if ch in present
                                for i in range(ci * 16, (ci + 1) * 16)], dtype=int)
    q = np.full_like(p, np.nan)
    q[valid_feature] = bh(p[valid_feature])
    pmat = p.reshape(19, 16); qmat = q.reshape(19, 16); emat = effect.reshape(19, 16)
    for i, ch in enumerate(CH):
        if ch not in present:
            pmat[i] = np.nan
            emat[i] = np.nan
    # Bonferroni within the 16 features at each sensor, then Holm across the 19 sensors.
    sensor_p = np.full(19, np.nan)
    for i, ch in enumerate(CH):
        if ch in present:
            sensor_p[i] = min(1.0, float(np.min(pmat[i]) * 16))
    sensor_holm = holm(sensor_p)
    sensor_rows = []
    feature_names = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma", "IAF",
                     "alpha_prominence", "spectral_edge", "aperiodic_exponent", "theta_alpha",
                     "slowing", "spectral_entropy", "IAF_ref_theta", "IAF_ref_alpha",
                     "aperiodic_offset", "alpha_asymmetry"]
    for i, ch in enumerate(CH):
        if ch not in present:
            continue
        order = np.argsort(qmat[i])
        sensor_rows.append({"electrode": ch, "p_sensor_bonferroni16": float(sensor_p[i]),
                            "p_sensor_holm19": float(sensor_holm[i]),
                            "top_feature_effects_sd": [
                                {"feature": feature_names[j], "effect_case_minus_cn_sd": float(emat[i, j]),
                                 "p_hc3": float(pmat[i, j]), "q_bh_304": float(qmat[i, j])}
                                for j in order[:4]]})
    sensor_rows.sort(key=lambda r: (r["p_sensor_holm19"], -abs(r["top_feature_effects_sd"][0]["effect_case_minus_cn_sd"])))
    return {"n_complete": int(len(yy)), "age_adjusted": age_adjusted, "adjustment": adjustment,
            "feature_family": "19 electrodes x 16 features; HC3 OLS; BH q across 304 tests",
            "electrode_family": "Bonferroni across 16 features per electrode, then Holm across 19 electrodes",
            "n_feature_tests": int(len(valid_feature)), "n_features_q_bh_lt_0_05": int(np.sum(q < .05)),
            "n_electrodes_holm_lt_0_05": int(np.sum(sensor_holm < .05)),
            "electrodes": sensor_rows,
            "effect_matrix_sd": emat.tolist(), "p_matrix": pmat.tolist(), "q_matrix": qmat.tolist()}


def fit_logit(X, y):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                               solver="lbfgs", max_iter=2000))
    pipe.fit(X, y)
    return pipe


def single_sensor_predictions(task, sets, feature_map, age_maps):
    source = sets["CAUEEG_OOF"]
    tr_sub, ytr, folds = source["subjects"], source["y"], source["fold"]
    Xtr = subject_matrix(tr_sub, feature_map)
    atr = np.asarray([age_for(s, "CAUEEG", age_maps) for s in tr_sub], float)
    age_trainable = np.isfinite(atr).sum() >= 0.80 * len(atr)
    age_oof = np.full(len(tr_sub), np.nan, np.float32)
    sensor_oof = np.full((len(tr_sub), len(CH)), np.nan, np.float32)
    age_sensor_oof = np.full((len(tr_sub), len(CH)), np.nan, np.float32)
    ext_pred_sensor = {site: np.zeros((len(data["y"]), len(CH)), np.float32)
                       for site, data in sets.items() if site != "CAUEEG_OOF"}
    ext_pred_age = {site: np.zeros_like(v) for site, v in ext_pred_sensor.items()}
    ext_pred_age_sensor = {site: np.zeros_like(v) for site, v in ext_pred_sensor.items()}
    age_obs = {}
    Xsite = {}
    for site, data in sets.items():
        if site == "CAUEEG_OOF":
            continue
        Xsite[site] = subject_matrix(data["subjects"], feature_map)
        age_obs[site] = np.asarray([age_for(s, data["cohort"], age_maps)
                                    for s in data["subjects"]], float)

    for fold in range(5):
        fit = folds != fold; held = folds == fold
        for ch_i in range(len(CH)):
            fit_all = fit
            model = fit_logit(Xtr[fit_all, ch_i, :], ytr[fit_all])
            sensor_oof[held, ch_i] = model.predict_proba(Xtr[held, ch_i, :])[:, 1]
            for site in ext_pred_sensor:
                ext_pred_sensor[site][:, ch_i] += model.predict_proba(Xsite[site][:, ch_i, :])[:, 1] / 5.0
            if age_trainable:
                fit_age = fit & np.isfinite(atr)
                held_age = held & np.isfinite(atr)
                aug_tr = np.column_stack([atr[fit_age], atr[fit_age] ** 2, Xtr[fit_age, ch_i, :]])
                aug_model = fit_logit(aug_tr, ytr[fit_age])
                aug_held = np.column_stack([atr[held_age], atr[held_age] ** 2, Xtr[held_age, ch_i, :]])
                age_sensor_oof[held_age, ch_i] = aug_model.predict_proba(aug_held)[:, 1]
                for site in ext_pred_sensor:
                    a = age_obs[site]
                    if np.isfinite(a).sum() >= max(30, 0.8 * len(a)):
                        ok = np.isfinite(a)
                        aug_eval = np.column_stack([a[ok], a[ok] ** 2, Xsite[site][ok, ch_i, :]])
                        ext_pred_age_sensor[site][ok, ch_i] += aug_model.predict_proba(aug_eval)[:, 1] / 5.0
                if ch_i == 0:
                    age_model = fit_logit(np.column_stack([atr[fit_age], atr[fit_age] ** 2]), ytr[fit_age])
                    age_oof[held_age] = age_model.predict_proba(np.column_stack(
                        [atr[held_age], atr[held_age] ** 2]))[:, 1]
                    for site in ext_pred_sensor:
                        a = age_obs[site]
                        if np.isfinite(a).sum() >= max(30, 0.8 * len(a)):
                            ok = np.isfinite(a)
                            ext_pred_age[site][ok, :] += age_model.predict_proba(
                                np.column_stack([a[ok], a[ok] ** 2]))[:, 1, None] / 5.0

    out = {}
    for site, data in sets.items():
        if site == "CAUEEG_OOF":
            sensor = sensor_oof
            sensor_age = age_sensor_oof
            base_age = age_oof
            cohort = "CAUEEG"
        else:
            sensor = ext_pred_sensor[site]
            sensor_age = ext_pred_age_sensor[site]
            base_age = ext_pred_age[site][:, 0] if age_trainable else np.full(len(data["y"]), np.nan)
            cohort = data["cohort"]
        # `ext_pred_age` is initialized with zeros for sites where no individual ages are
        # available. Do not mistake those placeholders for valid age-only predictions (notably
        # GENEEG, which publishes only a cohort mean). Check observed target ages directly.
        observed_age = (atr if site == "CAUEEG_OOF" else age_obs[site])
        age_available = (age_trainable and
                         np.isfinite(observed_age).sum() >= max(30, .8 * len(data["y"])))
        if age_available:
            selected = sensor_age
            base = base_age
            evaluation = "source-trained logistic: age + age^2 + one electrode's 16 descriptors"
        else:
            selected = sensor
            base = None
            evaluation = "source-trained logistic: one electrode's 16 descriptors; unadjusted"
        pvals = []
        rows = []
        for ch_i, ch in enumerate(CH):
            if cohort == "GENEEG" and ch in ("T5", "T6"):
                continue
            s = selected[:, ch_i]
            if age_available:
                delta = auc(data["y"], s) - auc(data["y"], base)
                p = delong_paired(data["y"], s, base)
                pvals.append(p)
            else:
                delta, p = None, None
            rows.append({"electrode": ch, "auc": auc(data["y"], s),
                         "auc95": bootstrap_auc(data["y"], s, 1000,
                                                 zlib.crc32(f"solo:{task}:{site}:{ch}".encode()))[1],
                         "delta_auc_vs_age_only": delta, "p_delong_vs_age_only": p})
        if age_available:
            adj = holm(pvals)
            for row, p_holm in zip(rows, adj):
                row["p_holm19_vs_age_only"] = float(p_holm)
            age_auc = auc(data["y"], base)
        else:
            age_auc = None
        rows.sort(key=lambda x: -x["auc"])
        out[site] = {"method": evaluation, "age_only_auc": age_auc,
                     "n_test": int(len(data["y"])), "electrodes": rows}
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; refusing a silent CPU model audit")
    device = args.device
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    age_maps = load_ages()
    feature_map = E237.load_feats()
    report = {
        "model": "SpectraLite", "parameters": 75971, "features_per_electrode": 16,
        "source": "CAUEEG; existing five subject-grouped folds and seeds 42,43,44",
        "methods": {
            "masking": "fixed-checkpoint attention-pooling mask; original 16 features unchanged; no raw re-reference",
            "primary_region": list(TP_SET), "secondary_region": list(TP_PAIR),
            "disease_features": "subject median across windows; OLS case coefficient with age and age^2 when available; HC3 SE; BH over 304 feature tests; Bonferroni x16 then Holm x19 for electrode tests",
            "single_sensor": "C=1 balanced logistic; scaler fitted within each CAUEEG fold; age+age^2+sensor compared with age-only where individual age exists",
            "inference": "paired DeLong tests; paired subject bootstrap percentile 95% CIs",
            "external_use": "external labels never enter model training, sensor selection, or preprocessing",
            "limits": ["sensor masking does not re-reference EEG or validate an end-to-end reduced montage",
                       "GENEEG has no per-subject age and lacks T5/T6; its feature associations are unadjusted",
                       "sex was not included because coverage/coding is incomplete across sites",
                       "the external-site cohort is not a prospective clinical sample"]},
        "tasks": {}, "runtime_device": device, "n_bootstrap": int(args.bootstraps)}
    for task in [x for x in args.tasks.split(",") if x]:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task}; expected one of {TASKS}")
        shard_by_seed = {}
        for seed in SEEDS:
            path = CAN / task / f"spectralite16__seed{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Canonical SpectraLite-16 result shard missing: {path}")
            shard_by_seed[seed] = dict(np.load(path, allow_pickle=True))
        sets = get_sets(shard_by_seed[42], task)
        missing = sorted({str(s) for d in sets.values() for s in d["subjects"] if str(s) not in feature_map})
        if missing:
            raise KeyError(f"{task}: feature cache lacks {len(missing)} evaluated subjects: {missing[:8]}")
        print(f"\n[{task}] {len(sets)} cohort/task sets | {device}", flush=True)
        masking, equivalence = model_masking(task, sets, shard_by_seed, feature_map, device,
                                             args.bootstraps)
        task_out = {"sites": {}, "full_path_max_abs_error_vs_canonical": equivalence}
        solo = single_sensor_predictions(task, sets, feature_map, age_maps)
        for site, data in sets.items():
            feat_assoc = feature_associations(data["cohort"], data["subjects"], data["y"],
                                              feature_map, age_maps, args.bootstraps,
                                              zlib.crc32(f"feat:{task}:{site}".encode()))
            task_out["sites"][site] = {
                "n": int(len(data["y"])), "n_cn": int((data["y"] == 0).sum()),
                "n_positive": int((data["y"] == 1).sum()),
                "placement_validation_status": ("previously queried in the ds004504 montage experiment"
                                                if site == "ds004504" else
                                                ("additional external site for the electrode-level audit"
                                                 if site not in ("CAUEEG_OOF", "P-ADIC_all_controls") else
                                                 ("internal source-cohort OOF" if site == "CAUEEG_OOF" else
                                                  "same P-ADIC site; sensitivity cohort with all controls"))),
                "model_masking": masking[site], "age_adjusted_feature_tests": feat_assoc,
                "single_electrode_prediction": solo[site]}
            loo = sorted(((k, v["delta_full_minus_masked"])
                          for k, v in masking[site]["mask_tests"].items()
                          if k.startswith("drop_") and k.count("_") == 1),
                         key=lambda item: item[1], reverse=True)
            print(f"  {site:22s} n={len(data['y']):4d} full AUC={masking[site]['auc_full']:.3f} "
                  f"top mask deltas={loo[:3]}", flush=True)
        report["tasks"][task] = task_out
        if device == "cuda":
            torch.cuda.empty_cache()
    # The two pre-specified ROI ablations are adjusted across every task/site comparison.
    for condition in ("drop_T5_T6_P3_P4", "drop_T5_T6", "only_T5_T6_P3_P4", "only_T5_T6"):
        refs = []
        for task, task_out in report["tasks"].items():
            for site, site_out in task_out["sites"].items():
                test = site_out["model_masking"]["mask_tests"].get(condition)
                if test is not None and np.isfinite(test["p_delong"]):
                    refs.append((task, site, test))
        adjusted = holm([x[2]["p_delong"] for x in refs])
        for (task, site, test), p_adj in zip(refs, adjusted):
            test["p_holm_across_tasks_sites"] = float(p_adj)
    dest = CAN / "electrode_validation_spectralite16.json"
    from src.utils.provenance import write_result
    write_result(dest, report, script=Path(__file__).name)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
