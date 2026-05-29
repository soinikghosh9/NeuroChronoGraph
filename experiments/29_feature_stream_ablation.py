"""Feature stream ablation experiment.

Loads the trained NeuroChronoGraph V2 checkpoint (with all four hand-crafted
biomarker streams -- spectral, connectivity, complexity, microstate --
fused into the final embedding via :class:`FeatureStreamFusion`) and
evaluates each stream's contribution by zeroing it at inference time on
the isolated hold-out set, using the geometric-mean chain rule that
collapses the three hierarchical heads into a four-class prediction
(matching the manuscript's evaluation protocol).

Outputs:
  outputs/results/feature_stream_ablation.csv
  outputs/results/feature_stream_ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

from src.config.config import DATALOADER_CONFIG, DATASET_CONFIG, DEVICE
from src.data.dataset_factory import DatasetFactory
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2

STREAMS = ["spectral", "connectivity", "complexity", "microstate"]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Class indices: 0=AD, 1=FTD, 2=CN, 3=MCI (matches src/data/dataset_wrapper.py)
CLASS_AD, CLASS_FTD, CLASS_CN, CLASS_MCI = 0, 1, 2, 3


def _hierarchical_to_4class(probs_screen, probs_stage, probs_subtype):
    """Apply geometric-mean chain rule (Eq. eq:gm-chain) to fuse the three
    binary hierarchical heads into a calibrated four-class prediction."""
    p_h = probs_screen[:, 0]            # P(CN)
    p_imp = probs_screen[:, 1]
    p_mci_given_imp = probs_stage[:, 0]
    p_dem_given_imp = probs_stage[:, 1]
    p_ad_given_dem = probs_subtype[:, 0]
    p_ftd_given_dem = probs_subtype[:, 1]

    eps = 1e-8
    p_cn = p_h
    p_mci = (p_imp * p_mci_given_imp + eps) ** (1.0 / 2.0)
    p_ad = (p_imp * p_dem_given_imp * p_ad_given_dem + eps) ** (1.0 / 3.0)
    p_ftd = (p_imp * p_dem_given_imp * p_ftd_given_dem + eps) ** (1.0 / 3.0)

    stack = np.stack([p_ad, p_ftd, p_cn, p_mci], axis=-1)
    stack = stack / (stack.sum(axis=-1, keepdims=True) + eps)
    return stack.argmax(axis=-1), stack


def evaluate(model, loader, mask):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            band = {k: v.to(DEVICE) for k, v in batch["band_features"].items()}
            streams = {k: v.to(DEVICE) for k, v in batch["feature_streams"].items()}
            meta = batch["metadata"].to(DEVICE)
            clinical = {"age": meta[:, 0], "sex": meta[:, 1], "mmse": meta[:, 2]}
            out = model(
                x,
                band_features=band,
                clinical_data=clinical,
                feature_streams=streams,
                stream_mask=mask,
            )
            ps = out["probs_screen"].cpu().numpy()
            pst = out["probs_stage"].cpu().numpy()
            psub = out["probs_subtype"].cpu().numpy()
            batch_preds, _ = _hierarchical_to_4class(ps, pst, psub)
            preds.extend(batch_preds.tolist())
            trues.extend(batch["label"].tolist())
    preds, trues = np.array(preds), np.array(trues)
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
        "f1_macro": float(f1_score(trues, preds, average="macro")),
        "kappa": float(cohen_kappa_score(trues, preds)),
    }


def main() -> None:
    dataset_paths = {
        "ds004504": PROJECT_ROOT / "datasets" / "openneuro_ds004504",
        "ds006036": PROJECT_ROOT / "datasets" / "ds006036",
        "Alz_EEG": PROJECT_ROOT / "datasets" / "Alz_EEG_data",
        "Mendeley": PROJECT_ROOT / "datasets" / "Mendeley Dataset",
        "MCI_Dataset": PROJECT_ROOT / "datasets" / "mci dataset",
    }

    factory = DatasetFactory()
    for name, path in dataset_paths.items():
        if path.exists():
            factory.add_dataset(name, path)

    cfg = dict(DATASET_CONFIG)
    cfg["compute_feature_streams"] = True
    dataset, groups, labels = factory.create_torch_datasets(config=cfg)
    train_idx, holdout_idx = factory.get_holdout_split(groups, labels, test_size=0.10)

    holdout_loader = DataLoader(
        Subset(dataset, holdout_idx), batch_size=16, shuffle=False, **DATALOADER_CONFIG
    )

    ckpt = PROJECT_ROOT / "outputs" / "models" / "ncg_v2_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {ckpt}. Train it first with "
            "`python experiments/18_train_hierarchical.py` -- the script saves "
            "the best CV checkpoint there once feature streams are enabled in "
            "DATASET_CONFIG (compute_feature_streams=True)."
        )

    model = create_neuro_chrono_graph_v2({
        "n_classes": 3,
        "n_times": cfg["n_times"],
        "hidden_dim": 128,
        "dropout": 0.4,
        "feature_streams": STREAMS,
        "stream_dim": 64,
    }).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)

    rows = []
    print("Stream-leave-one-out ablation:")
    full_mask = {s: True for s in STREAMS}
    base = evaluate(model, holdout_loader, full_mask)
    rows.append({"config": "all_streams", **base})
    print(f"  all streams: acc={base['accuracy']:.3f}  bacc={base['balanced_accuracy']:.3f}")

    for s in STREAMS:
        mask = dict(full_mask)
        mask[s] = False
        m = evaluate(model, holdout_loader, mask)
        rows.append({"config": f"-{s}", **m,
                     "delta_acc": m["accuracy"] - base["accuracy"]})
        print(f"  -{s:<13}: acc={m['accuracy']:.3f}  Δ={m['accuracy']-base['accuracy']:+.3f}")

    print("\nStream-only (single stream enabled):")
    for s in STREAMS:
        mask = {x: (x == s) for x in STREAMS}
        m = evaluate(model, holdout_loader, mask)
        rows.append({"config": f"only_{s}", **m})
        print(f"  only {s:<11}: acc={m['accuracy']:.3f}")

    print("\nNo stream baseline (raw + clinical only):")
    none_mask = {s: False for s in STREAMS}
    m = evaluate(model, holdout_loader, none_mask)
    rows.append({"config": "no_streams", **m})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "feature_stream_ablation.csv", index=False)
    with (RESULTS_DIR / "feature_stream_ablation.json").open("w") as f:
        json.dump({"streams": STREAMS, "results": rows,
                   "stream_gates": model.stream_fusion.gate_values()}, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR / 'feature_stream_ablation.csv'}")


if __name__ == "__main__":
    main()
