"""Mean electrode attention weights of the released SpectraLite (Fig. 1b uses these instead of illustrative values).

For each task, every seed x fold checkpoint (237) is applied to the windows of its held-out CAUEEG fold; the
masked gated attention weights a_c (one per electrode, summing to 1 per window) are averaged over a subject's
windows, then over subjects (all, controls, cases). Out-of-fold only, so no checkpoint scores its own training
subjects.
-> outputs/results/canonical/attention_weights.json

    python experiments/256_attention_weights.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("e237", ROOT / "experiments" / "237_canonical_spectralite16.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


def main():
    import torch
    from src.config.config import CHANNEL_NAMES
    from src.models.spectral75k import SpectraLite
    feats = E.load_feats()
    out = {"channels": list(CHANNEL_NAMES), "tasks": {}}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        ref = np.load(E.CAN / task / "spectralite16__seed42.npz", allow_pickle=True)
        subs = np.array([str(s) for s in ref["train_subjects"]]); y = ref["train_y"].astype(int)
        fold = ref["fold_of_subject"].astype(int)
        per_subject = {s: [] for s in subs}
        for seed in (42, 43, 44):
            for k in range(5):
                ck = E.CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{k}.pt"
                model = SpectraLite(n_classes=2, n_features=E.NFEAT)
                model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); model.eval()
                held = subs[fold == k]
                X, g = E.stack(held, y[fold == k], feats)
                with torch.no_grad():
                    _, w = model(node_features=torch.tensor(X), return_attention=True)
                w = w.reshape(len(X), -1).numpy()
                for s in held:
                    per_subject[s].append(w[g == s].mean(0))
        W = np.stack([np.mean(per_subject[s], 0) for s in subs])
        row = {"all": W.mean(0).tolist(), "controls": W[y == 0].mean(0).tolist(), "cases": W[y == 1].mean(0).tolist(),
               "n_subjects": int(len(subs)), "checkpoints": 15, "row_sum_check": float(W.sum(1).mean())}
        out["tasks"][task] = row
        top = np.argsort(W.mean(0))[::-1][:6]
        print(task, "top:", [(CHANNEL_NAMES[i], round(float(W.mean(0)[i]), 3)) for i in top],
              "uniform", round(1 / W.shape[1], 3), "sum", round(row["row_sum_check"], 3), flush=True)
    (E.CAN / "attention_weights.json").write_text(json.dumps(out, indent=1))


def main12():
    """Same summary for SpectraLite (12 features, spectral75k): its fold checkpoints compute the node features from
    the matched CAUEEG windows themselves. Written to attention_weights_spectralite12.json."""
    import torch
    from src.config.config import CHANNEL_NAMES
    from src.models.spectral75k import Spectral75k
    C = importlib.util.module_from_spec(importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py"))
    C.__spec__.loader.exec_module(C)
    Xw = np.load(C.windows_path("channel_clean"), mmap_mode="r")
    gw = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)["groups"].astype(str)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"channels": list(CHANNEL_NAMES), "model": "spectral75k", "tasks": {}}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        ref = np.load(E.CAN / task / "spectral75k__seed42.npz", allow_pickle=True)
        subs = np.array([str(s) for s in ref["train_subjects"]]); y = ref["train_y"].astype(int)
        fold = ref["fold_of_subject"].astype(int)
        per_subject = {s: [] for s in subs}
        for k in range(5):
            held = subs[fold == k]; idx = np.flatnonzero(np.isin(gw, held)); X = np.asarray(Xw[idx]); g = gw[idx]
            for seed in (42, 43, 44):
                model = Spectral75k(n_classes=2).to(dev)
                model.load_state_dict(torch.load(E.CKPT / task / "spectral75k" / f"seed{seed}" / f"fold{k}.pt",
                                                 map_location=dev, weights_only=True)); model.eval()
                ws = []
                with torch.no_grad():
                    for i in range(0, len(X), 512):
                        _, w = model(x=torch.from_numpy(X[i:i + 512]).float().to(dev), return_attention=True)
                        ws.append(w.reshape(min(512, len(X) - i), -1).float().cpu().numpy())
                w = np.concatenate(ws)
                for s in held:
                    per_subject[s].append(w[g == s].mean(0))
        W = np.stack([np.mean(per_subject[s], 0) for s in subs])
        out["tasks"][task] = {"all": W.mean(0).tolist(), "controls": W[y == 0].mean(0).tolist(), "cases": W[y == 1].mean(0).tolist(),
                              "n_subjects": int(len(subs)), "checkpoints": 15, "row_sum_check": float(W.sum(1).mean())}
        top = np.argsort(W.mean(0))[::-1][:6]
        print(task, "top:", [(CHANNEL_NAMES[i], round(float(W.mean(0)[i]), 3)) for i in top],
              "uniform", round(1 / W.shape[1], 3), flush=True)
    (E.CAN / "attention_weights_spectralite12.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--spectralite12" in sys.argv:
        main12()
    else:
        main()
