"""Package the canonical checkpoints for Hugging Face, one folder per model family.

The export is a format conversion, not retraining: every canonical PyTorch checkpoint written by
211/237 is loaded into its class (strict), saved as safetensors, reloaded, and compared tensor by tensor
with the source. Output (not tracked by git; uploaded separately, see the guide in the output folder):

    hf_release/<Family>/<model>/<task>/seed<s>/fold<k>.safetensors
    hf_release/<Family>/<model>/config.json
    hf_release/<Family>/manifest.json, SHA256SUMS.txt

    python experiments/241_export_hf_release.py [--families SpectraLite,NeuroChronoGraph]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles")) / "checkpoints" / "211"
OUT = ROOT / "hf_release"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired", "CN_vs_MCI")
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
RAW = "raw windows (batch, 19, 1024): 4 s at 256 Hz, robust-scaled per window and channel"


def spectralite(n):
    from src.models.spectral75k import SpectraLite
    return SpectraLite(n_features=n)


def ncg(version):
    if version == 2:
        from src.models.v2.neuro_chrono_graph_v2 import NeuroChronoGraphV2 as M
    elif version == 3:
        from src.models.v3.neuro_chrono_graph_v3 import NeuroChronoGraphV3 as M
    else:
        from src.models.v5.neuro_chrono_graph_v5 import NeuroChronoGraphV5 as M
    return M(n_channels=19, n_classes=2)


def ladder(**flags):
    from src.models.spectral75k import LadderNet
    return LadderNet(**flags)


# family -> model folder -> (source directory, constructor, constructor text, input, status)
FAMILIES = {
    "SpectraLite": {
        "spectralite": ("spectral75k", lambda: spectralite(12), "SpectraLite(n_features=12)",
                        RAW + "; or node_features (batch, 19, 12)", "reported model; fixed before external scoring"),
        "spectralite16": ("spectralite16", lambda: spectralite(16), "SpectraLite(n_features=16)",
                          "node_features (batch, 19, 16) from the 16-feature cache",
                          "exploratory variant; chosen after external scoring"),
    },
    "NeuroChronoGraph": {
        "ncg_v3": ("ncg_v3", lambda: ncg(3), "NeuroChronoGraphV3(n_channels=19, n_classes=2)", RAW,
                   "best NCG by development AUC; shown beside SpectraLite in the paper"),
        "ncg_v2": ("ncg_v2", lambda: ncg(2), "NeuroChronoGraphV2(n_channels=19, n_classes=2)", RAW,
                   "earlier version; hierarchical head, screening output trained"),
        "ncg_v5": ("ncg_v5", lambda: ncg(5), "NeuroChronoGraphV5(n_channels=19, n_classes=2)", RAW,
                   "SpectraLite trunk plus raw, graph, lobe, and temporal streams"),
        "ladder_raw": ("ladder_spectral+raw", lambda: ladder(raw=True), "LadderNet(raw=True)", RAW,
                       "ablation: SpectraLite + raw-waveform stream"),
        "ladder_graph": ("ladder_spectral+graph", lambda: ladder(graph=True), "LadderNet(graph=True)", RAW,
                         "ablation: SpectraLite + graph stream"),
        "ladder_raw_graph": ("ladder_spectral+raw+graph", lambda: ladder(raw=True, graph=True),
                             "LadderNet(raw=True, graph=True)", RAW, "ablation: + raw and graph streams"),
        "ladder_raw_graph_lobe": ("ladder_spectral+raw+graph+lobe", lambda: ladder(raw=True, graph=True, lobe=True),
                                  "LadderNet(raw=True, graph=True, lobe=True)", RAW,
                                  "ablation: + raw, graph, and lobe streams"),
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_family(family: str) -> None:
    fam_dir = OUT / family
    files, models = [], []
    for folder, (src_name, make, ctor, inputs, status) in FAMILIES[family].items():
        tasks = [t for t in TASKS if (SOURCE / t / src_name).is_dir()]
        n_params = None
        for task in tasks:
            for seed in SEEDS:
                for fold in FOLDS:
                    src = SOURCE / task / src_name / f"seed{seed}" / f"fold{fold}.pt"
                    state = torch.load(src, map_location="cpu", weights_only=True)
                    state = state.get("state_dict", state)
                    model = make()
                    model.load_state_dict(state, strict=True)
                    n_params = sum(p.numel() for p in model.parameters())
                    tensors = {k: v.detach().contiguous().clone() for k, v in state.items()}
                    dst = fam_dir / folder / task / f"seed{seed}" / f"fold{fold}.safetensors"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(dst), metadata={"model": folder, "task": task, "seed": str(seed),
                                                           "fold": str(fold), "constructor": ctor})
                    back = load_file(str(dst))
                    if back.keys() != tensors.keys() or any(not torch.equal(back[k], tensors[k]) for k in back):
                        raise SystemExit(f"round-trip mismatch: {dst}")
                    files.append({"path": dst.relative_to(fam_dir).as_posix(), "bytes": dst.stat().st_size,
                                  "sha256": sha256(dst), "model": folder, "task": task, "seed": seed, "fold": fold})
        cfg = {"model": folder, "constructor": ctor, "parameters": n_params, "input": inputs, "status": status,
               "output": "logits (batch, 2); class 1 = case", "tasks": tasks, "seeds": list(SEEDS),
               "folds": list(FOLDS), "subject_score": "mean over 24 windows of softmax(logits)[:, 1]",
               "external_scoring": "mean of the five fold models per seed, then mean over the three seeds",
               "source_checkpoints": f"checkpoints/211/<task>/{src_name}/seed<s>/fold<k>.pt"}
        (fam_dir / folder / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        models.append(cfg)
        print(f"{family}/{folder}: {len(tasks)} tasks, {n_params:,} parameters", flush=True)
    manifest = {"family": family, "models": models,
                "training": {"cohort": "CAUEEG", "subject_grouped_folds": 5, "seeds": list(SEEDS), "epochs": 25,
                             "batch_size": 128, "optimizer": "AdamW", "max_learning_rate": 0.003,
                             "weight_decay": 0.0001, "loss": "inverse-class-weighted cross-entropy",
                             "selection": "final epoch; no validation-based selection"},
                "files": files}
    (fam_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (fam_dir / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files), encoding="utf-8")
    print(f"{family}: {len(files)} files", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()
    if not SOURCE.exists():
        raise SystemExit(f"canonical checkpoint root not found: {SOURCE}")
    for fam in args.families.split(","):
        export_family(fam)


if __name__ == "__main__":
    main()
