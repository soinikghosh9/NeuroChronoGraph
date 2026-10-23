r"""Exact per-module trainable-parameter counts for the released model, the graph network and the ladder.

The supplement's budget table was maintained by hand and had already gone wrong once (the ablation was
described as stripping a 6.3M network when it strips a 964k one). This instantiates every model and
counts, so the table cannot drift from the code.

    python experiments/227_param_budget.py
Out: outputs/results/canonical/param_budget.json, model_paper/tables/param_budget.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "results" / "canonical"
TAB = ROOT / "model_paper" / "tables"


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def by_module(model, wanted):
    out = {}
    for name, mod in model.named_children():
        if name in wanted:
            out[name] = count(mod)
    return out


def main():
    import torch
    from src.models.spectral75k import LADDER, LadderNet, Spectral75k
    from src.models.v5.neuro_chrono_graph_v5 import NeuroChronoGraphV5
    torch.manual_seed(0)
    s75 = Spectral75k(n_classes=2)
    v5 = NeuroChronoGraphV5(n_channels=19, n_classes=2)
    rep = {"spectral75k": {"total": count(s75), "modules": {
        "spectral extractor (parameter-free)": count(s75.spectral),
        "node projection 12->128": count(s75.proj),
        "coordinate montage embedding": count(s75.montage),
        "gated attention pool over electrodes": count(s75.pool),
        "classifier head 128->128->2": count(s75.head)}},
        "graph_0_96M": {"total": count(v5), "modules": {
            "spectral extractor (parameter-free)": count(v5.spectral),
            "node projection 12->128": count(v5.spectral_proj),
            "coordinate montage embedding": count(v5.montage),
            "raw-waveform path": count(v5.raw_nodes) + count(v5.raw_proj),
            "band encoder (ablation only)": count(v5.band_encoder),
            "node LayerNorm": count(v5.node_norm),
            "wPLI band prior (ablation only)": count(v5.prior),
            "adaptive graph learning": count(v5.adaptive_graph),
            "gated graph convolutions (x2)": sum(count(g) for g in v5.gat_layers),
            "graph attention pool": count(v5.graph_pool),
            "lobe-coupling stream": count(v5.modular),
            "temporal stream": count(v5.temporal),
            "fusion LayerNorm": count(v5.embed_norm),
            "classifier head 384->128->2": count(v5.classifier)}},
        "ladder": {}}
    for k, kw in LADDER.items():
        rep["ladder"][k] = count(LadderNet(**kw))
    rep["ladder"]["full graph network (v5)"] = count(v5)

    # every trainable network in the benchmark, counted the same way, for the capacity correlation
    import importlib.util
    spec = importlib.util.spec_from_file_location("e161", ROOT / "experiments" / "161_deep_per_task.py")
    e161 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e161)
    allp = {"spectral75k": count(s75), "ncg_v5": count(v5)}
    for name in ("shallow", "deep4", "eegnet", "conformer", "atcnet", "eegnex", "eegitnet", "ncg_v2", "ncg_v3"):
        try:
            allp[name] = count(e161.make_binary(name, "cpu"))
        except Exception as e:                                           # noqa: BLE001
            print(f"  [skip] {name}: {type(e).__name__} {e}")
    for k, kw in LADDER.items():
        if k != "spectral":
            allp[f"ladder_{k}"] = count(LadderNet(**kw))
    rep["all_models"] = allp
    print("parameter counts:", json.dumps(allp, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "param_budget.json").write_text(json.dumps(rep, indent=1))

    s, g = rep["spectral75k"], rep["graph_0_96M"]
    keys = list(g["modules"])
    rows = []
    for k in keys:
        sv = s["modules"].get(k)
        note = "kept" if sv else ("ablation only" if "ablation" in k else "removed")
        if "parameter-free" in k:
            note = "shared, parameter-free"
        rows.append(f"{k} & {g['modules'][k]:,} & " + (f"{sv:,}" if sv else "--") + f" & {note} \\\\")
    rows = [r.replace(",", "{,}") for r in rows]
    TAB.mkdir(exist_ok=True)
    (TAB / "param_budget.tex").write_text("\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Trainable parameters per module, counted from the instantiated models"
        rf" ($d=128$, binary head). Spectral-75k is the front-end that Graph-0.96M contains:"
        rf" {s['total']:,} of its {g['total']:,} parameters"
        rf" ({100 * s['total'] / g['total']:.1f}\%).}}".replace(",", "{,}"),
        r"\label{tab:parambudget}", r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
        r"Module & Graph-0.96M & Spectral-75k & Note \\", r"\midrule", *rows, r"\midrule",
        (rf"\textbf{{Total}} & \textbf{{{g['total']:,}}} & \textbf{{{s['total']:,}}} & \\").replace(",", "{,}"),
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("wrote", TAB / "param_budget.tex")


if __name__ == "__main__":
    main()
