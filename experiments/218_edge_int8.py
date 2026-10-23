"""Edge deployment, measured rather than asserted (review W6/E9).

This legacy entry point now delegates to `240_edge_spectralite16.py`, which measures the released
16-feature SpectraLite. The old raw-window implementation remains below for audit history but is not
used for the paper output.

For the released SpectraLite (canonical checkpoints, seed 42, five folds) on each task:
  * model size on disk: FP32 state dict, and INT8 after post-training dynamic quantisation of every
    nn.Linear (torch.ao.quantization.quantize_dynamic, per-tensor symmetric weights);
  * external AUC of the FP32 vs INT8 fold ensemble on identical windows (does quantisation cost anything?);
  * single-thread CPU latency (torch.set_num_threads(1)) for one 4-s window and for a full subject
    decision (24 windows), and for spectral feature extraction alone, median of 200 (50) runs.
Graph-0.96M is measured the same way for comparison. On-microcontroller latency is NOT claimed; the
host is recorded so the numbers are interpretable.

    python experiments/218_edge_int8.py
Out: outputs/results/canonical/edge.json
"""
from __future__ import annotations

import importlib.util
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
CAN = ROOT / "outputs" / "results" / "canonical"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")


def size_bytes(model):
    import torch
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.tell()


def main():
    import torch
    torch.set_num_threads(1)
    e161 = C.load_161()
    Xw = np.load(C.windows_path("channel_clean"), mmap_mode="r")
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    groups = meta["groups"]; win_coh = np.array([str(x) for x in meta["cohort"]])
    win_cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    rep = {"host": {"cpu": platform.processor(), "machine": platform.machine(), "python": platform.python_version(),
                    "torch": torch.__version__, "threads": 1}, "tasks": {}}
    for task in TASKS:
        pos, neg, site = C.TASKS[task]
        keep = np.isin(win_cls, pos + neg) & (win_coh == site)
        idx = np.flatnonzero(keep); Xte = np.asarray(Xw[idx]); gte = groups[idx]
        te_sub = np.unique(gte); y = np.array([int(win_cls[np.flatnonzero(groups == s)[0]] in pos) for s in te_sub])
        tr = {}
        for name in ("spectral75k", "ncg_v5"):
            cdir = C.CKPT / task / name / "seed42"
            if not (cdir / "fold4.pt").exists():
                continue
            fp32, int8 = [], []
            for k in range(5):
                m = C.build(name, "cpu", e161); m.load_state_dict(torch.load(cdir / f"fold{k}.pt", map_location="cpu",
                                                                              weights_only=True)); m.eval()
                fp32.append(m)
                try:
                    q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
                    with torch.no_grad():                 # a quantised module that cannot run is not usable
                        e161.logits(q, torch.zeros(1, 19, 1024))
                    int8.append(q)
                except Exception as e:                                   # noqa: BLE001
                    print(f"   {name}: int8 not applicable ({type(e).__name__}: {str(e)[:80]})", flush=True)
                    int8 = None
                    break

            def ens(models):
                p = []
                with torch.no_grad():
                    for mm in models:
                        out = []
                        for i in range(0, len(Xte), 256):
                            out.append(torch.softmax(e161.logits(mm, torch.from_numpy(np.ascontiguousarray(Xte[i:i + 256])).float()), 1)[:, 1].numpy())
                        p.append(np.concatenate(out))
                return C.subject_mean(np.mean(p, 0), gte, te_sub)
            torch.set_num_threads(8)
            s32 = ens(fp32); s8 = ens(int8) if int8 else None
            torch.set_num_threads(1)
            x1 = torch.from_numpy(np.ascontiguousarray(Xte[:1])).float()
            x24 = torch.from_numpy(np.ascontiguousarray(Xte[:24])).float()

            def lat(fn, n=200):
                ts = []
                with torch.no_grad():
                    for _ in range(10):
                        fn()
                    for _ in range(n):
                        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
                return float(np.median(ts) * 1000)
            row = {"auc_fp32": C.auc(y, s32),
                   "size_fp32_bytes": size_bytes(fp32[0]),
                   "n_params": int(sum(p.numel() for p in fp32[0].parameters())),
                   "latency_ms_window_fp32": lat(lambda: e161.logits(fp32[0], x1)),
                   "latency_ms_subject24_fp32": lat(lambda: e161.logits(fp32[0], x24), 50)}
            if int8:
                row.update(auc_int8=C.auc(y, s8), max_abs_score_diff=float(np.max(np.abs(s32 - s8))),
                           size_int8_bytes=size_bytes(int8[0]),
                           latency_ms_window_int8=lat(lambda: e161.logits(int8[0], x1)),
                           latency_ms_subject24_int8=lat(lambda: e161.logits(int8[0], x24), 50))
            if name == "spectral75k":
                ex = fp32[0].spectral
                row["latency_ms_features_window"] = lat(lambda: ex(x1))
                row["latency_ms_features_subject24"] = lat(lambda: ex(x24), 50)
            tr[name] = row
            q = f"int8 {row['auc_int8']:.4f} | {row['size_int8_bytes']/1024:.0f} KiB | " \
                f"{row['latency_ms_subject24_int8']:.1f} ms" if "auc_int8" in row else "int8 n/a"
            print(f"{task:15s} {name:12s} AUC fp32 {row['auc_fp32']:.4f} ({q}) | size "
                  f"{row['size_fp32_bytes']/1024:.0f} KiB | subject latency "
                  f"{row['latency_ms_subject24_fp32']:.1f} ms | features "
                  f"{row.get('latency_ms_features_subject24', float('nan')):.1f} ms", flush=True)
        rep["tasks"][task] = tr
    (CAN / "edge.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("edge16", ROOT / "experiments" / "240_edge_spectralite16.py")
    edge16 = importlib.util.module_from_spec(spec); spec.loader.exec_module(edge16)
    edge16.main()
