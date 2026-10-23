"""CPU edge measurements for the released 16-feature SpectraLite.

The released model consumes cached 16 per-electrode features. This measures the model forward pass,
including post-training dynamic INT8 quantisation, on the canonical external subjects. Feature extraction
is an upstream preprocessing step and is reported as such; this is not a microcontroller measurement.

    python experiments/240_edge_spectralite16.py
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
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired", "CN_vs_MCI")


def size_bytes(model):
    import torch
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.tell()


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    import torch
    from src.models.spectral75k import SpectraLite

    torch.set_num_threads(1)
    fdir = C.FEAT / "channel_clean"
    meta = np.load(fdir / "meta.npz", allow_pickle=False)
    groups = np.array([str(x) for x in meta["groups"]])
    node16 = np.concatenate([np.load(fdir / "node12.npy", mmap_mode="r"),
                             np.load(fdir / "node_ext4.npy", mmap_mode="r")], -1)
    rep = {"host": {"cpu": platform.processor(), "machine": platform.machine(),
                     "python": platform.python_version(), "torch": torch.__version__, "threads": 1},
           "input": "precomputed channel_clean node12 + node_ext4 features (19 electrodes x 16 features)",
           "tasks": {}}

    for task in TASKS:
        shard = CAN / task / "spectralite16__seed42.npz"
        if not shard.exists():
            continue
        z = np.load(shard, allow_pickle=True)
        te_sub, y = z["ext_subjects"], z["ext_y"]
        blocks, gte = [], []
        for s in te_sub:
            mask = groups == str(s)
            blocks.append(np.asarray(node16[mask], dtype=np.float32))
            gte.extend([str(s)] * int(mask.sum()))
        Xte = np.concatenate(blocks, 0); gte = np.asarray(gte); te_sub = np.asarray(te_sub)
        cdir = C.CKPT / task / "spectralite16" / "seed42"
        fp32, int8 = [], []
        for k in range(5):
            ck = cdir / f"fold{k}.pt"
            if not ck.exists():
                continue
            m = SpectraLite(n_classes=2, n_features=16).to("cpu")
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); m.eval()
            fp32.append(m)
            try:
                q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
                with torch.no_grad():
                    q(node_features=torch.zeros(1, 19, 16))
                int8.append(q)
            except Exception as exc:  # noqa: BLE001
                print(f"{task}: INT8 unavailable ({type(exc).__name__}: {str(exc)[:100]})", flush=True)
                int8 = None
                break
        if len(fp32) != 5:
            continue

        def ens(models):
            scores = []
            with torch.no_grad():
                for mm in models:
                    p = []
                    for i in range(0, len(Xte), 512):
                        xb = torch.from_numpy(np.ascontiguousarray(Xte[i:i + 512]))
                        p.append(torch.softmax(mm(node_features=xb), 1)[:, 1].numpy())
                    scores.append(np.concatenate(p))
            return C.subject_mean(np.mean(scores, 0), gte, te_sub)

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

        row = {"auc_fp32": auc(y, s32), "n_params": int(sum(p.numel() for p in fp32[0].parameters())),
               "size_fp32_bytes": size_bytes(fp32[0]),
               "latency_ms_window_fp32": lat(lambda: fp32[0](node_features=x1)),
               "latency_ms_subject24_fp32": lat(lambda: fp32[0](node_features=x24), 50),
               "n_external_subjects": int(len(te_sub))}
        if int8:
            row.update(auc_int8=auc(y, s8), max_abs_score_diff=float(np.max(np.abs(s32 - s8))),
                       size_int8_bytes=size_bytes(int8[0]),
                       latency_ms_window_int8=lat(lambda: int8[0](node_features=x1)),
                       latency_ms_subject24_int8=lat(lambda: int8[0](node_features=x24), 50))
        rep["tasks"][task] = {"spectralite16": row}
        print(f"{task:15s} SpectraLite AUC fp32 {row['auc_fp32']:.4f} | "
              f"INT8 {row.get('auc_int8', float('nan')):.4f} | size {row['size_fp32_bytes']/1024:.0f} KiB | "
              f"subject24 {row['latency_ms_subject24_fp32']:.2f} ms", flush=True)
    (CAN / "edge.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
