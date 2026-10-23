"""Age-conditioning wired into the 75k spectral model, with the anchored operating point + DCA.

204 showed age-conditioned normative deviation + control-anchoring fixes the transfer operating point
for a logistic on concept features. This wires the same idea into the deployable spectral-node model
(205): replace the model's per-channel spectral node features (12/ch) by their signed age-normative
deviation z = (x - mu(age)) / sigma(age) -- reference fit on TRAINING CONTROLS at the subject-median
level, applied per window -- and optionally re-anchored on the TARGET site's controls (fit_anchor).

Three arms, one architecture (SpectralOnly with a node-feature override), same folds, seed 42:
    raw            node features from the signal (= the 205 model)
    agedev         node features replaced by source-referenced age deviations
    agedev_anchor  + control-anchoring on ds004504 controls (~20 volunteers, no patient labels)

Reported per contrast: external AUC; operating point at a threshold fixed to 80% specificity on
SOURCE OOF controls (target spec/sens); and decision-curve net benefit re-weighted to 5/10/20%
service prevalence at that operating point. The claim under test: does age-conditioning recover the
deployable model's transfer sensitivity (as it did for the logistic), giving the full system
= 75k spectral model + montage-robust + transport-fixed operating point.

    python experiments/207_agecond_spectral.py --tasks CN_vs_dementia,CN_vs_AD,CN_vs_impaired
Out: outputs/results/audit/agecond_spectral.json
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "outputs" / "results" / "audit"
_FALLBACK = Path("C:/Users/user/Downloads/Alzfiles")
SEED, EPOCHS, BATCH, LR = 42, 25, 128, 3e-3
FLOOR = 1e-9
_LOG_ABS_TO_SD = (np.euler_gamma + np.log(2)) / 2
CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD", "CN": "CN", "C": "CN",
         "CONTROL": "CN", "MCI": "MCI"}
TASKS = {"CN_vs_dementia": (("AD", "FTD"), ("CN",), "ds004504_raw"),
         "CN_vs_AD": (("AD",), ("CN",), "ds004504_raw"),
         "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",), "ds004504_raw")}
PREV = [0.05, 0.10, 0.20]


def _resolve(rel):
    for base in (ROOT, _FALLBACK):
        if (base / rel).exists():
            return base / rel
    return ROOT / rel


def _rw(rel):
    for base in (ROOT / "outputs" / "harmonised_windows", _FALLBACK / "harmonised_windows"):
        if (base / rel).exists():
            return base / rel
    raise SystemExit("windows not found")


CAU_ANN = _resolve("datasets/caueeg-dataset/annotation.json")
ON_TSV = _resolve("datasets/openneuro_ds004504_raw/participants.tsv")


def _design(age):
    age = np.asarray(age, float)
    return np.column_stack([np.ones_like(age), age, age ** 2])


def norm_fit(X, age, min_n=30):
    X = np.asarray(X, float); age = np.asarray(age, float)
    nf = X.shape[1]; beta = np.full((nf, 3), np.nan); gamma = np.full((nf, 2), np.nan)
    usable = np.zeros(nf, bool)
    for j in range(nf):
        ok = np.isfinite(X[:, j]) & np.isfinite(age)
        if ok.sum() < min_n:
            continue
        A = _design(age[ok]); b, *_ = np.linalg.lstsq(A, X[ok, j], rcond=None)
        resid = X[ok, j] - A @ b
        L = np.column_stack([np.ones(ok.sum()), age[ok]])
        g, *_ = np.linalg.lstsq(L, np.log(np.abs(resid) + FLOOR), rcond=None)
        beta[j], gamma[j], usable[j] = b, g, True
    return {"beta": beta, "gamma": gamma, "usable": usable}


def _moments(m, age):
    A = _design(age); L = np.column_stack([np.ones(len(A)), np.asarray(age, float)])
    return A @ m["beta"].T, np.maximum(np.exp(L @ m["gamma"].T + _LOG_ABS_TO_SD), FLOOR)


def norm_dev(m, X, age, anchor=None):
    mu, sigma = _moments(m, age); z = (np.asarray(X, float) - mu) / sigma
    if anchor is not None:
        z = (z - anchor["shift"]) / anchor["scale"]
    z[:, ~m["usable"]] = 0.0
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def fit_anchor(m, Xc, agec):
    z = norm_dev(m, Xc, agec); shift = np.median(z, 0)
    mad = np.median(np.abs(z - shift), 0) * 1.4826
    return {"shift": shift, "scale": np.where(mad > FLOOR, mad, 1.0)}


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(s, method="average"); n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def net_benefit(y, prob, prev, thresholds=None):
    y = np.asarray(y); prob = np.asarray(prob, float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.51, 0.02), 3)
    obs = float((y == 1).mean()); wp = prev / obs if obs > 0 else 0; wn = (1 - prev) / (1 - obs)
    ne = wp * (y == 1).sum() + wn * (y == 0).sum()
    best = -1
    for pt in thresholds:
        fl = prob >= pt
        tp = float(((y == 1) & fl).sum()) * wp; fp = float(((y == 0) & fl).sum()) * wn
        nb = tp / ne - (fp / ne) * (pt / (1 - pt)); nb_all = prev - (1 - prev) * (pt / (1 - pt))
        best = max(best, nb - max(nb_all, 0.0))
    return round(best * 1000, 1)


def load_ages():
    age = {}
    ann = {r["serial"]: r for r in json.loads(CAU_ANN.read_text())["data"]}
    for s, r in ann.items():
        if r.get("age") is not None:
            age[("CAUEEG", str(s).zfill(5))] = float(r["age"])
    if ON_TSV.exists():
        for r in csv.DictReader(open(ON_TSV, newline="", encoding="utf-8"), delimiter="\t"):
            try:
                age[("ds004504_raw", r["participant_id"].strip())] = float(r["Age"])
            except (TypeError, ValueError, KeyError):
                pass
    return age


def keyf(s, c):
    s = str(s)
    if c == "CAUEEG":
        mm = re.search(r"(\d+)$", s); return ("CAUEEG", mm.group(1).zfill(5)) if mm else None
    if c == "ds004504_raw":
        mm = re.search(r"(sub-\d+)", s); return ("ds004504_raw", mm.group(1)) if mm else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--tasks", default="CN_vs_dementia,CN_vs_AD,CN_vs_impaired")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = [t for t in args.tasks.split(",") if t]

    import torch
    import torch.nn as nn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                                                        # noqa: BLE001
        pass
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    from src.models.v5.montage import MontageEmbedding
    from src.models.v5.neuro_chrono_graph_v5 import AttentionPool

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NF = SpectralNodeFeatures.N_FEATURES

    class SpectralOnly2(nn.Module):
        def __init__(self, n_classes=2, hidden=128, sfreq=256.0, dropout=0.3):
            super().__init__()
            self.spectral = SpectralNodeFeatures(sfreq=sfreq)
            self.proj = nn.Sequential(nn.Linear(NF, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, hidden))
            self.montage = MontageEmbedding(hidden, dropout=dropout)
            self.pool = AttentionPool(hidden, hidden // 2, dropout)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x=None, node_features=None):
            feats = node_features if node_features is not None else self.spectral(x)
            node = self.proj(feats); m, mm = self.montage(None, None)
            node = node + m.unsqueeze(0); pooled, _ = self.pool(node, mm)
            return self.head(pooled)

    Xw = np.load(_rw(f"{args.tag}/windows.npy"), mmap_mode="r")
    meta = np.load(ROOT / "outputs" / "harmonised_features" / args.tag / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    win_cls = np.array([CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    amap = load_ages()
    win_age = np.array([amap.get(keyf(g, c), np.nan) for g, c in zip(groups, win_coh)], float)

    # --- extract per-window spectral node features once: F [N, 19, NF] ---
    print("extracting spectral node features for all windows...", flush=True)
    extractor = SpectralNodeFeatures(sfreq=256.0).to(dev).eval()
    F = np.zeros((len(Xw), 19, NF), np.float32)
    with torch.no_grad():
        for i in range(0, len(Xw), 512):
            xb = torch.from_numpy(np.ascontiguousarray(Xw[i:i + 512])).float().to(dev)
            F[i:i + 512] = extractor(xb).cpu().numpy()
    Fflat = F.reshape(len(F), -1)                              # [N, 19*NF]
    print("done. F:", F.shape, flush=True)

    def _predict(model, Xin, isnf):
        p = []
        with torch.no_grad():
            for i in range(0, len(Xin), 256):
                kw = {"node_features": Xin[i:i + 256].to(dev)} if isnf else {"x": Xin[i:i + 256].to(dev)}
                p.append(torch.softmax(model(**kw), 1)[:, 1].cpu().numpy())
        return np.concatenate(p) if len(Xin) else np.array([])

    def train_eval(get_input, y, g, tr_sub, te_sub, is_nodefeat):
        """Returns (te_subs, te_scores, tr_subs, tr_oof_scores). tr OOF sets the source threshold."""
        sub_y = np.array([y[g == s][0] for s in tr_sub])
        te = np.isin(g, te_sub)
        Xte = torch.from_numpy(np.ascontiguousarray(get_input(te))).float()
        g_all = g
        test_preds = []
        oof_win = np.full(len(g), np.nan)                     # per-window OOF over train windows
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr_sub, sub_y):
            fit = np.isin(g, tr_sub[a]); held = np.isin(g, tr_sub[b])
            Xtr = torch.from_numpy(np.ascontiguousarray(get_input(fit))).float()
            ytr = torch.from_numpy(y[fit].astype(np.int64))
            torch.manual_seed(SEED); model = SpectralOnly2().to(dev)
            cls, cnt = np.unique(y[fit], return_counts=True); wt = np.zeros(2, np.float32)
            wt[cls] = len(ytr) / (len(cls) * cnt)
            crit = nn.CrossEntropyLoss(weight=torch.tensor(wt, device=dev))
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                total_steps=EPOCHS * max(1, len(ytr) // BATCH + 1))
            dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
            model.train()
            for _ in range(EPOCHS):
                for xb, yb in dl:
                    opt.zero_grad()
                    kw = {"node_features": xb.to(dev)} if is_nodefeat else {"x": xb.to(dev)}
                    loss = crit(model(**kw), yb.to(dev)); loss.backward(); opt.step()
                    try:
                        sched.step()
                    except Exception:                                        # noqa: BLE001
                        pass
            model.eval()
            Xheld = torch.from_numpy(np.ascontiguousarray(get_input(held))).float()
            oof_win[held] = _predict(model, Xheld, is_nodefeat)
            test_preds.append(_predict(model, Xte, is_nodefeat))
            del model; torch.cuda.empty_cache()
        win = np.mean(test_preds, 0); g_te = g[te]; te_subs = np.unique(g_te)
        te_score = np.array([win[g_te == s].mean() for s in te_subs])
        tr_score = np.array([np.nanmean(oof_win[g == s]) for s in tr_sub])
        return te_subs, te_score, tr_sub, tr_score

    report = {"tag": args.tag, "seed": SEED, "arms": ["raw", "agedev", "agedev_anchor"],
              "prevalences": PREV, "tasks": {}}
    for task in want:
        pos, neg, site = TASKS[task]
        keep = np.isin(win_cls, pos + neg) & np.isfinite(win_age)
        y = np.isin(win_cls, pos).astype(int)
        tr_sub = np.unique(groups[keep & (win_coh == "CAUEEG")])
        te_sub = np.unique(groups[keep & (win_coh == site)])
        if len(tr_sub) < 40 or len(te_sub) < 16:
            print(f"[{task}] skipped", flush=True); continue
        idx = np.flatnonzero(keep)
        gk = groups[idx]; yk = y[idx]
        yte_sub = np.array([yk[gk == s][0] for s in te_sub])
        # normative fit on TRAIN CONTROL subjects (subject-median features)
        subj = np.unique(gk); s_age = {}; s_med = {}; s_coh = {}; s_cls = {}
        for s in subj:
            m = gk == s
            s_med[s] = np.nanmedian(Fflat[idx][m], 0); s_age[s] = win_age[idx][m][0]
            s_coh[s] = win_coh[idx][m][0]; s_cls[s] = yk[m][0]
        ctrl_tr = [s for s in tr_sub if s_cls[s] == 0]
        Xc = np.stack([s_med[s] for s in ctrl_tr]); Ac = np.array([s_age[s] for s in ctrl_tr])
        nm = norm_fit(Xc, Ac)
        ctrl_te = [s for s in te_sub if s_cls[s] == 0]
        anchor = fit_anchor(nm, np.stack([s_med[s] for s in ctrl_te]),
                            np.array([s_age[s] for s in ctrl_te]))

        # per-window deviation inputs (source-ref and anchored), reshaped to [N,19,NF]
        wa = win_age[idx]
        Zsrc = norm_dev(nm, Fflat[idx], wa).reshape(-1, 19, NF).astype(np.float32)
        Zanc = norm_dev(nm, Fflat[idx], wa, anchor=anchor).reshape(-1, 19, NF).astype(np.float32)
        Xraw = Xw[idx]

        entry = {"eval_cohort": site, "n_eval": int(len(te_sub)), "n_eval_pos": int(yte_sub.sum()),
                 "arms": {}}
        tr_y = np.array([yk[gk == s][0] for s in tr_sub])
        for arm, inp, isnf in (("raw", Xraw, False), ("agedev", Zsrc, True), ("agedev_anchor", Zanc, True)):
            subs, sc, trs, tr_sc = train_eval(lambda mask, _inp=inp: _inp[mask],
                                              yk, gk, tr_sub, te_sub, isnf)
            order = {s: i for i, s in enumerate(subs)}
            s_al = np.array([sc[order[s]] for s in te_sub])
            a = auc(yte_sub, s_al)
            # threshold fixed at 80% specificity on SOURCE out-of-fold controls (the transport test)
            thr = float(np.quantile(tr_sc[tr_y == 0], 0.80))
            tspec = float(np.mean(s_al[yte_sub == 0] < thr))
            tsens = float(np.mean(s_al[yte_sub == 1] >= thr))
            nb = {f"{p:.2f}": net_benefit(yte_sub, s_al, p) for p in PREV}
            entry["arms"][arm] = {"auc_external": round(a, 4),
                                  "ext_spec_at80src": round(tspec, 3), "ext_sens_at80src": round(tsens, 3),
                                  "delta_spec_vs_0.80": round(tspec - 0.80, 3),
                                  "net_benefit_per_1000": nb}
            print(f"   [{task}] {arm:13s} AUC {a:.4f}  ext@80%src-spec: spec {tspec:.3f} "
                  f"(d{tspec-0.80:+.3f}) sens {tsens:.3f}  NB/1000 {nb}", flush=True)
        report["tasks"][task] = entry
        (OUT / "agecond_spectral.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {OUT / 'agecond_spectral.json'}", flush=True)


if __name__ == "__main__":
    main()
