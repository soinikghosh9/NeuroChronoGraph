"""SpectraLite and the additive component ladder between it and Graph-0.96M.

Until 2026-09-17 the compact spectral network was redefined inline in five experiment scripts
(205-210). SpectraLite is the model reported in the paper: a coordinate-aware electrode encoder with
gated attention pooling over 12 spectral features per electrode (75,459 parameters; `SpectraLite(n_features=12)`,
the same network as `Spectral75k`). The 16-feature variant (75,971 parameters, the constructor default kept
for older scripts) was chosen after external scoring and is exploratory. This module is the single definition
used by the canonical benchmark, release export, montage analysis, attribution analysis, and paper-facing
statistics.

The ladder isolates what each stream of Graph-0.96M contributes on top of the shared front-end:

    spectral            Spectral-75k: spectral nodes + montage embedding -> gated attention pool -> head
    spectral+raw        + the per-channel raw-waveform path (and node LayerNorm, as in v5)
    spectral+graph      + adaptive graph + two gated graph convolutions (graph-stream pool)
    spectral+raw+graph  both
    spectral+raw+graph+lobe   + lobe-coupling stream
    (full v5)           + temporal stream  -> use NeuroChronoGraphV5 itself

Multi-stream variants fuse exactly as v5 does: concatenate pooled stream vectors, LayerNorm, then
Linear(n*128 -> 128) -> LN -> GELU -> Dropout -> Linear(128 -> n_classes).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

# Running this module directly places ``src/models`` ahead of the standard library,
# where its local ``ssl`` package can shadow Python's ``ssl`` module during torch import.
# Keep direct self-tests usable while leaving package imports unchanged.
if __name__ == "__main__":
    _MODEL_DIR = str(Path(__file__).resolve().parent)
    if _MODEL_DIR in sys.path:
        sys.path.remove(_MODEL_DIR)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from src.models.v2.adaptive_graph import AdaptiveGraphLearning, GatedGraphConvolution
from src.models.v5.montage import MontageEmbedding, masked_adjacency
from src.models.v5.neuro_chrono_graph_v5 import AttentionPool, LobeCoupling
from src.models.v5.spectral_nodes import SpectralNodeFeatures

N_PARAMS_SPECTRAL75K = 75_459


class Spectral75k(nn.Module):
    """Compatibility implementation of the historical 12-feature network.

    New code should instantiate SpectraLite, which is the only model reported in the paper.
    """

    def __init__(self, n_classes: int = 2, hidden: int = 128, sfreq: float = 256.0,
                 dropout: float = 0.3, channel_names: Optional[Sequence[str]] = None,
                 n_features: Optional[int] = None):
        super().__init__()
        # The compatibility class keeps its historical default. The released class below defaults to
        # the canonical 16-feature input.
        nf = int(n_features) if n_features is not None else SpectralNodeFeatures.N_FEATURES
        self.spectral = SpectralNodeFeatures(sfreq=sfreq)
        self.proj = nn.Sequential(nn.Linear(nf, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.montage = MontageEmbedding(hidden, channel_names=channel_names, dropout=dropout)
        self.pool = AttentionPool(hidden, hidden // 2, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def node_embed(self, x=None, node_features=None, channel_names=None):
        feats = node_features if node_features is not None else self.spectral(x)
        if channel_names is None:
            m, mm = self.montage(None, None)
        else:
            m, mm = self.montage.for_montage(channel_names, device=feats.device)
        return self.proj(feats) + m.unsqueeze(0), mm, feats

    def forward(self, x=None, node_features=None, channel_names=None, return_attention=False):
        node, mm, _ = self.node_embed(x, node_features, channel_names)
        pooled, w = self.pool(node, mm)
        logits = self.head(pooled)
        return (logits, w) if return_attention else logits


class SpectraLite(Spectral75k):
    """SpectraLite with ``n_features`` node descriptors per electrode.

    ``n_features=12`` is the reported model: it accepts raw windows ``x`` of shape ``(batch, electrodes,
    samples)`` at 256 Hz, from which the 12 spectral features are computed, or precomputed ``node_features``
    of shape ``(batch, electrodes, 12)``. ``n_features=16`` (the default, for older scripts) is the
    exploratory variant and needs the 16-feature node cache passed as ``node_features``.
    """

    def __init__(self, *args, n_features: int = 16, **kwargs):
        super().__init__(*args, n_features=n_features, **kwargs)

    def forward(self, x=None, node_features=None, channel_names=None, return_attention=False):
        if node_features is None and x is not None and self.proj[0].in_features == 16:
            raise ValueError("SpectraLite expects the released 16-feature node cache; pass node_features.")
        return super().forward(x=x, node_features=node_features, channel_names=channel_names,
                               return_attention=return_attention)


class LadderNet(nn.Module):
    """Spectral-75k plus any subset of {raw path, graph stream, lobe stream}."""

    def __init__(self, raw: bool = False, graph: bool = False, lobe: bool = False,
                 n_classes: int = 2, hidden: int = 128, sfreq: float = 256.0,
                 dropout: float = 0.3, n_heads: int = 4, n_graph_layers: int = 2):
        super().__init__()
        if lobe and not graph:
            raise ValueError("the lobe stream reads graph-updated nodes; enable graph")
        self.raw, self.graph, self.lobe = raw, graph, lobe
        nf = SpectralNodeFeatures.N_FEATURES
        self.spectral = SpectralNodeFeatures(sfreq=sfreq)
        self.spectral_proj = nn.Sequential(nn.Linear(nf, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.montage = MontageEmbedding(hidden, dropout=dropout)
        if raw:
            self.raw_nodes = nn.Sequential(
                nn.Conv1d(1, hidden // 4, kernel_size=25, stride=8, padding=12), nn.GELU(),
                nn.Conv1d(hidden // 4, hidden // 2, kernel_size=9, stride=8, padding=4), nn.GELU(),
                nn.AdaptiveAvgPool1d(1))
            self.raw_proj = nn.Linear(hidden // 2, hidden)
            self.node_norm = nn.LayerNorm(hidden)
        if graph:
            self.adaptive_graph = AdaptiveGraphLearning(hidden, hidden // 2, max(n_heads // 2, 1),
                                                        dropout, use_prior=True)
            self.gat_layers = nn.ModuleList([GatedGraphConvolution(hidden, hidden, dropout)
                                             for _ in range(n_graph_layers)])
        self.node_pool = AttentionPool(hidden, hidden // 2, dropout)
        if lobe:
            self.modular = LobeCoupling(hidden, max(n_heads // 2, 1), dropout)
        n_streams = 1 + int(lobe)
        if n_streams == 1:
            self.embed_norm = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                            nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(hidden, n_classes))
        else:
            self.embed_norm = nn.LayerNorm(hidden * n_streams)
            self.classifier = nn.Sequential(nn.Linear(hidden * n_streams, hidden),
                                            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(hidden, n_classes))

    def forward(self, x):
        feats = self.spectral(x)
        node = self.spectral_proj(feats)
        m, mm = self.montage(None, None)
        node = node + m.unsqueeze(0)
        if self.raw:
            B, C, T = x.shape
            r = self.raw_nodes(x.reshape(B * C, 1, T)).reshape(B, C, -1)
            node = self.node_norm(node + self.raw_proj(r))
        h = node
        if self.graph:
            adj, _ = self.adaptive_graph(node, None)
            adj = masked_adjacency(adj, mm)
            for gat in self.gat_layers:
                h = gat(h, adj)
        pooled, _ = self.node_pool(h, mm)
        streams = [pooled]
        if self.lobe:
            lob, _ = self.modular(h, self.montage.default_coords, mm)
            streams.append(lob)
        return self.classifier(self.embed_norm(torch.cat(streams, dim=-1)))


LADDER = {
    "spectral": dict(),
    "spectral+raw": dict(raw=True),
    "spectral+graph": dict(graph=True),
    "spectral+raw+graph": dict(raw=True, graph=True),
    "spectral+raw+graph+lobe": dict(raw=True, graph=True, lobe=True),
}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conformal_threshold(control_scores, target_specificity: float = 0.80) -> float:
    """Split-conformal operating point for deployment at a new clinic.

    Given the model's subject scores for ``k`` of a receiving clinic's own healthy controls (no patient
    labels needed), return the threshold whose expected specificity on further exchangeable controls is
    at least ``target_specificity`` for any ``k``. A subject is referred when its score exceeds the
    threshold. This is the rule recommended in the paper (Section: transporting the operating point):
    the ceil(s(k+1))-th smallest control score.
    """
    import numpy as np

    s = np.sort(np.asarray(control_scores, dtype=float))
    k = s.size
    if k == 0:
        raise ValueError("need at least one control score")
    idx = min(int(np.ceil(target_specificity * (k + 1))), k)  # 1-based rank, clamped
    return float(s[idx - 1])


if __name__ == "__main__":
    m = Spectral75k()
    assert n_params(m) == N_PARAMS_SPECTRAL75K, n_params(m)
    lite = SpectraLite()
    assert n_params(lite) == 75_971, n_params(lite)
    x = torch.randn(3, 19, 1024)
    assert m(x).shape == (3, 2)
    for k, kw in LADDER.items():
        net = LadderNet(**kw)
        print(f"{k:26s} {n_params(net):>9,d}  out {tuple(net(x).shape)}")
    from src.models.v5.neuro_chrono_graph_v5 import NeuroChronoGraphV5
    print(f"{'full v5':26s} {n_params(NeuroChronoGraphV5(n_classes=2)):>9,d}")
    print("Spectral75k OK:", n_params(m))
