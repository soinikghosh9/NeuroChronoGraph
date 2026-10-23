"""
NeuroChronoGraph v5 -- rebuilt for cross-site clinical screening.

What changed relative to v2/v3, and why. Each item traces to a measured result in this
project rather than to architecture search.

1.  **Spectral features are the backbone, not an accessory.** Every node carries 12 named,
    gain-invariant physiological quantities (:class:`SpectralNodeFeatures`), including
    individual alpha peak frequency and the aperiodic 1/f exponent, neither of which v2
    could represent. Spectral features are what transfers across sites here: LODO AUC 0.673
    versus 0.589 for the learned representation, and 0.734 versus 0.475 on the CAUEEG fold.

2.  **Electrodes are identified by position, not index** (:class:`MontageEmbedding`). v2 used
    ``nn.Parameter(torch.randn(n_channels, hidden))``, locking it to one 19-channel ordering.
    v5 scores an arbitrary montage and masks absent electrodes out of the graph.

3.  **The connectivity prior is actually supplied.** ``AdaptiveGraphLearning`` has always had
    a ``fc_prior`` branch; no training run in this project ever passed one, so the reported
    "wPLI prior is neutral" ablation measured nothing. Here band-wise wPLI enters as a
    learned-weighted prior, so the graph is built from measured phase coupling rather than
    from similarity of time-averaged band power.

4.  **The two redundant streams are separated.** v2 concatenated ``modular_out`` with
    ``graph_pooled = h.mean(1)`` -- two reductions of the same tensor, which is why the
    modular transformer ablated at no cost. v5 pools the graph by *attention* instead, so
    the streams carry different information.

5.  **Capacity is reallocated.** v2 spent 55.8% of its parameters on a temporal transformer
    that operates at a 12.5 Hz effective feature rate and is therefore blind to peak
    frequency, and 1.2% on the graph module that gives the model its name.

Subject-level training and the split-half consistency objective live in
:class:`NCGv5MIL` and :mod:`src.models.v5.losses`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.v2.adaptive_graph import AdaptiveGraphLearning, GatedGraphConvolution
from src.models.v2.neuro_chrono_graph_v2 import BandSpecificEncoder
from src.models.v5.montage import (LOBES, MontageEmbedding, lobe_assignment,
                                   masked_adjacency)
from src.models.v5.spectral_nodes import SpectralNodeFeatures


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al., 2018), used over nodes and over windows."""

    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``[B, N, D]`` -> pooled ``[B, D]`` and weights ``[B, N]``."""
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        if mask is not None:
            m = mask if mask.dim() == 2 else mask.unsqueeze(0).expand(a.shape[0], -1)
            a = a.masked_fill(~m.bool(), float("-inf"))
        w = torch.softmax(a, dim=-1)
        return (self.drop(x) * w.unsqueeze(-1)).sum(dim=1), w


class SlimTemporalEncoder(nn.Module):
    """Small temporal path. Deliberately small: it cannot resolve peak frequency, so it
    should not hold the majority of the parameters as it did in v2.

    The stem is applied **per channel with shared weights**, then averaged across channels.
    A ``Conv1d(n_channels, ...)`` stem would treat the channel axis as an input feature
    dimension and make the whole model order-dependent, silently contradicting the
    montage-agnostic design -- and it would break outright on a montage with a different
    channel count.
    """

    def __init__(self, n_channels: int = 19, hidden_dim: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(hidden_dim // 2), nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = AttentionPool(hidden_dim, hidden_dim // 2, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T = x.shape
        h = self.stem(x.reshape(B * C, 1, T))            # [B*C, H, T']
        h = h.reshape(B, C, h.shape[1], h.shape[2])
        if mask is not None:
            m = mask.view(1, C, 1, 1).to(h.dtype)
            h = (h * m).sum(1) / (m.sum(1) + 1e-8)       # masked mean over channels
        else:
            h = h.mean(dim=1)
        h = self.encoder(h.transpose(1, 2))              # [B, T', H]
        pooled, _ = self.pool(h)
        return pooled


class LobeCoupling(nn.Module):
    """Pool nodes into anatomical lobes by coordinate, then attend across lobes.

    Replaces ``ModularBrainTransformer``, which identifies lobes by fixed channel index
    (``'frontal': [0,1,2,3,10,11,16]``). Those indices are right for this repository's
    canonical order and wrong for any other montage, and they make the model
    order-dependent. Assigning lobes from coordinates keeps the clinically interesting
    output -- an interpretable lobe-by-lobe coupling matrix, which carries the
    frontal-versus-posterior axis -- while remaining permutation-equivariant.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_lobes = len(LOBES)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPool(hidden_dim, hidden_dim // 2, dropout)

    def forward(self, h: torch.Tensor, coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        lobes = lobe_assignment(coords)                        # [C]
        onehot = F.one_hot(lobes, self.n_lobes).to(h.dtype)    # [C, L]
        if mask is not None:
            onehot = onehot * mask.unsqueeze(-1).to(h.dtype)
        counts = onehot.sum(dim=0).clamp(min=1.0)              # [L]
        lobe_h = torch.einsum("bch,cl->blh", h, onehot) / counts.view(1, -1, 1)

        att, coupling = self.attn(lobe_h, lobe_h, lobe_h, need_weights=True)
        lobe_h = self.norm(lobe_h + att)
        pooled, _ = self.pool(lobe_h)
        return pooled, coupling


class BandWeightedPrior(nn.Module):
    """Collapse band-wise wPLI ``[B, n_bands, N, N]`` into one prior with learned weights.

    The weights are the model's statement about *which band's* connectivity matters for the
    decision -- a directly reportable quantity, unlike an opaque learned adjacency.

    Each edge additionally passes through a gate conditioned on its **full band profile**,
    which is what using connectivity as an *edge feature* means rather than as a scalar
    prior. (v2's ``AdaptiveGraphLearning`` constructs an ``edge_transform`` for this and
    never calls it -- a fourth dead component alongside ``band_features``, the wPLI prior
    and ``CrossBandAttention``. It is left untouched there so v2's published behaviour does
    not change.)
    """

    def __init__(self, n_bands: int = 5, hidden: int = 16, gate_edges: bool = True):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_bands))
        self.edge_gate = nn.Sequential(
            nn.Linear(n_bands, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid(),
        ) if gate_edges else None

    def forward(self, wpli: torch.Tensor) -> torch.Tensor:
        if wpli.dim() == 3:                        # already collapsed
            return wpli
        w = torch.softmax(self.logits, dim=0).view(1, -1, 1, 1)
        prior = (wpli * w).sum(dim=1)              # [B, N, N]
        if self.edge_gate is not None:
            gate = self.edge_gate(wpli.permute(0, 2, 3, 1)).squeeze(-1)
            prior = prior * gate
        return prior

    def band_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.detach(), dim=0)


class NeuroChronoGraphV5(nn.Module):
    """Window-level backbone. Wrap in :class:`NCGv5MIL` for subject-level training."""

    def __init__(self,
                 n_channels: int = 19,
                 n_classes: int = 3,
                 hidden_dim: int = 128,
                 sfreq: float = 256.0,
                 n_graph_layers: int = 2,
                 n_temporal_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.3,
                 channel_names: Optional[Sequence[str]] = None,
                 use_bands: bool = True,
                 use_modular: bool = True,
                 use_raw_nodes: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.use_bands = use_bands
        self.use_modular = use_modular
        self.use_raw_nodes = use_raw_nodes

        # --- node features -----------------------------------------------------------
        self.spectral = SpectralNodeFeatures(sfreq=sfreq)
        self.spectral_proj = nn.Sequential(
            nn.Linear(SpectralNodeFeatures.N_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.montage = MontageEmbedding(hidden_dim, channel_names=channel_names,
                                        dropout=dropout)
        if use_bands:
            self.band_encoder = BandSpecificEncoder(
                n_channels=n_channels, hidden_dim=hidden_dim, per_node=True, dropout=dropout)
        if use_raw_nodes:
            # A small learned path so the model is not purely hand-crafted; kept modest
            # because the raw front-end is what failed to transfer.
            self.raw_nodes = nn.Sequential(
                nn.Conv1d(1, hidden_dim // 4, kernel_size=25, stride=8, padding=12),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=9, stride=8, padding=4),
                nn.GELU(), nn.AdaptiveAvgPool1d(1),
            )
            self.raw_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # --- graph -------------------------------------------------------------------
        self.prior = BandWeightedPrior()
        self.adaptive_graph = AdaptiveGraphLearning(
            hidden_dim, hidden_dim // 2, max(n_heads // 2, 1), dropout, use_prior=True)
        self.gat_layers = nn.ModuleList(
            [GatedGraphConvolution(hidden_dim, hidden_dim, dropout) for _ in range(n_graph_layers)])
        self.graph_pool = AttentionPool(hidden_dim, hidden_dim // 2, dropout)

        if use_modular:
            self.modular = LobeCoupling(hidden_dim, max(n_heads // 2, 1), dropout)

        # --- temporal ----------------------------------------------------------------
        self.temporal = SlimTemporalEncoder(
            n_channels, hidden_dim, n_temporal_layers, n_heads, dropout)

        # --- heads -------------------------------------------------------------------
        n_streams = 2 + int(use_modular)
        self.fused_dim = hidden_dim * n_streams
        self.embed_norm = nn.LayerNorm(self.fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes),
        )

    # ---------------------------------------------------------------- node assembly
    def _nodes(self, x, band_features, coords, mask):
        feats = self.spectral(x)                                # [B, C, 12]
        node = self.spectral_proj(feats)

        m_emb, m_mask = self.montage(coords, mask)              # [C, H], [C]
        node = node + m_emb.unsqueeze(0)

        if self.use_bands and band_features:
            band = self.band_encoder(band_features)             # [B, Bd, C, H]
            node = node + band.mean(dim=1)

        if self.use_raw_nodes:
            B, C, T = x.shape
            r = self.raw_nodes(x.reshape(B * C, 1, T)).reshape(B, C, -1)
            node = node + self.raw_proj(r)

        return self.node_norm(node), m_mask, feats

    # ---------------------------------------------------------------- forward
    def forward(self,
                x: torch.Tensor,
                band_features: Optional[Dict[str, torch.Tensor]] = None,
                wpli: Optional[torch.Tensor] = None,
                coords: Optional[torch.Tensor] = None,
                channel_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        node, node_mask, spec_feats = self._nodes(x, band_features, coords, channel_mask)
        outputs["spectral_features"] = spec_feats

        fc_prior = self.prior(wpli) if wpli is not None else None
        if fc_prior is not None:
            outputs["band_weights"] = self.prior.band_weights()
        adj, attn = self.adaptive_graph(node, fc_prior)
        adj = masked_adjacency(adj, node_mask)
        outputs["learned_adjacency"] = adj

        h = node
        for gat in self.gat_layers:
            h = gat(h, adj)

        graph_vec, node_w = self.graph_pool(h, node_mask)
        outputs["node_attention"] = node_w

        streams = [graph_vec]
        if self.use_modular:
            node_coords = coords if coords is not None else self.montage.default_coords
            modular_out, coupling = self.modular(h, node_coords, node_mask)
            outputs["lobe_coupling"] = coupling
            streams.append(modular_out)
        streams.append(self.temporal(x, node_mask))

        embedding = self.embed_norm(torch.cat(streams, dim=-1))
        logits = self.classifier(embedding)
        outputs["logits"] = logits
        outputs["probs"] = F.softmax(logits, dim=-1)
        if return_embeddings:
            outputs["embedding"] = embedding
        return outputs


class NCGv5MIL(nn.Module):
    """Subject-level wrapper: a bag of windows in, one posterior out.

    v2 classified windows independently and soft-voted, so nothing in the network was ever
    trained on the quantity that is reported and acted on. Experiment L1 showed no
    aggregation rule recovers this, because the problem is the objective, not the read-out.

    Attention over windows additionally gives a route to down-weight artefactual segments,
    which matters because artefact burden alone separates classes at AUC 0.788 in GENEEG.
    """

    def __init__(self, backbone: NeuroChronoGraphV5, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.pool = AttentionPool(backbone.fused_dim, backbone.hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(backbone.fused_dim, backbone.hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(backbone.hidden_dim, n_classes),
        )

    def forward(self, bag: torch.Tensor, band_features=None, wpli=None,
                coords=None, channel_mask=None) -> Dict[str, torch.Tensor]:
        """``bag``: ``[K, C, T]`` windows for ONE subject."""
        out = self.backbone(bag, band_features=band_features, wpli=wpli,
                            coords=coords, channel_mask=channel_mask,
                            return_embeddings=True)
        pooled, w = self.pool(out["embedding"].unsqueeze(0))
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "window_attention": w.squeeze(0),
            "window_probs": out["probs"],
        }
