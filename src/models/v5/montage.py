"""
Montage-agnostic electrode representation for NeuroChronoGraph v5.

The v2/v3 encoder identifies an electrode by its **index**::

    self.channel_embeddings = nn.Parameter(torch.randn(n_channels, hidden_dim))

which locks the model to one 19-channel ordering. A site recording a different montage, a
different channel order, or 18 of the 19 electrodes cannot be scored at all without
retraining -- a hard blocker for a screening tool meant to run on data from any clinic.

Here an electrode is identified by **where it sits on the head**. A node's embedding is a
function of its 3-D coordinate, so:

* a montage the model never saw at training maps in, as long as its coordinates are known;
* a missing electrode is masked out of the graph rather than zero-filled, which otherwise
  presents the network with a dead node it has no way to interpret;
* electrode order carries no meaning, so loaders that order channels differently agree.

Coordinates come from ``src.config.config.CHANNEL_COORDS_10_20`` and are on a unit sphere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # keep this module importable in isolation (unit tests, notebooks)
    from src.config.config import CHANNEL_COORDS_10_20, CHANNEL_NAMES
except Exception:  # pragma: no cover
    CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    CHANNEL_COORDS_10_20 = {}

# Common aliases, so a montage using the modern names still resolves.
ALIASES: Dict[str, str] = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
}


def resolve_coords(names: Sequence[str],
                   coord_table: Optional[Dict[str, Tuple[float, float, float]]] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map electrode names to coordinates and a presence mask.

    Returns ``coords [C, 3]`` and ``mask [C]`` (True where the electrode is usable).
    Unknown names get zero coordinates and ``mask=False`` rather than raising, so a montage
    containing an electrode outside the 10-20 table degrades to "that node is absent".
    """
    table = coord_table if coord_table is not None else CHANNEL_COORDS_10_20
    coords, mask = [], []
    for raw in names:
        key = raw.strip()
        key = ALIASES.get(key, key)
        if key not in table:
            # try a case-insensitive match before giving up
            hit = next((k for k in table if k.lower() == key.lower()), None)
            key = hit if hit is not None else key
        if key in table:
            coords.append(tuple(table[key]))
            mask.append(True)
        else:
            coords.append((0.0, 0.0, 0.0))
            mask.append(False)
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


class MontageEmbedding(nn.Module):
    """Turn 3-D electrode coordinates into node embeddings ``[C, hidden_dim]``.

    A plain linear map of three numbers has very little capacity to separate 19 nearby
    points, so the coordinates are first lifted with sinusoidal features (the standard
    positional-encoding trick) before the MLP.
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 n_frequencies: int = 8,
                 channel_names: Optional[Sequence[str]] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_frequencies = n_frequencies

        names = list(channel_names) if channel_names is not None else list(CHANNEL_NAMES)
        coords, mask = resolve_coords(names)
        self.default_names = names
        self.register_buffer("default_coords", coords, persistent=False)
        self.register_buffer("default_mask", mask, persistent=False)

        # 3 coords x 2 (sin, cos) x n_frequencies, plus the raw coordinates
        in_dim = 3 + 3 * 2 * n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            "_freq_bands", 2.0 ** torch.arange(n_frequencies, dtype=torch.float32),
            persistent=False,
        )

    def lift(self, coords: torch.Tensor) -> torch.Tensor:
        """``[..., 3]`` -> ``[..., 3 + 6*n_frequencies]`` sinusoidal features."""
        scaled = coords.unsqueeze(-1) * self._freq_bands   # [..., 3, n_freq]
        feats = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # [..., 3, 2*n_freq]
        return torch.cat([coords, feats.flatten(-2)], dim=-1)

    def forward(self,
                coords: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(embeddings [C, hidden_dim], mask [C] bool)``."""
        if coords is None:
            coords, mask = self.default_coords, self.default_mask
        if mask is None:
            mask = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        emb = self.mlp(self.lift(coords.to(self._freq_bands.dtype)))
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)      # absent electrodes contribute nothing
        return emb, mask

    def for_montage(self, names: Sequence[str], device=None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: embed an arbitrary named montage."""
        coords, mask = resolve_coords(names)
        if device is not None:
            coords, mask = coords.to(device), mask.to(device)
        return self.forward(coords, mask)


LOBES: Tuple[str, ...] = ("frontal", "central", "temporal", "parietal", "occipital")


def lobe_assignment(coords: torch.Tensor) -> torch.Tensor:
    """Assign each electrode to an anatomical lobe from its coordinate. Returns ``[C]``.

    Coordinate-driven rather than index-driven. ``ModularBrainTransformer`` hard-codes
    channel indices (``'frontal': [0,1,2,3,10,11,16]``), which is correct for this repo's
    canonical order but silently wrong for any other montage -- and it makes the model
    order-dependent, contradicting the rest of this module.

    Axes: ``x`` left-right, ``y`` posterior-anterior, ``z`` inferior-superior.
    Checked against all 19 electrodes of the standard 10-20 table.
    """
    x, y = coords[..., 0], coords[..., 1]
    out = torch.full(coords.shape[:-1], LOBES.index("central"),
                     dtype=torch.long, device=coords.device)
    # Order matters: most specific first.
    is_parietal = y <= -0.4
    is_frontal = y >= 0.5
    is_temporal = (x.abs() >= 0.78) & (y <= 0.2)     # T3/T4/T5/T6 but not F7/F8
    is_occipital = y <= -0.85

    out = torch.where(is_parietal, torch.full_like(out, LOBES.index("parietal")), out)
    out = torch.where(is_frontal, torch.full_like(out, LOBES.index("frontal")), out)
    out = torch.where(is_temporal, torch.full_like(out, LOBES.index("temporal")), out)
    out = torch.where(is_occipital, torch.full_like(out, LOBES.index("occipital")), out)
    return out


def masked_adjacency(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out rows/columns of absent electrodes and renormalise.

    Applied after the adjacency is formed so an absent node neither sends nor receives
    messages, instead of silently participating as a zero-valued node.
    """
    if mask is None:
        return adj
    m = mask.to(adj.dtype)
    while m.dim() < adj.dim() - 1:
        m = m.unsqueeze(0)
    adj = adj * m.unsqueeze(-1) * m.unsqueeze(-2)
    return adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
