"""Feature-stream fusion module for NeuroChronoGraph.

Provides explicit injection points for hand-crafted biomarker streams
(spectral, connectivity, complexity, microstate) into the fused
representation. Each stream is gated independently so that individual
streams can be disabled at inference time for ablation studies.

Stream dimensions (per 4 s window, 19-channel montage):
  - spectral     : 19*5 + 4*5 + 3 = 118  (channel band power + regional + iAPF/TAR/SE)
  - connectivity : 5*171           = 855  (upper-triangle wPLI per band; 19*18/2=171)
  - complexity   : 19*4            =  76  (LZC, DFA, PE, sample entropy per channel)
  - microstate   : 4 + 4 + 16      =  24  (coverage, duration, transition matrix)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


STREAM_DIMS: Dict[str, int] = {
    "spectral": 118,
    "connectivity": 855,
    "complexity": 76,
    "microstate": 24,
}


class StreamEncoder(nn.Module):
    """Two-layer MLP that projects a flat biomarker vector to ``out_dim``."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(out_dim, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureStreamFusion(nn.Module):
    """Gated fusion of hand-crafted biomarker streams with the learned embedding.

    For each enabled stream, a learnable scalar gate (sigmoid-squashed)
    controls how much that stream contributes after projection. Disabling
    a stream at inference time is equivalent to zeroing its input vector.
    """

    def __init__(
        self,
        embed_dim: int,
        stream_dim: int = 64,
        streams: Optional[List[str]] = None,
        dropout: float = 0.3,
        stream_dropout: float = 0.2,
    ):
        super().__init__()
        if streams is None:
            streams = list(STREAM_DIMS.keys())
        unknown = set(streams) - set(STREAM_DIMS)
        if unknown:
            raise ValueError(f"Unknown feature streams: {sorted(unknown)}")

        self.streams = streams
        self.stream_dim = stream_dim
        self.stream_dropout = float(stream_dropout)
        self.encoders = nn.ModuleDict(
            {name: StreamEncoder(STREAM_DIMS[name], stream_dim, dropout) for name in streams}
        )
        self.gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in streams}
        )

        fused_dim = embed_dim + len(streams) * stream_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def gate_values(self) -> Dict[str, float]:
        return {name: torch.sigmoid(p).item() for name, p in self.gates.items()}

    def forward(
        self,
        embedding: torch.Tensor,
        feature_streams: Optional[Dict[str, torch.Tensor]] = None,
        enabled: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        """Fuse biomarker streams into the learned embedding.

        Args:
            embedding: ``[B, embed_dim]`` learned representation.
            feature_streams: dict ``{stream_name: [B, dim]}`` with the per-window
                hand-crafted vectors. Missing streams are treated as zeros.
            enabled: optional ablation mask (default: all enabled).
        """
        batch = embedding.shape[0]
        device = embedding.device
        if enabled is None:
            enabled = {name: True for name in self.streams}

        encoded: List[torch.Tensor] = []
        # Stream-level dropout: during training, randomly zero entire streams
        # so no single stream dominates the fused representation.
        active_names = list(self.streams)
        if self.training and self.stream_dropout > 0.0 and len(active_names) > 1:
            keep_prob = 1.0 - self.stream_dropout
            keep_mask = torch.rand(len(active_names), device=device) < keep_prob
            if not keep_mask.any():
                keep_mask[torch.randint(0, len(active_names), (1,), device=device)] = True
            keep_mask = keep_mask.tolist()
        else:
            keep_mask = [True] * len(active_names)

        for idx, name in enumerate(active_names):
            if not enabled.get(name, True) or not keep_mask[idx]:
                encoded.append(torch.zeros(batch, self.stream_dim, device=device))
                continue
            if feature_streams is not None and name in feature_streams:
                vec = feature_streams[name].to(device).float()
            else:
                vec = torch.zeros(batch, STREAM_DIMS[name], device=device)
            gate = torch.sigmoid(self.gates[name])
            encoded.append(gate * self.encoders[name](vec))

        fused = torch.cat([embedding] + encoded, dim=-1)
        return self.proj(fused)
