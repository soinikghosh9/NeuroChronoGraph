"""NeuroChronoGraph v5: spectral-first, montage-agnostic, subject-level."""

from src.models.v5.losses import (SplitHalfConsistencyLoss, consistency_diagnostics,
                                  split_half_posteriors)
from src.models.v5.montage import MontageEmbedding, masked_adjacency, resolve_coords
from src.models.v5.neuro_chrono_graph_v5 import (AttentionPool, BandWeightedPrior,
                                                 NCGv5MIL, NeuroChronoGraphV5)
from src.models.v5.spectral_nodes import BAND_NAMES, BANDS, SpectralNodeFeatures

__all__ = [
    "NeuroChronoGraphV5", "NCGv5MIL", "AttentionPool", "BandWeightedPrior",
    "SpectralNodeFeatures", "BANDS", "BAND_NAMES",
    "MontageEmbedding", "resolve_coords", "masked_adjacency",
    "SplitHalfConsistencyLoss", "split_half_posteriors", "consistency_diagnostics",
]
