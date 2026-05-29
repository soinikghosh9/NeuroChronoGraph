"""Models module initialization."""

# V1 Models (Original architecture)
from .neuro_gnn import (
    SensorLevelGNN,
    SourceLevelGNN,
    DualTrackNeuroGNN,
    SingleTrackGNN
)

# Loss functions (consolidated)
from .loss_functions import (
    FocalLoss,
    LabelSmoothingLoss,
    CombinedLoss,
    MultiObjectiveLoss,
    create_loss_function
)

# V2 Models (Advanced architecture - recommended)
from .v2 import (
    NeuroChronoGraphV2,
    create_neuro_chrono_graph_v2,
    AdaptiveGraphLearning,
    ModularBrainTransformer,
    TemporalGraphTransformer,
    HierarchicalFiLMConditioner,
    ContrastivePretrainer,
    BarlowTwinsLoss
)

__all__ = [
    # V1
    'DualTrackNeuroGNN',
    'SingleTrackGNN',
    'SensorLevelGNN',
    'SourceLevelGNN',
    
    # V2 (recommended)
    'NeuroChronoGraphV2',
    'create_neuro_chrono_graph_v2',
    
    # Losses
    'FocalLoss',
    'CombinedLoss',
    'MultiObjectiveLoss',
    'create_loss_function',
]
