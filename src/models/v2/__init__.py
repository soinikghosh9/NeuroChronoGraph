"""
NeuroChronoGraph v2 Module.

Advanced deep learning architecture for dementia classification.
"""

from .neuro_chrono_graph_v2 import (
    NeuroChronoGraphV2,
    create_neuro_chrono_graph_v2,
    EEGEncoder
)

from .adaptive_graph import (
    AdaptiveGraphLearning,
    GatedGraphConvolution
)

from .modular_brain_transformer import (
    ModularBrainTransformer,
    TemporalGraphTransformer,
    IntraModuleAttention,
    InterModuleAttention,
    DEFAULT_BRAIN_MODULES
)

from .clinical_conditioning import (
    FiLMLayer,
    HierarchicalFiLMConditioner,
    DiseaseStageEncoder,
    UncertaintyHead
)

from .contrastive_pretrain import (
    ContrastivePretrainer,
    BarlowTwinsLoss,
    EEGAugmentation,
    ProjectionHead,
    SubjectInvariantLoss,
    DisentangledEncoder
)

from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    ExplainabilityLoss,
    MMDLoss,
    MultiObjectiveLoss,
    CurriculumScheduler
)

__all__ = [
    # Main model
    'NeuroChronoGraphV2',
    'create_neuro_chrono_graph_v2',
    
    # Graph components
    'AdaptiveGraphLearning',
    'GatedGraphConvolution',
    
    # Transformer components
    'ModularBrainTransformer',
    'TemporalGraphTransformer',
    
    # Conditioning
    'HierarchicalFiLMConditioner',
    'UncertaintyHead',
    
    # Pretraining
    'ContrastivePretrainer',
    'BarlowTwinsLoss',
    
    # Losses
    'FocalLoss',
    'MultiObjectiveLoss',
]
