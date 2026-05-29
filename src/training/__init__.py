"""Training module initialization."""

from .loso_validator import LOSOValidator
from .training_utils import (
    MixedPrecisionTrainer,
    EEGDataset,
    load_real_data,
    create_optimized_loader,
    get_loso_splits,
    setup_training_environment
)
