"""
Configuration module for NeuroChronoGraph EEG Analysis Pipeline.

This module contains all global settings, paths, and hyperparameters
for the AD/FTD classification project.
"""

from pathlib import Path
import torch

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Data paths
RAW_DATA_PATH = DATA_ROOT
DERIVATIVES_PATH = DATA_ROOT / "derivatives"
PARTICIPANTS_FILE = DATA_ROOT / "participants.tsv"

# Output paths
PREPROCESSED_PATH = OUTPUT_ROOT / "preprocessed"
SOURCE_ESTIMATES_PATH = OUTPUT_ROOT / "source_estimates"
FEATURES_PATH = OUTPUT_ROOT / "features"
MODELS_PATH = OUTPUT_ROOT / "models"
FIGURES_PATH = OUTPUT_ROOT / "figures"
RESULTS_PATH = OUTPUT_ROOT / "results"

# Ensure output directories exist
for path in [PREPROCESSED_PATH, SOURCE_ESTIMATES_PATH, FEATURES_PATH, 
             MODELS_PATH, FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

# Class mapping
CLASS_MAPPING = {
    'A': 0,  # Alzheimer's Disease
    'F': 1,  # Frontotemporal Dementia
    'C': 2,  # Cognitively Normal (Control)
    'M': 3   # Mild Cognitive Impairment
}

CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']

# Number of subjects per class (updated from preprocessed data)
N_SUBJECTS = {
    'AD': 302,
    'FTD': 46,
    'CN': 198,
    'MCI': 14,
    'total': 560
}

# EEG Configuration
# Changed from 500Hz to 256Hz to minimize resampling artifacts
# Most datasets are 250-256Hz native; this avoids 4x upsampling from 128Hz files
# Nyquist = 128Hz, sufficient for all analysis bands (max gamma = 45Hz)
SAMPLING_RATE = 256  # Hz
N_CHANNELS = 19

# Channel names (10-20 system)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Channel to ROI mapping for sensor-level analysis
CHANNEL_TO_ROI = {
    'L_Frontal': ['Fp1', 'F3', 'F7'],
    'R_Frontal': ['Fp2', 'F4', 'F8'],
    'L_Temporal': ['T3', 'T5'],
    'R_Temporal': ['T4', 'T6'],
    'L_Central': ['C3'],
    'R_Central': ['C4'],
    'L_Parietal': ['P3'],
    'R_Parietal': ['P4'],
    'L_Occipital': ['O1'],
    'R_Occipital': ['O2'],
    'Midline': ['Fz', 'Cz', 'Pz']
}

# Standard 10-20 System 3D Coordinates (MNI-ish approximations for distance)
# Format: {Channel: (x, y, z)}
CHANNEL_COORDS_10_20 = {
    'Fp1': (-0.30, 0.95, -0.03), 'Fp2': (0.30, 0.95, -0.03),
    'F3': (-0.55, 0.67, 0.50),   'F4': (0.55, 0.67, 0.50),
    'C3': (-0.71, 0.00, 0.71),   'C4': (0.71, 0.00, 0.71),
    'P3': (-0.55, -0.67, 0.50),  'P4': (0.55, -0.67, 0.50),
    'O1': (-0.30, -0.95, -0.03), 'O2': (0.30, -0.95, -0.03),
    'F7': (-0.81, 0.59, -0.03),  'F8': (0.81, 0.59, -0.03),
    'T3': (-0.95, 0.00, -0.31),  'T4': (0.95, 0.00, -0.31),
    'T5': (-0.81, -0.59, -0.03), 'T6': (0.81, -0.59, -0.03),
    'Fz': (0.00, 0.72, 0.69),    'Cz': (0.00, 0.00, 1.00),
    'Pz': (0.00, -0.72, 0.69)
}

def get_physical_adjacency(sigma: float = 0.5, threshold: float = 0.2):
    """
    Compute Gaussian kernel adjacency matrix based on physical distances.
    
    Args:
        sigma: Width of the Gaussian kernel
        threshold: Minimum connection strength to keep (sparsification)
    """
    import numpy as np
    
    n_channels = len(CHANNEL_NAMES)
    coords = np.array([CHANNEL_COORDS_10_20[ch] for ch in CHANNEL_NAMES])
    
    # Compute pairwise Euclidean distances
    # shape: [N, 1, 3] - [1, N, 3] -> [N, N, 3] -> norm -> [N, N]
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    # Gaussian kernel: A_ij = exp(-dist^2 / (2*sigma^2))
    adj = np.exp(-dists**2 / (2 * sigma**2))
    
    # Remove self-loops (optional, but usually handled by graph layers)
    np.fill_diagonal(adj, 0)
    
    # Thresholding for sparsity (keep strong connections)
    adj[adj < threshold] = 0
    
    return adj

# 8-ROI coarse parcellation for source-level
ROI_NAMES_8 = [
    'L_Frontal', 'R_Frontal',
    'L_Temporal', 'R_Temporal', 
    'L_Parietal', 'R_Parietal',
    'L_Occipital', 'R_Occipital'
]

# ==============================================================================
# PREPROCESSING CONFIGURATION
# ==============================================================================

PREPROCESSING = {
    # Filtering
    'l_freq': 0.5,           # High-pass frequency (Hz)
    'h_freq': 45.0,          # Low-pass frequency (Hz)
    'notch_freq': 50.0,      # Notch filter frequency (Hz)

    # Epoching
    'epoch_duration': 4.0,   # Epoch length in seconds
    'epoch_overlap': 0.25,   # Overlap ratio (0.25 = 25% overlap - balance data quantity vs overfitting)

    # Artifact rejection
    'reject_threshold': 200e-6,  # 200 µV in Volts (Relaxed from 100uV to prevent dropping all epochs)

    # Reference
    'reference': 'average',  # 'average' or specific channels

    # Use preprocessed data from derivatives
    'use_derivatives': True,
}

# ==============================================================================
# DATASET WRAPPER CONFIGURATION (Anti-Overfitting)
# ==============================================================================

DATASET_CONFIG = {
    'n_times': 1024,                   # Window size in samples (4 sec @ 256Hz)
    'overlap_ratio': 0.25,             # 25% overlap (balance training samples vs overfitting)
    'max_windows_per_subject': 50,     # Limit windows per subject
    'enable_augmentation': True,       # Enable minority class augmentation
    'sfreq': 256.0,                    # Sampling frequency (Must match global SAMPLING_RATE)
    'compute_feature_streams': False,  # Disabled: ablation showed streams hurt CV/hold-out (see paper §Ablation)
}

# ==============================================================================
# FREQUENCY BANDS
# ==============================================================================

FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

# Bands used for connectivity analysis
CONNECTIVITY_BANDS = ['theta', 'alpha', 'beta']

# ==============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ==============================================================================

FEATURE_CONFIG = {
    # Spectral features
    'compute_psd': True,
    'psd_method': 'welch',
    'psd_n_fft': 1024,
    
    # Complexity features
    'compute_mse': True,
    'mse_scales': 20,
    'mse_m': 2,           # Embedding dimension
    'mse_r': 0.15,        # Tolerance (fraction of SD)
    
    'compute_lzc': True,
    
    # Connectivity features
    'connectivity_method': 'wpli',  # 'wpli', 'pli', 'coh', 'plv'
    
    # Dynamic connectivity
    'dynamic_window_size': 4.0,    # seconds
    'dynamic_step_size': 2.0,      # seconds (50% overlap)
    
    # Graph metrics
    'compute_graph_metrics': True,
    'graph_threshold': 0.3,        # Threshold for binarization
}

# ==============================================================================
# SOURCE LOCALIZATION CONFIGURATION
# ==============================================================================

SOURCE_CONFIG = {
    # Template brain
    'template': 'fsaverage',
    
    # Source space
    'spacing': 'ico4',      # Source space resolution
    
    # Inverse method
    'method': 'eLORETA',    # 'eLORETA', 'sLORETA', 'MNE', 'dSPM'
    'snr': 3.0,
    'loose': 0.2,
    'depth': 0.8,
    
    # Parcellation
    'parcellation': 'aparc',  # Desikan-Killiany
    'n_rois': 8,              # Coarse parcellation
    
    # Label extraction mode
    'label_mode': 'mean_flip',
}

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

MODEL_CONFIG = {
    # Architecture
    'sensor_node_features': 32,
    'source_node_features': 32,
    'hidden_dim': 64,
    'n_heads': 4,
    'n_temporal_steps': 30,
    'n_classes': 4,
    'dropout': 0.3,
    
    # Metadata
    'metadata_dim': 3,  # Age, Sex, MMSE
}

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

TRAINING_CONFIG = {
    # Optimizer
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    
    # Scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 100,
    'eta_min': 1e-6,
    
    # Training
    'n_epochs': 100,
    'batch_size': 8,
    'gradient_clip': 1.0,
    
    # Early stopping
    'patience': 15,
    
    # Class weights (inverse frequency)
    'class_weights': [1.0, 1.5, 1.1, 1.8],  # AD, FTD, CN, MCI (Approx weights)
    
    # Focal loss
    'focal_gamma': 2.0,
    
    # Regularization
    'drop_edge_rate': 0.1,
}

# ==============================================================================
# VALIDATION CONFIGURATION
# ==============================================================================

VALIDATION_CONFIG = {
    # Cross-validation strategy
    'cv_strategy': 'LOSO',    # Leave-One-Subject-Out
    
    # Nested CV for hyperparameter tuning
    'inner_cv_folds': 3,
    
    # Metrics
    'primary_metric': 'f1_macro',
    'metrics': ['accuracy', 'f1_macro', 'f1_weighted', 'mcc'],
    
    # Statistical tests
    'n_permutations': 1000,
    'alpha': 0.05,
    'correction_method': 'fdr_bh',
}

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_JOBS = 8  # Optimal for Intel Ultra 9 (24 cores, use subset for I/O)

# ==============================================================================
# VISUALIZATION CONFIGURATION (Publication Ready)
# ==============================================================================

VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8-whitegrid',
    'context': 'paper',
    'font_scale': 1.4,
    'palette': {
        'AD': '#D55E00',   # Vermilion
        'FTD': '#0072B2',  # Blue
        'CN': '#009E73',   # Bluish Green
        'MCI': '#E69F00',  # Orange
        'General': '#333333'
    },
    'dpi': 300,
    'cmap_sequential': 'Blues',
    'cmap_diverging': 'RdBu_r'
}

# ==============================================================================
# GPU OPTIMIZATION (HP Omen 16 - RTX 5070 8GB VRAM)
# ==============================================================================

GPU_CONFIG = {
    'enabled': torch.cuda.is_available(),
    'mixed_precision': True,       # FP16 training for faster GPU performance
    'gradient_checkpointing': True,  # Memory efficient for 8GB VRAM
    'pin_memory': True,
    'non_blocking': True,
    'cudnn_benchmark': True,       # Optimize conv operations
    'optimal_batch_size': 64,      # Increased for better GPU utilization
    'accumulation_steps': 1,       # No accumulation needed with larger batch
}

# ==============================================================================
# DATALOADER OPTIMIZATION (Intel Ultra 9 with 24 cores)
# ==============================================================================

DATALOADER_CONFIG = {
    'num_workers': 0,              # Set to 0 to fix MemoryError on Windows (Avoids spawn/pickle overhead)
    'persistent_workers': False,   # Must be False if num_workers is 0
    'prefetch_factor': None,       # Must be None if num_workers is 0
    'pin_memory': True,            # Faster GPU transfer
    'drop_last': False,
}

# ==============================================================================
# MEMORY OPTIMIZATION
# ==============================================================================

MEMORY_CONFIG = {
    'max_epochs_in_memory': 50,    # Limit epochs loaded at once per subject
    'use_mmap': True,              # Memory-mapped file loading
    'clear_cache_interval': 10,    # Clear CUDA cache every N batches
}

# ==============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ==============================================================================

RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
