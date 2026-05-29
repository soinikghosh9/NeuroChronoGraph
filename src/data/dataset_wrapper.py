"""
EEG Dataset Wrapper.

Converts MNE Raw buffers into PyTorch sliding windows.
Supports:
- Window limiting per subject (prevents overfitting)
- Minority class augmentation (MCI, FTD)
- Configurable overlap
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional
from scipy.signal import butter, filtfilt

from .augmentation import EEGAugmentation, MinorityClassAugmenter
from .window_features import compute_all_streams

# Class labels
CLASS_AD = 0
CLASS_FTD = 1
CLASS_CN = 2
CLASS_MCI = 3


class EEGDatasetWrapper(Dataset):
    """
    Wraps a BaseDataset loader.
    Preloads data into RAM (float32) for maximum efficiency and stability.

    Features:
    - Limits windows per subject to prevent overfitting
    - Supports online augmentation for minority classes
    - Configurable overlap ratio
    """

    def __init__(self, loader, config):
        self.loader = loader
        self.config = config
        self.window_size = config['n_times']  # 2000 samples

        # Configurable overlap: 0.0 = no overlap, 0.5 = 50% overlap
        overlap_ratio = config.get('overlap_ratio', 0.0)  # DEFAULT: No overlap (changed from 0.5)
        self.stride = int(self.window_size * (1 - overlap_ratio))
        if self.stride < 1:
            self.stride = 1

        self.sfreq = config.get('sfreq', 200.0)  # Default sampling rate

        # Limit windows per subject to prevent overfitting
        self.max_windows_per_subject = config.get('max_windows_per_subject', 50)

        # Optional hand-crafted biomarker streams returned alongside band features
        # (used by feature_streams= setting in NeuroChronoGraphV3 for ablation).
        self.compute_streams = config.get('compute_feature_streams', False)
        # Stream values are deterministic functions of (orig_sid, start) — augmentation
        # is applied AFTER stream extraction — so caching by that key amortizes the
        # ~60 ms/window cost across epochs (first epoch pays once, rest are free).
        self._stream_cache: dict = {} if self.compute_streams else None

        # Enable/disable augmentation
        self.enable_augmentation = config.get('enable_augmentation', True)
        self.augmenter = None
        self.minority_augmenter = None

        if self.enable_augmentation:
            self.augmenter = EEGAugmentation(
                time_shift_max=100,
                amplitude_scale_range=(0.9, 1.1),
                noise_std=0.02,
                channel_dropout_prob=0.1,
                augment_prob=0.3  # Lower prob for training augmentation
            )
            self.minority_augmenter = MinorityClassAugmenter()

        # Define Bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        self.filters = self._design_filters()

        self.samples = []  # List of (subject_id, start_idx, is_augmented)
        self.groups = []   # Subject IDs for splitting
        self.labels = []   # 0, 1, 2, 3

        # Data Store: {sid: np.array(channels, times)}
        self.data_store = {}

        # Augmented samples storage (for minority classes)
        self.augmented_samples = []  # List of pre-computed augmented samples

        print(f"[{loader.__class__.__name__}] Preloading data...")
        self._prepare_and_load()

        # Print class distribution
        self._print_class_distribution()
        
    def _design_filters(self):
        """Pre-calculate Butterworth filters for each band."""
        filters = {}
        nyq = 0.5 * self.sfreq
        for band, (l, h) in self.bands.items():
            b, a = butter(4, [l / nyq, h / nyq], btype='band')
            filters[band] = (b, a)
        return filters
        
    def _apply_filters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply filters to a data window [n_channels, n_times]."""
        filtered = {}
        for band, (b, a) in self.filters.items():
            # Apply filter along time axis (axis=-1)
            # Using float32 for speed, filtfilt handles the rest
            filt_data = filtfilt(b, a, data, axis=-1).astype(np.float32)
            filtered[band] = filt_data
        return filtered
        
    def _prepare_and_load(self):
        """
        Pre-calculate indices and load data into RAM.
        Applies window limiting and generates augmented samples for minority classes.
        """
        ids = self.loader.get_subject_ids()

        # Track samples by class for augmentation
        class_samples = {CLASS_AD: [], CLASS_FTD: [], CLASS_CN: [], CLASS_MCI: []}

        for sid in ids:
            try:
                # 1. Get metadata
                info = self.loader.get_subject_info(sid)
                label = info['label']

                # 2. Load Raw Data
                raw = self.loader.load_raw(sid)

                # Convert to float32 numpy array to save RAM (MNE uses float64)
                data = raw.get_data().astype(np.float32)

                n_samples = data.shape[1]

                # 3. Store in RAM
                self.data_store[sid] = data

                # 4. Generate windows with limiting
                # Handle recordings shorter than window_size by padding
                min_required_samples = self.window_size // 2  # At least half window size

                if n_samples >= self.window_size:
                    # Normal case: generate sliding windows
                    possible_starts = list(range(0, n_samples - self.window_size + 1, self.stride))

                    # Limit windows per subject
                    if len(possible_starts) > self.max_windows_per_subject:
                        # Random subsample (deterministic with seed based on sid)
                        np.random.seed(hash(sid) % (2**32))
                        selected_starts = np.random.choice(
                            possible_starts,
                            self.max_windows_per_subject,
                            replace=False
                        )
                        np.random.seed(None)  # Reset seed
                    else:
                        selected_starts = possible_starts

                    for start in selected_starts:
                        self.samples.append((sid, start, False))  # False = not augmented
                        self.groups.append(sid)
                        self.labels.append(label)

                        # Store sample reference for minority class augmentation
                        if label in [CLASS_MCI, CLASS_FTD]:
                            class_samples[label].append((sid, start, info['metadata']))

                elif n_samples >= min_required_samples:
                    # Short recording: create a single padded sample at start=0
                    # The __getitem__ method will handle padding
                    self.samples.append((sid, 0, False))
                    self.groups.append(sid)
                    self.labels.append(label)

                    # Store sample reference for minority class augmentation
                    if label in [CLASS_MCI, CLASS_FTD]:
                        class_samples[label].append((sid, 0, info['metadata']))
                else:
                    print(f"  Warning: {sid} has only {n_samples} samples (need at least {min_required_samples}), skipping.")

                # Clean up MNE object
                del raw

            except Exception as e:
                print(f"Skipping {sid}: {e}")

        # 5. Generate augmented samples for minority classes
        if self.enable_augmentation and self.minority_augmenter:
            self._generate_minority_augmentations(class_samples)

    def _generate_minority_augmentations(self, class_samples: Dict):
        """
        Generate augmented samples for minority classes.
        """
        # Augmentation multipliers (MINIMAL to prevent synthetic bias)
        # MCI augmentation was causing AD→MCI confusion via staging head
        # Reduced to minimum necessary for class balance
        aug_factors = {
            CLASS_MCI: 2,   # 2x augmentation for MCI (was 3x - still too aggressive)
            CLASS_FTD: 2,   # 2x augmentation for FTD
        }

        for label, factor in aug_factors.items():
            samples = class_samples.get(label, [])
            if not samples:
                continue

            n_original = len(samples)
            n_to_add = n_original * (factor - 1)

            print(f"  Augmenting class {label}: {n_original} -> {n_original + n_to_add} samples")

            for _ in range(factor - 1):
                for sid, start, metadata in samples:
                    # Create augmented sample reference
                    self.samples.append((sid, start, True))  # True = augmented
                    self.groups.append(f"{sid}_aug")  # Different group to avoid leakage
                    self.labels.append(label)

    def _print_class_distribution(self):
        """Print class distribution after loading."""
        labels_arr = np.array(self.labels)
        class_names = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
        print("  Class Distribution:")
        for idx in range(4):
            count = (labels_arr == idx).sum()
            print(f"    {class_names[idx]}: {count} samples")
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sid = sample_info[0]
        start = sample_info[1]
        is_augmented = sample_info[2] if len(sample_info) > 2 else False

        # Handle augmented group ID (e.g., "sub-001_aug")
        orig_sid = sid.replace("_aug", "") if isinstance(sid, str) and "_aug" in sid else sid

        # Retrieve preloaded data
        full_data = self.data_store[orig_sid]
        end = start + self.window_size

        # Slicing numpy array is extremely fast
        data = full_data[:, start:end].copy()  # Copy for augmentation safety

        # Check integrity
        if data.shape[1] != self.window_size:
            diff = self.window_size - data.shape[1]
            data = np.pad(data, ((0, 0), (0, diff)), mode='constant')

        # ========== PER-SAMPLE Z-SCORE NORMALIZATION ==========
        # Critical for combining multi-source datasets with different recording equipment
        # Normalizes each channel independently within the sample window
        eps = 1e-8
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + eps
        data = (data - mean) / std

        # Convert to tensor
        x = torch.from_numpy(data.astype(np.float32))

        # Apply augmentation if flagged
        if is_augmented and self.augmenter is not None:
            x = self.augmenter(x)

        # Get metadata
        info = self.loader.get_subject_info(orig_sid)

        # Generate Band Features
        band_features_np = self._apply_filters(data)
        band_features_torch = {k: torch.from_numpy(v) for k, v in band_features_np.items()}

        # Apply augmentation to band features if needed
        if is_augmented and self.augmenter is not None:
            for band_name in band_features_torch:
                band_features_torch[band_name] = self.augmenter.amplitude_scale(
                    band_features_torch[band_name]
                )
                band_features_torch[band_name] = self.augmenter.add_noise(
                    band_features_torch[band_name]
                )

        sample = {
            'x': x,
            'band_features': band_features_torch,
            'label': torch.tensor(info['label'], dtype=torch.long),
            'metadata': torch.tensor(info['metadata'], dtype=torch.float32),
            'subject_id': orig_sid,
            'is_augmented': is_augmented,
        }

        if self.compute_streams:
            cache_key = (orig_sid, int(start))
            cached = self._stream_cache.get(cache_key)
            if cached is None:
                cached = compute_all_streams(data, band_features_np, self.sfreq)
                self._stream_cache[cache_key] = cached
            sample['feature_streams'] = {
                k: torch.from_numpy(v) for k, v in cached.items()
            }

        return sample
