"""
EEG Data Augmentation Module.

Implements EEG-specific augmentations for improved generalization:
- Time shift
- Amplitude scaling
- Gaussian noise injection
- Channel dropout
- Minority class oversampling (MCI)
- Mixup augmentation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List


# Class labels
CLASS_AD = 0
CLASS_FTD = 1
CLASS_CN = 2
CLASS_MCI = 3


class EEGAugmentation:
    """
    EEG-specific data augmentation for training regularization.
    """
    
    def __init__(self,
                 time_shift_max: int = 100,
                 amplitude_scale_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.02,
                 channel_dropout_prob: float = 0.1,
                 augment_prob: float = 0.5):
        """
        Args:
            time_shift_max: Maximum samples to shift (±)
            amplitude_scale_range: (min, max) scaling factor
            noise_std: Standard deviation of Gaussian noise
            channel_dropout_prob: Probability of zeroing a channel
            augment_prob: Probability of applying each augmentation
        """
        self.time_shift_max = time_shift_max
        self.amplitude_scale_range = amplitude_scale_range
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
        self.augment_prob = augment_prob
        
    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly shift signal in time.
        
        Args:
            x: EEG tensor [channels, time] or [batch, channels, time]
        Returns:
            Shifted tensor
        """
        if np.random.rand() > self.augment_prob:
            return x
            
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return x
            
        return torch.roll(x, shifts=shift, dims=-1)
    
    def amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale amplitude.
        
        Args:
            x: EEG tensor
        Returns:
            Scaled tensor
        """
        if np.random.rand() > self.augment_prob:
            return x
            
        scale = np.random.uniform(*self.amplitude_scale_range)
        return x * scale
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise.
        
        Args:
            x: EEG tensor
        Returns:
            Noisy tensor
        """
        if np.random.rand() > self.augment_prob:
            return x
            
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero out channels.
        
        Args:
            x: EEG tensor [channels, time] or [batch, channels, time]
        Returns:
            Tensor with some channels zeroed
        """
        if np.random.rand() > self.augment_prob:
            return x
            
        # Determine channel dimension
        if x.dim() == 2:
            n_channels = x.shape[0]
            mask = torch.rand(n_channels, 1) > self.channel_dropout_prob
            return x * mask.to(x.device)
        else:
            n_channels = x.shape[1]
            batch_size = x.shape[0]
            mask = torch.rand(batch_size, n_channels, 1) > self.channel_dropout_prob
            return x * mask.to(x.device)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations.
        
        Args:
            x: EEG tensor
        Returns:
            Augmented tensor
        """
        x = self.time_shift(x)
        x = self.amplitude_scale(x)
        x = self.add_noise(x)
        x = self.channel_dropout(x)
        return x


def augment_batch(batch: Dict, augmenter: Optional[EEGAugmentation] = None) -> Dict:
    """
    Apply augmentation to a batch dictionary.

    Args:
        batch: Batch dict with 'x' key containing EEG data
        augmenter: EEGAugmentation instance

    Returns:
        Augmented batch
    """
    if augmenter is None:
        return batch

    batch['x'] = augmenter(batch['x'])
    return batch


class MinorityClassAugmenter:
    """
    Augmenter specifically designed for minority class oversampling.
    Creates synthetic samples for MCI and rare classes.
    """

    def __init__(self,
                 augmentation_factor: Dict[int, int] = None,
                 augmenter: Optional[EEGAugmentation] = None):
        """
        Args:
            augmentation_factor: Dict mapping class label to augmentation multiplier
                                e.g., {3: 10} means MCI gets 10x augmented copies
            augmenter: Base EEGAugmentation instance for transformations
        """
        # Default: Augment MCI (3) heavily, FTD (1) moderately
        self.augmentation_factor = augmentation_factor or {
            CLASS_MCI: 10,  # 10x augmentation for MCI
            CLASS_FTD: 2,   # 2x augmentation for FTD
            CLASS_AD: 1,
            CLASS_CN: 1
        }

        # Use aggressive augmentation for synthetic samples
        self.augmenter = augmenter or EEGAugmentation(
            time_shift_max=200,           # Larger shifts
            amplitude_scale_range=(0.85, 1.15),  # More variation
            noise_std=0.03,               # More noise
            channel_dropout_prob=0.15,    # More dropout
            augment_prob=0.8              # Higher probability
        )

    def generate_augmented_samples(self,
                                   x: np.ndarray,
                                   label: int,
                                   band_features: Dict[str, np.ndarray] = None,
                                   metadata: np.ndarray = None) -> List[Dict]:
        """
        Generate augmented samples for a single input.

        Args:
            x: EEG data [channels, time]
            label: Class label (0-3)
            band_features: Optional band-filtered features
            metadata: Optional metadata array

        Returns:
            List of augmented sample dictionaries
        """
        factor = self.augmentation_factor.get(label, 1)

        if factor <= 1:
            return []  # No augmentation needed

        augmented = []
        x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x

        for _ in range(factor - 1):  # -1 because original is already included
            # Apply augmentation
            aug_x = self.augmenter(x_tensor.clone())

            sample = {
                'x': aug_x,
                'label': torch.tensor(label, dtype=torch.long),
                'is_augmented': True
            }

            # Augment band features if provided
            if band_features is not None:
                aug_bands = {}
                for band_name, band_data in band_features.items():
                    band_tensor = torch.from_numpy(band_data) if isinstance(band_data, np.ndarray) else band_data
                    # Apply same augmentations to band features
                    aug_band = self.augmenter.amplitude_scale(band_tensor.clone())
                    aug_band = self.augmenter.add_noise(aug_band)
                    aug_bands[band_name] = aug_band
                sample['band_features'] = aug_bands

            if metadata is not None:
                # Add small perturbation to metadata (age: ±1 year, MMSE: ±1 point)
                meta_tensor = torch.from_numpy(metadata) if isinstance(metadata, np.ndarray) else metadata.clone()
                meta_tensor = meta_tensor.float()
                # Perturb age (index 0) by ±1
                meta_tensor[0] += np.random.uniform(-1, 1)
                # Perturb MMSE (index 2) by ±1
                if len(meta_tensor) > 2:
                    meta_tensor[2] += np.random.uniform(-1, 1)
                sample['metadata'] = meta_tensor

            augmented.append(sample)

        return augmented


class MixupAugmenter:
    """
    Implements Mixup augmentation for EEG data.
    Creates interpolated samples between classes.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha

    def mixup(self,
              x1: torch.Tensor,
              x2: torch.Tensor,
              y1: int,
              y2: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create mixup sample.

        Args:
            x1, x2: EEG tensors to mix
            y1, y2: Labels (0-3)

        Returns:
            (mixed_x, soft_label) where soft_label is [4] one-hot with mixing
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix signals
        mixed_x = lam * x1 + (1 - lam) * x2

        # Create soft labels
        soft_label = torch.zeros(4)
        soft_label[y1] = lam
        soft_label[y2] = 1 - lam

        return mixed_x, soft_label

    def mixup_within_class(self,
                           samples: List[torch.Tensor],
                           label: int,
                           n_synthetic: int = 5) -> List[Tuple[torch.Tensor, int]]:
        """
        Create mixup samples within the same class.
        Useful for MCI class expansion.

        Args:
            samples: List of EEG tensors from same class
            label: Class label
            n_synthetic: Number of synthetic samples to create

        Returns:
            List of (mixed_x, label) tuples
        """
        if len(samples) < 2:
            return []

        synthetic = []
        for _ in range(n_synthetic):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(len(samples), 2, replace=False)
            x1, x2 = samples[idx1], samples[idx2]

            # Mix with random coefficient
            lam = np.random.uniform(0.3, 0.7)
            mixed = lam * x1 + (1 - lam) * x2

            synthetic.append((mixed, label))

        return synthetic


def calculate_class_weights(labels: np.ndarray,
                           method: str = 'inverse_sqrt',
                           min_weight: float = 0.1,
                           max_weight: float = 10.0) -> np.ndarray:
    """
    Calculate class weights for imbalanced data.

    Args:
        labels: Array of class labels
        method: 'inverse', 'inverse_sqrt', or 'effective'
        min_weight: Minimum weight (prevents overweighting)
        max_weight: Maximum weight cap

    Returns:
        Array of weights for each class
    """
    class_counts = np.bincount(labels, minlength=4)
    total = len(labels)
    n_classes = len(class_counts)

    if method == 'inverse':
        # Standard inverse frequency
        weights = total / (n_classes * class_counts + 1e-6)
    elif method == 'inverse_sqrt':
        # Softer weighting (recommended for extreme imbalance)
        weights = np.sqrt(total / (n_classes * class_counts + 1e-6))
    elif method == 'effective':
        # Effective number of samples (Class-Balanced Loss)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    weights = weights / weights.sum() * n_classes

    # Clip to bounds
    weights = np.clip(weights, min_weight, max_weight)

    return weights.astype(np.float32)
