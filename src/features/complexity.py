"""
Complexity Feature Extraction Module.

This module provides functions for computing nonlinear complexity measures
from EEG signals, including Multiscale Entropy and Lempel-Ziv Complexity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
import warnings

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    warnings.warn("neurokit2 not available. Some complexity features will be limited.")

try:
    import antropy
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

from ..config.config import FEATURE_CONFIG, SAMPLING_RATE


def compute_sample_entropy(data: np.ndarray,
                           m: int = 2,
                           r: float = 0.15) -> float:
    """
    Compute Sample Entropy for a 1D signal.
    
    Args:
        data: 1D signal array
        m: Embedding dimension
        r: Tolerance (as fraction of SD)
        
    Returns:
        Sample entropy value
    """
    if HAS_ANTROPY:
        try:
            return antropy.sample_entropy(data, order=m, metric='chebyshev')
        except:
            pass
    
    if HAS_NEUROKIT:
        try:
            result = nk.entropy_sample(data, dimension=m, tolerance=r * np.std(data))
            return result
        except:
            pass
    
    # Fallback implementation
    return _sample_entropy_basic(data, m, r * np.std(data))


def _sample_entropy_basic(data: np.ndarray, m: int, r: float) -> float:
    """Basic sample entropy implementation."""
    N = len(data)
    
    if N < m + 2:
        return np.nan
    
    def _count_matches(template_length):
        templates = np.array([data[i:i + template_length] for i in range(N - template_length)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 2
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if B == 0:
        return np.nan
    
    return -np.log(A / B) if A > 0 else np.nan


def compute_multiscale_entropy(data: np.ndarray,
                                scales: int = 20,
                                m: int = 2,
                                r: float = 0.15) -> Tuple[np.ndarray, float]:
    """
    Compute Multiscale Entropy (MSE) for a 1D signal.
    
    MSE captures complexity across different temporal scales.
    
    Args:
        data: 1D signal array
        scales: Maximum scale factor
        m: Embedding dimension
        r: Tolerance (as fraction of SD)
        
    Returns:
        (mse_values, complexity_index) tuple
        - mse_values: Sample entropy at each scale
        - complexity_index: Area under MSE curve
    """
    if HAS_NEUROKIT:
        try:
            mse, info = nk.entropy_multiscale(
                data,
                scale=scales,
                dimension=m,
                tolerance=r * np.std(data),
                composite=False
            )
            # mse_values are in info
            mse_values = info.get('Value', np.array([mse]))
            complexity_index = np.sum(mse_values[~np.isnan(mse_values)])
            return mse_values, complexity_index
        except:
            pass
    
    # Manual implementation
    mse_values = []
    tolerance = r * np.std(data)
    
    for scale in range(1, scales + 1):
        # Coarse-grain the signal
        coarse_grained = _coarse_grain(data, scale)
        
        if len(coarse_grained) < m + 2:
            mse_values.append(np.nan)
            continue
        
        # Compute sample entropy at this scale
        se = _sample_entropy_basic(coarse_grained, m, tolerance)
        mse_values.append(se)
    
    mse_values = np.array(mse_values)
    complexity_index = np.nansum(mse_values)
    
    return mse_values, complexity_index


def _coarse_grain(data: np.ndarray, scale: int) -> np.ndarray:
    """Coarse-grain a signal by averaging non-overlapping windows."""
    n = len(data)
    n_segments = n // scale
    
    if n_segments == 0:
        return np.array([])
    
    # Truncate to multiple of scale
    data_truncated = data[:n_segments * scale]
    
    # Reshape and average
    coarse = data_truncated.reshape(n_segments, scale).mean(axis=1)
    
    return coarse


def compute_lempel_ziv_complexity(data: np.ndarray,
                                   normalize: bool = True) -> float:
    """
    Compute Lempel-Ziv Complexity for a 1D signal.
    
    LZC measures the number of distinct patterns in a binary sequence.
    
    Args:
        data: 1D signal array
        normalize: Whether to normalize by random sequence expectation
        
    Returns:
        Lempel-Ziv complexity value
    """
    if HAS_NEUROKIT:
        try:
            lzc = nk.complexity_lempelziv(data, normalize=normalize)
            return lzc
        except:
            pass
    
    # Manual implementation
    return _lempel_ziv_basic(data, normalize)


def _lempel_ziv_basic(data: np.ndarray, normalize: bool = True) -> float:
    """Basic Lempel-Ziv complexity implementation."""
    # Binarize signal using median threshold
    threshold = np.median(data)
    binary = ''.join(['1' if x > threshold else '0' for x in data])
    
    n = len(binary)
    if n == 0:
        return 0.0
    
    # Count distinct subsequences
    c = 1
    i = 0
    k = 1
    k_max = 1
    
    while i + k <= n:
        if binary[i:i+k] not in binary[:i+k-1]:
            c += 1
            i += k
            k = 1
            k_max = 1
        else:
            k += 1
            if k > k_max:
                k_max = k
    
    if normalize:
        # Normalize by expected complexity for random sequence
        if n > 0:
            c_max = n / np.log2(n) if n > 1 else 1
            return c / c_max
        return 0.0
    
    return float(c)


def compute_permutation_entropy(data: np.ndarray,
                                 order: int = 3,
                                 delay: int = 1,
                                 normalize: bool = True) -> float:
    """
    Compute Permutation Entropy for a 1D signal.
    
    Args:
        data: 1D signal array
        order: Embedding dimension
        delay: Time delay
        normalize: Whether to normalize
        
    Returns:
        Permutation entropy value
    """
    if HAS_ANTROPY:
        try:
            return antropy.perm_entropy(data, order=order, delay=delay, normalize=normalize)
        except:
            pass
    
    if HAS_NEUROKIT:
        try:
            return nk.entropy_permutation(data, dimension=order, delay=delay, normalize=normalize)
        except:
            pass
    
    # Fallback: return NaN
    return np.nan


def compute_dfa(data: np.ndarray,
                scales: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute Detrended Fluctuation Analysis (DFA).
    
    Args:
        data: 1D signal array
        scales: Scales for DFA
        
    Returns:
        (alpha, r_squared) tuple
        - alpha: DFA exponent (scaling exponent)
        - r_squared: R-squared of the fit
    """
    if HAS_NEUROKIT:
        try:
            dfa, info = nk.fractal_dfa(data, scale=scales)
            return dfa, info.get('R2', np.nan)
        except:
            pass
    
    return np.nan, np.nan


def extract_complexity_features(data: np.ndarray,
                                 sfreq: float = SAMPLING_RATE) -> Dict[str, np.ndarray]:
    """
    Extract all complexity features from epoched EEG data.
    
    Args:
        data: EEG data of shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        
    Returns:
        Dictionary of complexity features
    """
    n_epochs, n_channels, n_times = data.shape
    
    features = {
        'mse_ci': np.zeros((n_epochs, n_channels)),         # MSE Complexity Index
        'lzc': np.zeros((n_epochs, n_channels)),            # Lempel-Ziv Complexity
        'sample_entropy': np.zeros((n_epochs, n_channels)), # Sample Entropy
        'perm_entropy': np.zeros((n_epochs, n_channels)),   # Permutation Entropy
    }
    
    mse_scales = FEATURE_CONFIG.get('mse_scales', 20)
    mse_m = FEATURE_CONFIG.get('mse_m', 2)
    mse_r = FEATURE_CONFIG.get('mse_r', 0.15)
    
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            signal_data = data[epoch, ch, :]
            
            # Multiscale Entropy
            if FEATURE_CONFIG.get('compute_mse', True):
                try:
                    _, ci = compute_multiscale_entropy(
                        signal_data, 
                        scales=mse_scales, 
                        m=mse_m, 
                        r=mse_r
                    )
                    features['mse_ci'][epoch, ch] = ci
                except:
                    features['mse_ci'][epoch, ch] = np.nan
            
            # Lempel-Ziv Complexity
            if FEATURE_CONFIG.get('compute_lzc', True):
                try:
                    features['lzc'][epoch, ch] = compute_lempel_ziv_complexity(signal_data)
                except:
                    features['lzc'][epoch, ch] = np.nan
            
            # Sample Entropy (single scale)
            try:
                features['sample_entropy'][epoch, ch] = compute_sample_entropy(signal_data)
            except:
                features['sample_entropy'][epoch, ch] = np.nan
            
            # Permutation Entropy
            try:
                features['perm_entropy'][epoch, ch] = compute_permutation_entropy(signal_data)
            except:
                features['perm_entropy'][epoch, ch] = np.nan
    
    return features


def extract_complexity_features_from_epochs(epochs) -> Dict[str, np.ndarray]:
    """
    Extract complexity features from MNE Epochs object.
    
    Args:
        epochs: MNE Epochs object
        
    Returns:
        Dictionary of complexity features
    """
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    return extract_complexity_features(data, sfreq)
