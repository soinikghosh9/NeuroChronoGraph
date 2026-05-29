"""
Directed Connectivity Module.

This module provides functions for computing directed (effective)
connectivity measures including Transfer Entropy and Partial Directed Coherence.
"""

import numpy as np
from scipy.signal import hilbert
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..config.config import SAMPLING_RATE, FREQUENCY_BANDS


def compute_transfer_entropy(source: np.ndarray,
                              target: np.ndarray,
                              k: int = 1,
                              l: int = 1,
                              n_bins: int = 8) -> float:
    """
    Compute Transfer Entropy from source to target.
    
    Transfer Entropy measures the directed information flow from
    source to target, accounting for the target's own history.
    
    TE(X→Y) = H(Y_t | Y_t-1:t-l) - H(Y_t | Y_t-1:t-l, X_t-1:t-k)
    
    Args:
        source: Source time series
        target: Target time series
        k: History length for source
        l: History length for target
        n_bins: Number of bins for discretization
        
    Returns:
        Transfer entropy value (bits)
    """
    n = len(target)
    max_delay = max(k, l)
    
    if n <= max_delay + 1:
        return 0.0
    
    # Discretize signals
    source_bins = _discretize(source, n_bins)
    target_bins = _discretize(target, n_bins)
    
    # Build joint probability distributions
    # P(Y_t, Y_past, X_past)
    y_current = target_bins[max_delay:]
    y_past = np.column_stack([target_bins[max_delay-i:-i if i > 0 else None] 
                               for i in range(1, l+1)])
    x_past = np.column_stack([source_bins[max_delay-i:-i if i > 0 else None] 
                               for i in range(1, k+1)])
    
    # Compute entropies
    # H(Y_t | Y_past)
    h_y_given_ypast = _conditional_entropy(y_current, y_past)
    
    # H(Y_t | Y_past, X_past)
    joint_past = np.column_stack([y_past, x_past])
    h_y_given_joint = _conditional_entropy(y_current, joint_past)
    
    # Transfer Entropy
    te = h_y_given_ypast - h_y_given_joint
    
    return max(0, te)  # TE should be non-negative


def _discretize(signal: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize signal into bins."""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    
    if signal_max == signal_min:
        return np.zeros(len(signal), dtype=int)
    
    bins = np.linspace(signal_min, signal_max, n_bins + 1)
    return np.digitize(signal, bins[:-1]) - 1


def _conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    """Compute conditional entropy H(Y|X)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    n = len(y)
    
    # Create joint states
    x_states = {}
    for i in range(n):
        x_key = tuple(x[i])
        if x_key not in x_states:
            x_states[x_key] = []
        x_states[x_key].append(y[i])
    
    # Compute conditional entropy
    h = 0
    for x_key, y_values in x_states.items():
        p_x = len(y_values) / n
        # Entropy of Y given X = x
        y_unique, y_counts = np.unique(y_values, return_counts=True)
        y_probs = y_counts / len(y_values)
        h_y_x = -np.sum(y_probs * np.log2(y_probs + 1e-10))
        h += p_x * h_y_x
    
    return h


def compute_transfer_entropy_matrix(data: np.ndarray,
                                     k: int = 1,
                                     l: int = 1) -> np.ndarray:
    """
    Compute Transfer Entropy matrix between all channel pairs.
    
    Args:
        data: EEG data (channels x time)
        k: Source history length
        l: Target history length
        
    Returns:
        TE matrix (n_channels x n_channels)
    """
    n_channels = data.shape[0]
    te_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                te_matrix[i, j] = compute_transfer_entropy(
                    data[i], data[j], k, l
                )
    
    return te_matrix


def compute_partial_directed_coherence(data: np.ndarray,
                                        order: int = 10,
                                        sfreq: float = SAMPLING_RATE,
                                        fmin: float = 1,
                                        fmax: float = 45,
                                        n_freqs: int = 50) -> Dict[str, np.ndarray]:
    """
    Compute Partial Directed Coherence (PDC).
    
    PDC measures directed connectivity in the frequency domain
    based on multivariate autoregressive (MVAR) modeling.
    
    Args:
        data: EEG data (channels x time)
        order: MVAR model order
        sfreq: Sampling frequency
        fmin, fmax: Frequency range
        n_freqs: Number of frequency bins
        
    Returns:
        Dictionary with PDC matrices per frequency
    """
    n_channels, n_times = data.shape
    
    # Fit MVAR model
    A = _fit_mvar(data, order)
    
    # Compute PDC for each frequency
    freqs = np.linspace(fmin, fmax, n_freqs)
    pdc_matrices = {}
    
    for f in freqs:
        pdc = _compute_pdc_at_freq(A, f, sfreq)
        pdc_matrices[f'{f:.1f}Hz'] = pdc
    
    # Compute band-averaged PDC
    for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
        band_freqs = freqs[(freqs >= f_low) & (freqs <= f_high)]
        if len(band_freqs) > 0:
            band_pdc = np.mean([
                _compute_pdc_at_freq(A, f, sfreq) 
                for f in band_freqs
            ], axis=0)
            pdc_matrices[f'{band_name}_pdc'] = band_pdc
    
    return pdc_matrices


def _fit_mvar(data: np.ndarray, order: int) -> np.ndarray:
    """
    Fit Multivariate Autoregressive model.
    
    Uses least squares estimation.
    
    Args:
        data: Data matrix (channels x time)
        order: Model order
        
    Returns:
        MVAR coefficients (order x channels x channels)
    """
    n_channels, n_times = data.shape
    
    # Build design matrix
    Y = data[:, order:].T  # (n_times - order, n_channels)
    
    X_blocks = []
    for p in range(1, order + 1):
        X_blocks.append(data[:, order - p:n_times - p].T)
    X = np.hstack(X_blocks)  # (n_times - order, order * n_channels)
    
    # Solve least squares
    try:
        A_flat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        A_flat = np.zeros((order * n_channels, n_channels))
    
    # Reshape to (order, channels, channels)
    A = A_flat.T.reshape(n_channels, order, n_channels)
    A = np.transpose(A, (1, 0, 2))  # (order, channels, channels)
    
    return A


def _compute_pdc_at_freq(A: np.ndarray, freq: float, sfreq: float) -> np.ndarray:
    """Compute PDC matrix at a specific frequency."""
    order, n_channels, _ = A.shape
    
    # Compute A(f) = I - sum_p(A_p * exp(-2*pi*i*f*p/sfreq))
    A_f = np.eye(n_channels, dtype=complex)
    
    for p in range(order):
        phase = -2j * np.pi * freq * (p + 1) / sfreq
        A_f -= A[p] * np.exp(phase)
    
    # PDC = |A_ij(f)| / sqrt(sum_k |A_kj(f)|^2)
    pdc = np.zeros((n_channels, n_channels))
    
    for j in range(n_channels):
        col = A_f[:, j]
        norm = np.sqrt(np.sum(np.abs(col) ** 2))
        if norm > 0:
            pdc[:, j] = np.abs(col) / norm
    
    return pdc


def compute_granger_causality(source: np.ndarray,
                               target: np.ndarray,
                               max_lag: int = 10) -> Dict[str, float]:
    """
    Compute Granger Causality from source to target.
    
    Tests whether source helps predict target beyond target's own past.
    
    Args:
        source: Source time series
        target: Target time series
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary with GC statistics
    """
    n = len(target)
    
    if n <= max_lag + 1:
        return {'gc_value': 0, 'gc_significant': False}
    
    # Restricted model: predict target from its own past
    X_r = np.column_stack([target[max_lag-i:-i if i > 0 else None] 
                           for i in range(1, max_lag+1)])
    y = target[max_lag:]
    
    try:
        beta_r, residuals_r, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
        rss_r = np.sum((y - X_r @ beta_r) ** 2)
    except:
        return {'gc_value': 0, 'gc_significant': False}
    
    # Unrestricted model: add source's past
    X_u = np.column_stack([
        X_r,
        *[source[max_lag-i:-i if i > 0 else None] for i in range(1, max_lag+1)]
    ])
    
    try:
        beta_u, residuals_u, _, _ = np.linalg.lstsq(X_u, y, rcond=None)
        rss_u = np.sum((y - X_u @ beta_u) ** 2)
    except:
        return {'gc_value': 0, 'gc_significant': False}
    
    # Granger causality statistic
    if rss_u > 0:
        gc = np.log(rss_r / rss_u)
    else:
        gc = 0
    
    return {
        'gc_value': gc,
        'gc_ratio': rss_r / rss_u if rss_u > 0 else 1,
        'gc_significant': gc > 0.1  # Simple threshold
    }


def extract_directed_connectivity_features(data: np.ndarray,
                                            sfreq: float = SAMPLING_RATE) -> Dict:
    """
    Extract all directed connectivity features.
    
    Args:
        data: EEG data (channels x time)
        sfreq: Sampling frequency
        
    Returns:
        Dictionary of directed connectivity features
    """
    features = {}
    
    # Transfer Entropy matrix
    te_matrix = compute_transfer_entropy_matrix(data, k=2, l=2)
    
    features['te_mean'] = np.mean(te_matrix)
    features['te_std'] = np.std(te_matrix)
    features['te_asymmetry'] = np.mean(np.abs(te_matrix - te_matrix.T))
    
    # Compute inflow/outflow per region
    features['te_outflow'] = np.mean(np.sum(te_matrix, axis=1))
    features['te_inflow'] = np.mean(np.sum(te_matrix, axis=0))
    
    # PDC (simplified - just alpha band)
    try:
        pdc = compute_partial_directed_coherence(data, order=5, sfreq=sfreq)
        if 'alpha_pdc' in pdc:
            alpha_pdc = pdc['alpha_pdc']
            features['pdc_alpha_mean'] = np.mean(alpha_pdc)
            features['pdc_alpha_asymmetry'] = np.mean(np.abs(alpha_pdc - alpha_pdc.T))
    except Exception:
        features['pdc_alpha_mean'] = 0
        features['pdc_alpha_asymmetry'] = 0
    
    return features
