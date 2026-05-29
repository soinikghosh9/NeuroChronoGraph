"""
Cross-Frequency Coupling Analysis Module.

This module provides functions for computing cross-frequency coupling,
including Phase-Amplitude Coupling (PAC) which is a key biomarker
for AD and FTD differentiation.
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..config.config import SAMPLING_RATE


def bandpass_filter(signal: np.ndarray,
                    low_freq: float,
                    high_freq: float,
                    sfreq: float = SAMPLING_RATE,
                    order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        signal: Input signal
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        sfreq: Sampling frequency
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyq = sfreq / 2
    low = low_freq / nyq
    high = high_freq / nyq
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def compute_pac(signal: np.ndarray,
                phase_freq: Tuple[float, float] = (4, 8),
                amp_freq: Tuple[float, float] = (30, 80),
                sfreq: float = SAMPLING_RATE,
                n_bins: int = 18,
                method: str = 'mi') -> float:
    """
    Compute Phase-Amplitude Coupling using Modulation Index.
    
    PAC measures how the phase of a low-frequency oscillation
    modulates the amplitude of a high-frequency oscillation.
    
    Args:
        signal: Time series signal (1D array)
        phase_freq: Frequency band for phase extraction (low, high)
        amp_freq: Frequency band for amplitude extraction (low, high)
        sfreq: Sampling frequency
        n_bins: Number of phase bins
        method: 'mi' (Modulation Index) or 'plv' (Phase Locking Value)
        
    Returns:
        PAC value (Modulation Index)
    """
    # Filter signals
    phase_signal = bandpass_filter(signal, phase_freq[0], phase_freq[1], sfreq)
    amp_signal = bandpass_filter(signal, amp_freq[0], amp_freq[1], sfreq)
    
    # Extract phase and amplitude using Hilbert transform
    phase = np.angle(hilbert(phase_signal))
    amplitude = np.abs(hilbert(amp_signal))
    
    if method == 'mi':
        # Modulation Index (Tort et al., 2010)
        # Bin phase and compute mean amplitude per bin
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        
        for i in range(n_bins):
            idx = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(idx) > 0:
                mean_amp[i] = np.mean(amplitude[idx])
            else:
                mean_amp[i] = 0
        
        # Normalize to get probability distribution
        if np.sum(mean_amp) > 0:
            p = mean_amp / np.sum(mean_amp)
        else:
            return 0.0
        
        # Compute KL divergence from uniform distribution
        # H(p) = -sum(p * log(p))
        p_nonzero = p[p > 0]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero))
        
        # Maximum entropy (uniform)
        h_max = np.log(n_bins)
        
        # Modulation Index
        mi = (h_max - entropy) / h_max
        
        return mi
    
    elif method == 'plv':
        # Mean Vector Length approach
        composite = amplitude * np.exp(1j * phase)
        plv = np.abs(np.mean(composite)) / np.mean(amplitude)
        return plv
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_pac_matrix(data: np.ndarray,
                       phase_freqs: List[Tuple[float, float]] = None,
                       amp_freqs: List[Tuple[float, float]] = None,
                       sfreq: float = SAMPLING_RATE) -> np.ndarray:
    """
    Compute PAC comodulogram (matrix of PAC values).
    
    Args:
        data: Signal data (1D for single channel, 2D for multiple)
        phase_freqs: List of (low, high) for phase bands
        amp_freqs: List of (low, high) for amplitude bands
        sfreq: Sampling frequency
        
    Returns:
        PAC matrix (n_phase_bands x n_amp_bands)
    """
    if phase_freqs is None:
        phase_freqs = [(2, 4), (4, 8), (8, 12)]  # Delta, Theta, Alpha
    
    if amp_freqs is None:
        amp_freqs = [(30, 50), (50, 80), (80, 120)]  # Low, Mid, High Gamma
    
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    n_channels = data.shape[0]
    n_phase = len(phase_freqs)
    n_amp = len(amp_freqs)
    
    pac_matrix = np.zeros((n_phase, n_amp))
    
    for i, pf in enumerate(phase_freqs):
        for j, af in enumerate(amp_freqs):
            # Average PAC across channels
            pac_values = []
            for ch in range(n_channels):
                try:
                    pac = compute_pac(data[ch], pf, af, sfreq)
                    pac_values.append(pac)
                except Exception:
                    pac_values.append(0.0)
            
            pac_matrix[i, j] = np.mean(pac_values)
    
    return pac_matrix


def compute_theta_gamma_pac(data: np.ndarray,
                            sfreq: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Compute Theta-Gamma Phase-Amplitude Coupling.
    
    This is a key biomarker for AD - theta-gamma coupling is
    impaired in Alzheimer's disease patients.
    
    Args:
        data: EEG data (channels x time) or (time,)
        sfreq: Sampling frequency
        
    Returns:
        Dictionary with PAC values
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    n_channels = data.shape[0]
    
    # Theta phase (4-8 Hz) -> Low Gamma amplitude (30-50 Hz)
    theta_low_gamma = []
    # Theta phase (4-8 Hz) -> High Gamma amplitude (50-80 Hz)
    theta_high_gamma = []
    
    for ch in range(n_channels):
        try:
            pac_low = compute_pac(data[ch], (4, 8), (30, 50), sfreq)
            pac_high = compute_pac(data[ch], (4, 8), (50, 80), sfreq)
            theta_low_gamma.append(pac_low)
            theta_high_gamma.append(pac_high)
        except Exception:
            theta_low_gamma.append(0.0)
            theta_high_gamma.append(0.0)
    
    return {
        'theta_low_gamma_pac': np.mean(theta_low_gamma),
        'theta_high_gamma_pac': np.mean(theta_high_gamma),
        'theta_gamma_pac_mean': np.mean(theta_low_gamma + theta_high_gamma),
        'theta_low_gamma_pac_std': np.std(theta_low_gamma),
        'theta_high_gamma_pac_std': np.std(theta_high_gamma),
    }


def compute_alpha_beta_pac(data: np.ndarray,
                           sfreq: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Compute Alpha-Beta Phase-Amplitude Coupling.
    
    Args:
        data: EEG data (channels x time) or (time,)
        sfreq: Sampling frequency
        
    Returns:
        Dictionary with PAC values
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    n_channels = data.shape[0]
    
    # Alpha phase (8-13 Hz) -> Beta amplitude (13-30 Hz)
    alpha_beta = []
    
    for ch in range(n_channels):
        try:
            pac = compute_pac(data[ch], (8, 13), (15, 30), sfreq)
            alpha_beta.append(pac)
        except Exception:
            alpha_beta.append(0.0)
    
    return {
        'alpha_beta_pac': np.mean(alpha_beta),
        'alpha_beta_pac_std': np.std(alpha_beta),
    }


def extract_pac_features(data: np.ndarray,
                         sfreq: float = SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract all Phase-Amplitude Coupling features.
    
    Args:
        data: EEG data (channels x time)
        sfreq: Sampling frequency
        
    Returns:
        Dictionary of PAC features
    """
    features = {}
    
    # Theta-Gamma coupling (key for AD)
    tg_pac = compute_theta_gamma_pac(data, sfreq)
    features.update(tg_pac)
    
    # Alpha-Beta coupling
    ab_pac = compute_alpha_beta_pac(data, sfreq)
    features.update(ab_pac)
    
    # Delta-Beta coupling
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    delta_beta = []
    for ch in range(data.shape[0]):
        try:
            pac = compute_pac(data[ch], (1, 4), (15, 30), sfreq)
            delta_beta.append(pac)
        except Exception:
            delta_beta.append(0.0)
    
    features['delta_beta_pac'] = np.mean(delta_beta)
    
    return features


def compute_pac_from_epochs(epochs,
                            phase_band: Tuple[float, float] = (4, 8),
                            amp_band: Tuple[float, float] = (30, 80)) -> np.ndarray:
    """
    Compute PAC from MNE Epochs object.
    
    Args:
        epochs: MNE Epochs object
        phase_band: Phase frequency band
        amp_band: Amplitude frequency band
        
    Returns:
        PAC values per epoch and channel (n_epochs, n_channels)
    """
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']
    
    n_epochs, n_channels, n_times = data.shape
    pac_values = np.zeros((n_epochs, n_channels))
    
    for ep in range(n_epochs):
        for ch in range(n_channels):
            try:
                pac_values[ep, ch] = compute_pac(
                    data[ep, ch], phase_band, amp_band, sfreq
                )
            except Exception:
                pac_values[ep, ch] = 0.0
    
    return pac_values
