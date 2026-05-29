"""
Spectral Feature Extraction Module.

This module provides functions for extracting spectral features from EEG data,
including Power Spectral Density, band powers, and spectral ratios.
"""

import numpy as np
import mne
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..config.config import FREQUENCY_BANDS, SAMPLING_RATE, FEATURE_CONFIG


def compute_psd(epochs: mne.Epochs,
                fmin: float = 0.5,
                fmax: float = 45.0,
                method: str = 'welch',
                n_fft: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density for epochs.
    
    Args:
        epochs: MNE Epochs object
        fmin: Minimum frequency
        fmax: Maximum frequency
        method: PSD estimation method ('welch', 'multitaper')
        n_fft: FFT length (None for auto)
        
    Returns:
        (psd, freqs) tuple where psd has shape (n_epochs, n_channels, n_freqs)
    """
    if n_fft is None:
        n_fft = FEATURE_CONFIG.get('psd_n_fft', 1024)
    
    # Compute PSD
    spectrum = epochs.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        verbose=False
    )
    
    psd, freqs = spectrum.get_data(return_freqs=True)
    
    return psd, freqs


def compute_band_powers(psd: np.ndarray,
                        freqs: np.ndarray,
                        bands: Dict[str, Tuple[float, float]] = None,
                        relative: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute power in standard frequency bands.
    
    Args:
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
        freqs: Frequency array
        bands: Dictionary of band name to (fmin, fmax) tuples
        relative: Whether to compute relative (normalized) power
        
    Returns:
        Dictionary mapping band names to power arrays (n_epochs, n_channels)
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    
    band_powers = {}
    
    # Compute total power for relative calculation
    if relative:
        total_power = np.sum(psd, axis=-1, keepdims=True)
        total_power = np.where(total_power == 0, 1, total_power)  # Avoid division by zero
    
    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        # Sum power in band
        band_power = np.sum(psd[..., freq_mask], axis=-1)
        
        if relative:
            band_power = band_power / total_power.squeeze()
        
        band_powers[band_name] = band_power
    
    return band_powers


def compute_spectral_ratios(band_powers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute spectral ratio biomarkers.
    
    Args:
        band_powers: Dictionary of band powers
        
    Returns:
        Dictionary of spectral ratios
    """
    ratios = {}
    
    # Theta/Alpha ratio (commonly elevated in AD)
    if 'theta' in band_powers and 'alpha' in band_powers:
        alpha = np.where(band_powers['alpha'] == 0, 1e-10, band_powers['alpha'])
        ratios['theta_alpha'] = band_powers['theta'] / alpha
    
    # Delta/Alpha ratio
    if 'delta' in band_powers and 'alpha' in band_powers:
        alpha = np.where(band_powers['alpha'] == 0, 1e-10, band_powers['alpha'])
        ratios['delta_alpha'] = band_powers['delta'] / alpha
    
    # (Theta + Delta) / (Alpha + Beta) ratio
    if all(b in band_powers for b in ['theta', 'delta', 'alpha', 'beta']):
        slow = band_powers['theta'] + band_powers['delta']
        fast = band_powers['alpha'] + band_powers['beta']
        fast = np.where(fast == 0, 1e-10, fast)
        ratios['slow_fast'] = slow / fast
    
    # Alpha/Beta ratio
    if 'alpha' in band_powers and 'beta' in band_powers:
        beta = np.where(band_powers['beta'] == 0, 1e-10, band_powers['beta'])
        ratios['alpha_beta'] = band_powers['alpha'] / beta
    
    return ratios


def compute_spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """
    Compute spectral entropy for each channel.
    
    Spectral entropy measures the "flatness" of the power spectrum.
    Higher values indicate more uniform power distribution.
    
    Args:
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
        
    Returns:
        Spectral entropy array of shape (n_epochs, n_channels)
    """
    # Normalize PSD to probability distribution
    psd_sum = np.sum(psd, axis=-1, keepdims=True)
    psd_sum = np.where(psd_sum == 0, 1, psd_sum)
    psd_norm = psd / psd_sum
    
    # Avoid log(0)
    psd_norm = np.where(psd_norm == 0, 1e-10, psd_norm)
    
    # Compute Shannon entropy
    n_freqs = psd.shape[-1]
    entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=-1)
    
    # Normalize by max entropy
    max_entropy = np.log2(n_freqs)
    spectral_entropy = entropy / max_entropy
    
    return spectral_entropy


def compute_alpha_peak_frequency(psd: np.ndarray,
                                  freqs: np.ndarray,
                                  alpha_range: Tuple[float, float] = (8, 13)) -> np.ndarray:
    """
    Find the peak frequency in the alpha band.
    
    Reduced alpha peak frequency is a biomarker for AD.
    
    Args:
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
        freqs: Frequency array
        alpha_range: Alpha frequency range
        
    Returns:
        Peak frequency array of shape (n_epochs, n_channels)
    """
    # Find alpha band indices
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[..., alpha_mask]
    
    # Find peak frequency
    peak_idx = np.argmax(alpha_psd, axis=-1)
    peak_freq = alpha_freqs[peak_idx]
    
    return peak_freq


def compute_spectral_slope(psd: np.ndarray,
                           freqs: np.ndarray,
                           freq_range: Tuple[float, float] = (2, 40)) -> np.ndarray:
    """
    Compute the 1/f spectral slope.
    
    The spectral slope reflects the balance between excitation and inhibition.
    Flatter slopes may indicate synaptic dysfunction.
    
    Args:
        psd: PSD array of shape (n_epochs, n_channels, n_freqs)
        freqs: Frequency array
        freq_range: Frequency range for slope calculation
        
    Returns:
        Spectral slope array of shape (n_epochs, n_channels)
    """
    # Select frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    selected_freqs = freqs[freq_mask]
    selected_psd = psd[..., freq_mask]
    
    # Log-log transform
    log_freqs = np.log10(selected_freqs)
    log_psd = np.log10(selected_psd + 1e-10)
    
    # Fit linear regression for each epoch and channel
    n_epochs, n_channels, _ = log_psd.shape
    slopes = np.zeros((n_epochs, n_channels))
    
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Linear fit
            coeffs = np.polyfit(log_freqs, log_psd[epoch, ch], 1)
            slopes[epoch, ch] = coeffs[0]  # Slope
    
    return slopes


def extract_spectral_features(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """
    Extract all spectral features from epochs.
    
    Args:
        epochs: MNE Epochs object
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    # Compute PSD
    psd, freqs = compute_psd(epochs)
    
    # Band powers (absolute and relative)
    abs_powers = compute_band_powers(psd, freqs, relative=False)
    rel_powers = compute_band_powers(psd, freqs, relative=True)
    
    for band in FREQUENCY_BANDS:
        features[f'{band}_power_abs'] = abs_powers[band]
        features[f'{band}_power_rel'] = rel_powers[band]
    
    # Spectral ratios
    ratios = compute_spectral_ratios(rel_powers)
    for ratio_name, ratio_values in ratios.items():
        features[f'ratio_{ratio_name}'] = ratio_values
    
    # Spectral entropy
    features['spectral_entropy'] = compute_spectral_entropy(psd)
    
    # Alpha peak frequency
    features['alpha_peak_freq'] = compute_alpha_peak_frequency(psd, freqs)
    
    # Spectral slope
    features['spectral_slope'] = compute_spectral_slope(psd, freqs)
    
    return features


def extract_spectral_features_array(data: np.ndarray,
                                     sfreq: float = SAMPLING_RATE,
                                     ch_names: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Extract spectral features from numpy array.
    
    Args:
        data: EEG data of shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        ch_names: Channel names
        
    Returns:
        Dictionary of spectral features
    """
    from ..data.preprocessor import create_epochs_array
    
    epochs = create_epochs_array(data, sfreq=sfreq, ch_names=ch_names)
    return extract_spectral_features(epochs)
