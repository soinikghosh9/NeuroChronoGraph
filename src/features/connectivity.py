"""
Connectivity Feature Extraction Module.

This module provides functions for computing functional connectivity
measures between EEG channels/sources, including wPLI, PLI, and coherence.
"""

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..config.config import FREQUENCY_BANDS, CONNECTIVITY_BANDS, FEATURE_CONFIG, SAMPLING_RATE


def compute_connectivity_matrix(epochs: mne.Epochs,
                                 method: str = 'wpli',
                                 fmin: float = None,
                                 fmax: float = None,
                                 faverage: bool = True) -> np.ndarray:
    """
    Compute connectivity matrix between channels.
    
    Args:
        epochs: MNE Epochs object
        method: Connectivity method ('wpli', 'pli', 'coh', 'plv', 'imcoh')
        fmin: Minimum frequency
        fmax: Maximum frequency
        faverage: Whether to average over frequencies
        
    Returns:
        Connectivity matrix of shape (n_channels, n_channels) or
        (n_freqs, n_channels, n_channels) if faverage=False
    """
    # Compute spectral connectivity
    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        fmin=fmin,
        fmax=fmax,
        faverage=faverage,
        mode='multitaper',
        mt_bandwidth=2.0,
        n_jobs=-1,
        verbose=False
    )
    
    # Get connectivity data in dense matrix form
    con_data = con.get_data(output='dense')
    
    # Remove singleton dimensions
    con_data = np.squeeze(con_data)
    
    return con_data


def compute_multiband_connectivity(epochs: mne.Epochs,
                                    method: str = 'wpli',
                                    bands: Dict[str, Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """
    Compute connectivity matrices for multiple frequency bands.
    
    Args:
        epochs: MNE Epochs object
        method: Connectivity method
        bands: Dictionary of band names to (fmin, fmax) tuples
        
    Returns:
        Dictionary mapping band names to connectivity matrices
    """
    if bands is None:
        bands = {b: FREQUENCY_BANDS[b] for b in CONNECTIVITY_BANDS}
    
    connectivity = {}
    
    for band_name, (fmin, fmax) in bands.items():
        try:
            con_matrix = compute_connectivity_matrix(
                epochs,
                method=method,
                fmin=fmin,
                fmax=fmax,
                faverage=True
            )
            connectivity[band_name] = con_matrix
        except Exception as e:
            warnings.warn(f"Error computing {band_name} connectivity: {e}")
            # Return zero matrix
            n_channels = len(epochs.ch_names)
            connectivity[band_name] = np.zeros((n_channels, n_channels))
    
    return connectivity


def compute_dynamic_connectivity(data: np.ndarray,
                                  sfreq: float = SAMPLING_RATE,
                                  window_size: float = None,
                                  step_size: float = None,
                                  method: str = 'wpli',
                                  bands: Dict[str, Tuple[float, float]] = None,
                                  ch_names: List[str] = None) -> List[Dict[str, np.ndarray]]:
    """
    Compute dynamic functional connectivity using sliding windows.
    
    Args:
        data: EEG data of shape (n_channels, n_times)
        sfreq: Sampling frequency
        window_size: Window duration in seconds
        step_size: Step size in seconds
        method: Connectivity method
        bands: Frequency bands dictionary
        ch_names: Channel names
        
    Returns:
        List of connectivity dictionaries, one per time window
    """
    if window_size is None:
        window_size = FEATURE_CONFIG.get('dynamic_window_size', 4.0)
    if step_size is None:
        step_size = FEATURE_CONFIG.get('dynamic_step_size', 2.0)
    if bands is None:
        bands = {b: FREQUENCY_BANDS[b] for b in CONNECTIVITY_BANDS}
    
    n_channels, n_times = data.shape
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    
    if ch_names is None:
        ch_names = [f'CH{i+1}' for i in range(n_channels)]
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    dynamic_conn = []
    
    for start in range(0, n_times - window_samples + 1, step_samples):
        end = start + window_samples
        
        # Extract window
        window_data = data[:, start:end]
        
        # Create epochs object (single epoch)
        window_data = window_data[np.newaxis, :, :]  # Add epoch dimension
        epochs = mne.EpochsArray(window_data, info, verbose=False)
        
        # Compute connectivity for this window
        try:
            conn = compute_multiband_connectivity(epochs, method=method, bands=bands)
        except:
            # If fails, use zeros
            conn = {b: np.zeros((n_channels, n_channels)) for b in bands}
        
        dynamic_conn.append(conn)
    
    return dynamic_conn


def compute_connectivity_from_epochs(epochs: mne.Epochs,
                                      method: str = None) -> Dict[str, np.ndarray]:
    """
    Compute multi-band connectivity from epochs.
    
    This is the main connectivity extraction function.
    
    Args:
        epochs: MNE Epochs object
        method: Connectivity method (default: from config)
        
    Returns:
        Dictionary with connectivity matrices for each band
    """
    if method is None:
        method = FEATURE_CONFIG.get('connectivity_method', 'wpli')
    
    return compute_multiband_connectivity(epochs, method=method)


def connectivity_to_edge_features(connectivity: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert connectivity matrices to edge index and edge attributes for GNN.
    
    Args:
        connectivity: Dictionary of connectivity matrices
        
    Returns:
        (edge_index, edge_attr) tuple
        - edge_index: (2, n_edges) array
        - edge_attr: (n_edges, n_bands) array
    """
    # Get first matrix to determine size
    first_band = list(connectivity.keys())[0]
    n_nodes = connectivity[first_band].shape[0]
    
    # Stack connectivity values
    n_bands = len(connectivity)
    stacked = np.stack([connectivity[b] for b in sorted(connectivity.keys())], axis=-1)
    
    # Create edge index (fully connected graph)
    edge_index = []
    edge_attr = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # No self-loops
                edge_index.append([i, j])
                edge_attr.append(stacked[i, j])
    
    edge_index = np.array(edge_index).T  # Shape: (2, n_edges)
    edge_attr = np.array(edge_attr)       # Shape: (n_edges, n_bands)
    
    return edge_index, edge_attr


def threshold_connectivity(connectivity: np.ndarray,
                           threshold: float = None,
                           keep_ratio: float = None) -> np.ndarray:
    """
    Threshold connectivity matrix to create sparse network.
    
    Args:
        connectivity: Connectivity matrix
        threshold: Absolute threshold value
        keep_ratio: Alternatively, keep top ratio of connections
        
    Returns:
        Thresholded connectivity matrix
    """
    if threshold is None and keep_ratio is None:
        threshold = FEATURE_CONFIG.get('graph_threshold', 0.3)
    
    conn = connectivity.copy()
    
    if keep_ratio is not None:
        # Keep top N% of connections
        flat = conn.flatten()
        sorted_vals = np.sort(flat)[::-1]
        n_keep = int(len(sorted_vals) * keep_ratio)
        threshold = sorted_vals[n_keep] if n_keep < len(sorted_vals) else 0
    
    # Apply threshold
    conn[conn < threshold] = 0
    
    return conn


def binarize_connectivity(connectivity: np.ndarray,
                          threshold: float = None) -> np.ndarray:
    """
    Binarize connectivity matrix.
    
    Args:
        connectivity: Connectivity matrix
        threshold: Threshold value
        
    Returns:
        Binary connectivity matrix
    """
    if threshold is None:
        threshold = FEATURE_CONFIG.get('graph_threshold', 0.3)
    
    return (connectivity > threshold).astype(float)
