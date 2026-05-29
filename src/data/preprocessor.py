"""
EEG Preprocessing module.

This module provides preprocessing functions for EEG data including
filtering, artifact removal, re-referencing, and epoching.
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Optional, List, Tuple, Union
import warnings

from ..config.config import PREPROCESSING, SAMPLING_RATE


def apply_preprocessing_pipeline(raw: mne.io.Raw,
                                  apply_filter: bool = True,
                                  apply_notch: bool = True,
                                  apply_reference: bool = True,
                                  apply_csd: bool = False) -> mne.io.Raw:
    """
    Apply complete preprocessing pipeline to raw EEG data.
    
    Args:
        raw: MNE Raw object
        apply_filter: Whether to apply bandpass filter
        apply_notch: Whether to apply notch filter
        apply_reference: Whether to apply re-referencing
        apply_csd: Whether to apply Current Source Density (Surface Laplacian)
        
    Returns:
        Preprocessed MNE Raw object
    """
    raw = raw.copy()
    
    # Robustly handle channel types
    # Check if 'eeg' channels are present
    current_types = raw.get_channel_types()
    if 'eeg' not in current_types:
        # Try to infer based on standard 10-20 names
        from ..config.config import CHANNEL_NAMES
        
        # Normalize names for matching
        std_names_lower = [c.lower() for c in CHANNEL_NAMES]
        curr_channels = raw.ch_names
        
        type_map = {}
        for ch in curr_channels:
            # Simple heuristic: if channel name looks like a standard channel, mark it EEG
            if ch.lower() in std_names_lower or ch in CHANNEL_NAMES:
                type_map[ch] = 'eeg'
                
        if type_map:
             warnings.warn(f"Setting types for {len(type_map)} channels to 'eeg' based on name match.")
             raw.set_channel_types(type_map)
        else:
             # Fallback: Treat ALL channels as EEG if we can't match names but expected EEG data
             warnings.warn("No 'eeg' channels found and names don't match standard. FORCING ALL to 'eeg'.")
             raw.set_channel_types({ch: 'eeg' for ch in curr_channels})
             
    # Double check
    if 'eeg' not in raw.get_channel_types():
         # If still not present (e.g. set_channel_types failed?), force again blindly
         warnings.warn("Force-setting all channels to 'eeg' (Last Resort).")
         try:
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
         except Exception as e:
            print(f"DEBUG: Could not force set types: {e}")
                 
    # Pick only EEG channels
    # Use pick_types or pick matches
    try:
        raw.pick('eeg', exclude='bads')
    except ValueError as e:
        # If still fails, try picking by name if explicit mapping failed
        warnings.warn(f"Pick 'eeg' failed: {e}. Attempting numeric pick...")
        # This shouldn't happen if we fixed types above.
        raise e
    
    # Apply bandpass filter
    if apply_filter:
        raw = raw.filter(
            l_freq=PREPROCESSING['l_freq'],
            h_freq=PREPROCESSING['h_freq'],
            fir_design='firwin',
            verbose=False
        )
        # Check for numeric stability
        if np.isnan(raw.get_data()).any() or np.isinf(raw.get_data()).any():
             # Warning only, data cleaning happens in loaders usually, but filters can introduce instability
             warnings.warn("Filters introduced NaNs/Infs! clamping...")
             # Cannot easily modify Raw data in place without private access or set_data
             data = raw.get_data()
             data = np.nan_to_num(data, posinf=0, neginf=0, nan=0)
             raw._data = data

    
    # Apply notch filter for power line noise
    if apply_notch:
        raw = raw.notch_filter(
            freqs=PREPROCESSING['notch_freq'],
            verbose=False
        )
    
    # Re-reference
    if apply_reference:
        if PREPROCESSING['reference'] == 'average':
            raw = raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj()
        else:
            raw = raw.set_eeg_reference(PREPROCESSING['reference'], verbose=False)
    
    # Apply Current Source Density (Surface Laplacian)
    if apply_csd:
        raw = apply_surface_laplacian(raw)
        
    # Final Safety Check
    data = raw.get_data()
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, posinf=0, neginf=0, nan=0)
        raw._data = data
        
    return raw


def apply_surface_laplacian(raw: mne.io.Raw,
                            stiffness: int = 4,
                            lambda2: float = 1e-5) -> mne.io.Raw:
    """
    Apply Surface Laplacian (Current Source Density) transformation.
    
    This reduces the effects of volume conduction and increases
    spatial resolution of the EEG signal.
    
    Args:
        raw: MNE Raw object with montage set
        stiffness: Stiffness parameter for spline interpolation
        lambda2: Regularization parameter
        
    Returns:
        CSD-transformed Raw object
    """
    try:
        # Compute CSD
        raw_csd = mne.preprocessing.compute_current_source_density(
            raw,
            stiffness=stiffness,
            lambda2=lambda2,
            verbose=False
        )
        
        # Check CSD output stability
        data = raw_csd.get_data()
        if np.isnan(data).any() or np.isinf(data).any():
             warnings.warn("CSD produced NaNs/Infs! Clamping.")
             data = np.nan_to_num(data, posinf=0, neginf=0, nan=0)
             raw_csd._data = data
             
        return raw_csd
    except Exception as e:
        warnings.warn(f"Could not apply CSD: {e}. Returning original data.")
        return raw


def create_epochs(raw: mne.io.Raw,
                  duration: Optional[float] = None,
                  overlap: Optional[float] = None,
                  reject: Optional[dict] = None,
                  flat: Optional[dict] = None) -> mne.Epochs:
    """
    Create fixed-length epochs from continuous EEG data.
    
    Args:
        raw: MNE Raw object
        duration: Epoch duration in seconds. If None, uses config.
        overlap: Overlap ratio (0.5 = 50%). If None, uses config.
        reject: Rejection criteria dict. If None, uses config.
        flat: Flat channel criteria dict.
        
    Returns:
        MNE Epochs object
    """
    if duration is None:
        duration = PREPROCESSING['epoch_duration']
    if overlap is None:
        overlap = PREPROCESSING['epoch_overlap']
    
    # Sanitize Data (NaN/Inf check)
    # MNE drops epochs with NaNs even if reject=None
    if np.isnan(raw.get_data()).any() or np.isinf(raw.get_data()).any():
        warnings.warn("create_epochs: NaNs/Infs found in raw data! Replacing with zeros.")
        data = raw.get_data()
        data = np.nan_to_num(data, posinf=0, neginf=0, nan=0)
        raw._data = data

    # Debug: Check data range
    dmin, dmax = np.min(raw.get_data()), np.max(raw.get_data())
    # print(f"DEBUG: create_epochs data range: {dmin:.2e} to {dmax:.2e}")

    # Create events for fixed-length epochs
    events = mne.make_fixed_length_events(
        raw,
        duration=duration,
        overlap=duration * overlap,
        id=1
    )
    
    if len(events) == 0:
        warnings.warn("create_epochs: No events could be generated (data too short?)")
    
    # Set up rejection criteria
    picks = mne.pick_types(raw.info, eeg=True, csd=True, exclude='bads')
    eeg_channels = [raw.ch_names[i] for i in picks]
    
    if reject is None:
        if len(eeg_channels) > 0:
            if 'csd' in raw.get_channel_types():
                 reject = None
            else:
                 reject = {'eeg': PREPROCESSING['reject_threshold']}
        else:
            reject = None
    
    # Create epochs with graduated rejection thresholds
    # Try strict -> relaxed -> no rejection (last resort)
    rejection_thresholds = [
        reject,                    # Original threshold (200µV)
        {'eeg': 400e-6} if reject else None,  # Relaxed threshold (400µV)
        None                       # No rejection (last resort)
    ]

    epochs = None
    for i, curr_reject in enumerate(rejection_thresholds):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*All epochs were dropped.*',
                                       category=RuntimeWarning)
                epochs = mne.Epochs(
                    raw,
                    events,
                    tmin=0,
                    tmax=duration,
                    baseline=None,
                    reject=curr_reject,
                    flat=flat if i == 0 else None,  # Only use flat on first try
                    preload=True,
                    verbose=False
                )

            # Check if we have enough epochs (at least 20% of events)
            if len(epochs) >= max(1, len(events) * 0.2):
                break  # Success - use these epochs
            elif i < len(rejection_thresholds) - 1:
                # Too few epochs, try next threshold
                continue

        except Exception as e:
            if i == len(rejection_thresholds) - 1:
                warnings.warn(f"Epoch creation failed completely ({e})")
                # Create minimal epochs array as fallback
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*All epochs were dropped.*',
                                           category=RuntimeWarning)
                    epochs = mne.Epochs(
                        raw,
                        events,
                        tmin=0,
                        tmax=duration,
                        baseline=None,
                        reject=None,
                        flat=None,
                        preload=True,
                        verbose=False
                    )
            continue
    
    return epochs


def create_epochs_array(data: np.ndarray,
                        sfreq: float = SAMPLING_RATE,
                        ch_names: Optional[List[str]] = None,
                        ch_types: str = 'eeg') -> mne.EpochsArray:
    """
    Create MNE EpochsArray from numpy array.
    
    Args:
        data: EEG data array of shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        ch_names: List of channel names
        ch_types: Channel type(s)
        
    Returns:
        MNE EpochsArray object
    """
    n_epochs, n_channels, n_times = data.shape
    
    # Create info object
    if ch_names is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
    
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    
    # Create epochs array
    epochs = mne.EpochsArray(data, info, verbose=False)
    
    return epochs


def segment_continuous_data(raw: mne.io.Raw,
                            window_size: float = 4.0,
                            step_size: float = 2.0) -> List[np.ndarray]:
    """
    Segment continuous EEG data into overlapping windows.
    
    Args:
        raw: MNE Raw object
        window_size: Window duration in seconds
        step_size: Step size in seconds
        
    Returns:
        List of numpy arrays, each of shape (n_channels, window_samples)
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    
    segments = []
    n_samples = data.shape[1]
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)
    
    return segments


def check_data_quality(raw: mne.io.Raw) -> dict:
    """
    Check data quality metrics for a raw EEG recording.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        Dictionary with quality metrics
    """
    data = raw.get_data()
    
    # Compute metrics
    quality = {
        'n_channels': data.shape[0],
        'n_samples': data.shape[1],
        'duration_sec': data.shape[1] / raw.info['sfreq'],
        'sfreq': raw.info['sfreq'],
        'mean_amplitude': np.mean(np.abs(data)),
        'std_amplitude': np.std(data),
        'max_amplitude': np.max(np.abs(data)),
        'n_bad_channels': len(raw.info['bads']),
    }
    
    # Check for flat channels
    channel_stds = np.std(data, axis=1)
    quality['n_flat_channels'] = np.sum(channel_stds < 1e-10)
    
    # Check for high amplitude artifacts
    threshold = PREPROCESSING['reject_threshold']
    quality['n_high_amplitude_points'] = np.sum(np.abs(data) > threshold)
    quality['pct_high_amplitude'] = quality['n_high_amplitude_points'] / data.size * 100
    
    return quality


def interpolate_bad_channels(raw: mne.io.Raw,
                              bad_channels: Optional[List[str]] = None) -> mne.io.Raw:
    """
    Interpolate bad channels using spherical spline interpolation.
    
    Args:
        raw: MNE Raw object with montage set
        bad_channels: List of bad channel names. If None, uses raw.info['bads']
        
    Returns:
        Raw object with interpolated channels
    """
    raw = raw.copy()
    
    if bad_channels is not None:
        raw.info['bads'] = bad_channels
    
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    return raw


def run_ica_artifact_removal(raw: mne.io.Raw,
                              n_components: Optional[int] = None,
                              method: str = 'fastica',
                              random_state: int = 42) -> Tuple[mne.io.Raw, ICA]:
    """
    Run ICA for artifact removal (EOG, EMG).
    
    Note: The derivatives data already has ICA applied, so this is
    primarily for raw data processing.
    
    Args:
        raw: MNE Raw object
        n_components: Number of ICA components. If None, uses n_channels - 1
        method: ICA method ('fastica', 'infomax', 'picard')
        random_state: Random state for reproducibility
        
    Returns:
        (cleaned_raw, ica) tuple
    """
    raw = raw.copy()
    
    # Fit ICA
    n_channels = len(raw.ch_names)
    if n_components is None:
        n_components = n_channels - 1
    
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        verbose=False
    )
    
    ica.fit(raw, verbose=False)
    
    # Automatically detect EOG-related components
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices
    except:
        # No EOG channel available
        pass
    
    # Apply ICA
    raw_clean = ica.apply(raw.copy(), verbose=False)
    
    return raw_clean, ica
