"""
EEG Microstate Analysis Module.

This module provides functions for computing EEG microstates,
which represent quasi-stable global brain states.
"""

import numpy as np
import mne
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    from pycrostates.cluster import ModKMeans
    from pycrostates.preprocessing import extract_gfp_peaks, resample
    from pycrostates.metrics import silhouette_score as ms_silhouette
    HAS_PYCROSTATES = True
except ImportError:
    HAS_PYCROSTATES = False
    warnings.warn("pycrostates not available. Microstate analysis will be limited.")

from ..config.config import SAMPLING_RATE


class MicrostateAnalyzer:
    """
    EEG Microstate Analysis.
    
    Microstates are quasi-stable global brain states that represent
    the dominant topography of the EEG at any given time point.
    """
    
    def __init__(self,
                 n_states: int = 4,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize microstate analyzer.
        
        Args:
            n_states: Number of microstate classes (typically 4-7)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.n_states = n_states
        self.random_state = random_state
        self.verbose = verbose
        
        self.cluster_centers = None
        self.labels = None
        self.gfp = None
        
    def fit(self, 
            epochs: mne.Epochs,
            n_runs: int = 10) -> 'MicrostateAnalyzer':
        """
        Fit microstate model to EEG data.
        
        Args:
            epochs: MNE Epochs object
            n_runs: Number of random restarts for k-means
            
        Returns:
            Self
        """
        if not HAS_PYCROSTATES:
            return self._fit_basic(epochs)
        
        if self.verbose:
            print("Fitting microstate model...")
        
        # Extract GFP peaks for clustering
        gfp_peaks = extract_gfp_peaks(epochs)
        
        # Optional: resample to reduce computation
        # ChData objects have a different API - check n_samples or shape
        # Note: Resampling can be tricky with pycrostates API changes, so we skip it
        # The clustering will just take a bit longer but will be more accurate
        
        # Fit modified k-means
        self.model = ModKMeans(
            n_clusters=self.n_states,
            n_init=n_runs,
            random_state=self.random_state
        )
        self.model.fit(gfp_peaks)
        
        self.cluster_centers = self.model.cluster_centers_
        
        if self.verbose:
            print(f"Fitted {self.n_states} microstate classes")
        
        return self
    
    def _fit_basic(self, epochs: mne.Epochs) -> 'MicrostateAnalyzer':
        """Basic microstate fitting without pycrostates."""
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Compute GFP
        gfp = np.std(data, axis=1)  # (n_epochs, n_times)
        
        # Find GFP peaks
        all_peaks = []
        for ep in range(n_epochs):
            peaks = self._find_peaks(gfp[ep])
            peak_maps = data[ep, :, peaks]
            all_peaks.append(peak_maps.T)
        
        peak_maps = np.vstack(all_peaks)  # (n_peaks, n_channels)
        
        # Simple k-means
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_states, n_init=10, 
                    random_state=self.random_state)
        km.fit(peak_maps)
        
        self.cluster_centers = km.cluster_centers_
        
        return self
    
    def _find_peaks(self, gfp: np.ndarray, min_distance: int = 10) -> np.ndarray:
        """Find local maxima in GFP."""
        peaks = []
        for i in range(min_distance, len(gfp) - min_distance):
            if gfp[i] == max(gfp[i-min_distance:i+min_distance+1]):
                peaks.append(i)
        return np.array(peaks)
    
    def segment(self, 
                epochs: mne.Epochs,
                factor: int = 10,
                half_window_size: int = 3) -> np.ndarray:
        """
        Segment EEG into microstates.
        
        Args:
            epochs: MNE Epochs object
            factor: Smoothing factor
            half_window_size: Half window for temporal smoothing
            
        Returns:
            Segmentation labels (n_epochs, n_times)
        """
        if not HAS_PYCROSTATES:
            return self._segment_basic(epochs)
        
        if self.verbose:
            print("Segmenting data into microstates...")
        
        # Use model.predict() instead of segment() function
        segmentation = self.model.predict(
            epochs,
            factor=factor,
            half_window_size=half_window_size
        )
        
        self.labels = segmentation.labels
        self.gfp = getattr(segmentation, 'gfp', None)
        
        return self.labels
    
    def _segment_basic(self, epochs: mne.Epochs) -> np.ndarray:
        """Basic segmentation using correlation."""
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        labels = np.zeros((n_epochs, n_times), dtype=int)
        
        for ep in range(n_epochs):
            for t in range(n_times):
                topo = data[ep, :, t]
                # Find best matching microstate
                correlations = []
                for center in self.cluster_centers:
                    corr = np.abs(np.corrcoef(topo, center)[0, 1])
                    correlations.append(corr)
                labels[ep, t] = np.argmax(correlations)
        
        self.labels = labels
        return labels
    
    def compute_parameters(self) -> Dict[str, np.ndarray]:
        """
        Compute microstate parameters.
        
        Returns:
            Dictionary with:
            - duration: Mean duration of each microstate (ms)
            - coverage: Time coverage of each microstate (%)
            - occurrence: Occurrence rate per second
            - transition_matrix: Transition probabilities
            - gev: Global Explained Variance
        """
        if self.labels is None:
            raise ValueError("Must call segment() first")
        
        n_epochs, n_times = self.labels.shape
        sfreq = SAMPLING_RATE
        
        params = {
            'duration': np.zeros(self.n_states),
            'coverage': np.zeros(self.n_states),
            'occurrence': np.zeros(self.n_states),
        }
        
        # Compute per-state metrics
        for state in range(self.n_states):
            # Coverage
            state_samples = np.sum(self.labels == state)
            total_samples = self.labels.size
            params['coverage'][state] = state_samples / total_samples * 100
            
            # Duration and occurrence
            durations = []
            occurrences = 0
            
            for ep in range(n_epochs):
                # Find runs of this state
                in_state = (self.labels[ep] == state).astype(int)
                changes = np.diff(np.concatenate([[0], in_state, [0]]))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                
                for start, end in zip(starts, ends):
                    durations.append((end - start) / sfreq * 1000)  # ms
                    occurrences += 1
            
            if durations:
                params['duration'][state] = np.mean(durations)
            
            total_time_sec = n_epochs * n_times / sfreq
            params['occurrence'][state] = occurrences / total_time_sec
        
        # Transition matrix
        params['transition_matrix'] = self._compute_transition_matrix()
        
        return params
    
    def _compute_transition_matrix(self) -> np.ndarray:
        """Compute transition probability matrix."""
        n_states = self.n_states
        transitions = np.zeros((n_states, n_states))
        
        for ep in range(self.labels.shape[0]):
            for t in range(self.labels.shape[1] - 1):
                from_state = self.labels[ep, t]
                to_state = self.labels[ep, t + 1]
                if from_state != to_state:  # Only count actual transitions
                    transitions[from_state, to_state] += 1
        
        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transitions / row_sums
        
        return transition_probs


def extract_microstate_features(epochs: mne.Epochs,
                                 n_states: int = 4) -> Dict:
    """
    Extract microstate features from EEG epochs.
    
    Args:
        epochs: MNE Epochs object
        n_states: Number of microstate classes
        
    Returns:
        Dictionary of microstate features
    """
    analyzer = MicrostateAnalyzer(n_states=n_states, verbose=False)
    
    try:
        analyzer.fit(epochs)
        analyzer.segment(epochs)
        params = analyzer.compute_parameters()
        
        features = {
            'duration': params['duration'].tolist(),
            'coverage': params['coverage'].tolist(),
            'occurrence': params['occurrence'].tolist(),
            'transition_matrix': params['transition_matrix'].tolist(),
            'n_states': n_states,
            'cluster_centers': analyzer.cluster_centers.tolist() if analyzer.cluster_centers is not None else None
        }
        
        return features
        
    except Exception as e:
        warnings.warn(f"Microstate analysis failed: {e}")
        return {
            'duration': None,
            'coverage': None,
            'occurrence': None,
            'transition_matrix': None,
            'n_states': n_states,
            'error': str(e)
        }
