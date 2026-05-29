"""
ROI Parcellation Module.

This module provides functionality for creating coarse brain region
parcellations suitable for 19-channel EEG source localization.
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from ..config.config import SOURCE_CONFIG, ROI_NAMES_8


# Mapping of 8 ROIs to Desikan-Killiany atlas labels
ROI_TO_DK_LABELS = {
    'L_Frontal': [
        'superiorfrontal-lh', 'rostralmiddlefrontal-lh', 
        'caudalmiddlefrontal-lh', 'lateralorbitofrontal-lh',
        'frontalpole-lh', 'parsopercularis-lh', 
        'parstriangularis-lh', 'parsorbitalis-lh',
        'medialorbitofrontal-lh', 'rostralanteriorcingulate-lh',
        'caudalanteriorcingulate-lh', 'precentral-lh'
    ],
    'R_Frontal': [
        'superiorfrontal-rh', 'rostralmiddlefrontal-rh',
        'caudalmiddlefrontal-rh', 'lateralorbitofrontal-rh',
        'frontalpole-rh', 'parsopercularis-rh',
        'parstriangularis-rh', 'parsorbitalis-rh',
        'medialorbitofrontal-rh', 'rostralanteriorcingulate-rh',
        'caudalanteriorcingulate-rh', 'precentral-rh'
    ],
    'L_Temporal': [
        'superiortemporal-lh', 'middletemporal-lh',
        'inferiortemporal-lh', 'fusiform-lh',
        'entorhinal-lh', 'parahippocampal-lh', 
        'temporalpole-lh', 'transversetemporal-lh',
        'bankssts-lh', 'insula-lh'
    ],
    'R_Temporal': [
        'superiortemporal-rh', 'middletemporal-rh',
        'inferiortemporal-rh', 'fusiform-rh',
        'entorhinal-rh', 'parahippocampal-rh',
        'temporalpole-rh', 'transversetemporal-rh',
        'bankssts-rh', 'insula-rh'
    ],
    'L_Parietal': [
        'superiorparietal-lh', 'inferiorparietal-lh',
        'postcentral-lh', 'precuneus-lh', 
        'supramarginal-lh', 'posteriorcingulate-lh',
        'isthmuscingulate-lh', 'paracentral-lh'
    ],
    'R_Parietal': [
        'superiorparietal-rh', 'inferiorparietal-rh',
        'postcentral-rh', 'precuneus-rh',
        'supramarginal-rh', 'posteriorcingulate-rh',
        'isthmuscingulate-rh', 'paracentral-rh'
    ],
    'L_Occipital': [
        'lateraloccipital-lh', 'cuneus-lh',
        'pericalcarine-lh', 'lingual-lh'
    ],
    'R_Occipital': [
        'lateraloccipital-rh', 'cuneus-rh',
        'pericalcarine-rh', 'lingual-rh'
    ]
}


class ROIParcellation:
    """
    Create and manage coarse brain region parcellations.
    
    This class provides methods for creating 8-ROI parcellations
    from the Desikan-Killiany atlas, suitable for 19-channel EEG.
    """
    
    def __init__(self, 
                 subjects_dir: Optional[Path] = None,
                 verbose: bool = True):
        """
        Initialize the ROI parcellation.
        
        Args:
            subjects_dir: FreeSurfer subjects directory
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        
        if subjects_dir is None:
            subjects_dir = mne.get_config('SUBJECTS_DIR')
        
        self.subjects_dir = Path(subjects_dir) if subjects_dir else None
        self.labels = None
        self.merged_labels = None
        
    def load_dk_labels(self) -> List[mne.Label]:
        """
        Load Desikan-Killiany atlas labels from fsaverage.
        
        Returns:
            List of MNE Label objects
        """
        if self.subjects_dir is None:
            raise ValueError("subjects_dir not set")
        
        if self.verbose:
            print("Loading Desikan-Killiany labels...")
        
        self.labels = mne.read_labels_from_annot(
            'fsaverage',
            parc='aparc',  # Desikan-Killiany
            subjects_dir=self.subjects_dir,
            verbose=False
        )
        
        if self.verbose:
            print(f"Loaded {len(self.labels)} labels")
        
        return self.labels
    
    def create_coarse_parcellation(self) -> Dict[str, mne.Label]:
        """
        Create 8-ROI coarse parcellation by merging DK labels.
        
        Returns:
            Dictionary mapping ROI names to merged Label objects
        """
        if self.labels is None:
            self.load_dk_labels()
        
        if self.verbose:
            print("Creating 8-ROI coarse parcellation...")
        
        self.merged_labels = {}
        
        for roi_name, label_names in ROI_TO_DK_LABELS.items():
            # Find matching labels
            roi_labels = []
            for label in self.labels:
                if label.name in label_names:
                    roi_labels.append(label)
            
            if len(roi_labels) == 0:
                warnings.warn(f"No labels found for {roi_name}")
                continue
            
            # Merge labels
            if len(roi_labels) == 1:
                merged = roi_labels[0].copy()
            else:
                merged = roi_labels[0]
                for label in roi_labels[1:]:
                    try:
                        merged = merged + label
                    except:
                        # Labels might not be combinable
                        pass
            
            merged.name = roi_name
            self.merged_labels[roi_name] = merged
        
        if self.verbose:
            print(f"Created {len(self.merged_labels)} ROI labels")
            for name, label in self.merged_labels.items():
                print(f"  {name}: {len(label.vertices)} vertices")
        
        return self.merged_labels
    
    def extract_label_time_courses(self,
                                    stcs: List[mne.SourceEstimate],
                                    mode: str = 'mean_flip') -> np.ndarray:
        """
        Extract ROI time courses from source estimates.
        
        Args:
            stcs: List of SourceEstimate objects
            mode: Extraction mode ('mean', 'mean_flip', 'pca_flip')
            
        Returns:
            Array of shape (n_epochs, n_rois, n_times)
        """
        if self.merged_labels is None:
            self.create_coarse_parcellation()
        
        labels_list = [self.merged_labels[name] for name in ROI_NAMES_8 
                       if name in self.merged_labels]
        
        if self.verbose:
            print(f"Extracting time courses for {len(labels_list)} ROIs...")
        
        # Extract time courses for each epoch
        all_tcs = []
        
        for stc in stcs:
            epoch_tcs = []
            for label in labels_list:
                try:
                    tc = stc.extract_label_time_course(
                        label,
                        src=None,
                        mode=mode,
                        allow_empty=True
                    )
                    epoch_tcs.append(tc.flatten())
                except:
                    # If extraction fails, use zeros
                    epoch_tcs.append(np.zeros(stc.data.shape[1]))
            
            all_tcs.append(np.array(epoch_tcs))
        
        result = np.array(all_tcs)  # (n_epochs, n_rois, n_times)
        
        if self.verbose:
            print(f"Extracted time courses shape: {result.shape}")
        
        return result
    
    def get_label_names(self) -> List[str]:
        """Get the names of the ROI labels."""
        if self.merged_labels is None:
            return ROI_NAMES_8
        return list(self.merged_labels.keys())


def extract_roi_timecourses_simple(stcs: List[mne.SourceEstimate],
                                    src: mne.SourceSpaces,
                                    subjects_dir: Optional[Path] = None) -> np.ndarray:
    """
    Simplified ROI time course extraction.
    
    This function provides a simpler interface for extracting
    ROI time courses from source estimates.
    
    Args:
        stcs: List of SourceEstimate objects
        src: Source space used for forward model
        subjects_dir: FreeSurfer subjects directory
        
    Returns:
        Array of shape (n_epochs, n_rois, n_times)
    """
    parcellation = ROIParcellation(subjects_dir=subjects_dir, verbose=False)
    parcellation.create_coarse_parcellation()
    
    return parcellation.extract_label_time_courses(stcs)
