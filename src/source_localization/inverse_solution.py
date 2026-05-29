"""
Source Localization Module using eLORETA.

This module implements EEG source localization using the eLORETA method
with the fsaverage template brain for subjects without individual MRIs.
"""

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

from ..config.config import SOURCE_CONFIG, SAMPLING_RATE


class SourceLocalizer:
    """
    EEG Source Localization using eLORETA with fsaverage template.
    
    This class provides methods for computing source estimates from
    sensor-level EEG data using the eLORETA inverse method.
    """
    
    def __init__(self, 
                 subjects_dir: Optional[Path] = None,
                 verbose: bool = True):
        """
        Initialize the source localizer.
        
        Args:
            subjects_dir: Path to FreeSurfer subjects directory.
                         If None, uses MNE default.
            verbose: Whether to print progress information.
        """
        self.verbose = verbose
        
        # Fetch fsaverage if needed
        self._setup_fsaverage()
        
        if subjects_dir is None:
            self.subjects_dir = Path(mne.get_config('SUBJECTS_DIR'))
        else:
            self.subjects_dir = Path(subjects_dir)
        
        # Initialize components
        self.src = None
        self.bem = None
        self.fwd = None
        self.inv = None
        self.labels = None
        
    def _setup_fsaverage(self):
        """Download and setup fsaverage template if not present."""
        if self.verbose:
            print("Setting up fsaverage template...")
        
        # This will download fsaverage if not present
        fs_dir = fetch_fsaverage(verbose=self.verbose)
        
        # Set subjects_dir in MNE config
        subjects_dir = Path(fs_dir).parent
        mne.set_config('SUBJECTS_DIR', str(subjects_dir))
        
        if self.verbose:
            print(f"fsaverage located at: {fs_dir}")
    
    def setup_source_space(self, 
                           spacing: str = None) -> mne.SourceSpaces:
        """
        Setup the source space on fsaverage cortical surface.
        
        Args:
            spacing: Source space resolution ('ico4', 'ico5', etc.)
                    Lower numbers = coarser, fewer sources
                    
        Returns:
            MNE SourceSpaces object
        """
        if spacing is None:
            spacing = SOURCE_CONFIG['spacing']
        
        if self.verbose:
            print(f"Creating source space with spacing={spacing}...")
        
        self.src = mne.setup_source_space(
            subject='fsaverage',
            spacing=spacing,
            subjects_dir=self.subjects_dir,
            add_dist=False,
            verbose=False
        )
        
        if self.verbose:
            n_sources = sum(s['nuse'] for s in self.src)
            print(f"Source space created with {n_sources} sources")
        
        return self.src
    
    def setup_forward_model(self, 
                            raw: mne.io.Raw) -> mne.Forward:
        """
        Create the forward model (lead field matrix).
        
        Args:
            raw: MNE Raw object with channel info
            
        Returns:
            MNE Forward object
        """
        if self.src is None:
            self.setup_source_space()
        
        if self.verbose:
            print("Setting up forward model...")
        
        # Load fsaverage BEM solution
        bem_path = self.subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
        
        if bem_path.exists():
            self.bem = mne.read_bem_solution(bem_path, verbose=False)
        else:
            # Create BEM if not exists
            if self.verbose:
                print("BEM solution not found, creating...")
            model = mne.make_bem_model(
                subject='fsaverage',
                subjects_dir=self.subjects_dir,
                conductivity=(0.3, 0.006, 0.3),
                verbose=False
            )
            self.bem = mne.make_bem_solution(model, verbose=False)
        
        # Create forward solution
        # Use 'fsaverage' as trans for standard montage alignment
        try:
            self.fwd = mne.make_forward_solution(
                raw.info,
                trans='fsaverage',
                src=self.src,
                bem=self.bem,
                eeg=True,
                mindist=5.0,
                n_jobs=-1,
                verbose=False
            )
        except Exception as e:
            # Try with identity transform if fsaverage trans fails
            warnings.warn(f"Using identity transform: {e}")
            self.fwd = mne.make_forward_solution(
                raw.info,
                trans=None,
                src=self.src,
                bem=self.bem,
                eeg=True,
                mindist=5.0,
                n_jobs=-1,
                verbose=False
            )
        
        if self.verbose:
            print(f"Forward solution created: {self.fwd['nsource']} sources")
        
        return self.fwd
    
    def compute_inverse_operator(self,
                                  epochs: mne.Epochs,
                                  fwd: Optional[mne.Forward] = None) -> mne.minimum_norm.InverseOperator:
        """
        Compute the inverse operator for source estimation.
        
        Args:
            epochs: MNE Epochs object (for noise covariance estimation)
            fwd: Forward solution. If None, uses stored forward.
            
        Returns:
            MNE InverseOperator
        """
        if fwd is None:
            if self.fwd is None:
                raise ValueError("Forward solution not computed. Call setup_forward_model first.")
            fwd = self.fwd
        
        if self.verbose:
            print("Computing inverse operator...")
        
        # Compute noise covariance from the data
        # For resting-state, use the full data
        noise_cov = mne.compute_covariance(
            epochs,
            tmin=0,
            tmax=None,
            method='empirical',
            verbose=False
        )
        
        # Create inverse operator
        self.inv = make_inverse_operator(
            epochs.info,
            fwd,
            noise_cov,
            loose=SOURCE_CONFIG['loose'],
            depth=SOURCE_CONFIG['depth'],
            verbose=False
        )
        
        if self.verbose:
            print("Inverse operator computed")
        
        return self.inv
    
    def apply_inverse(self,
                      epochs: mne.Epochs,
                      method: str = None) -> List[mne.SourceEstimate]:
        """
        Apply inverse solution to get source estimates.
        
        Args:
            epochs: MNE Epochs object
            method: Inverse method ('eLORETA', 'sLORETA', 'MNE', 'dSPM')
            
        Returns:
            List of MNE SourceEstimate objects (one per epoch)
        """
        if self.inv is None:
            raise ValueError("Inverse operator not computed. Call compute_inverse_operator first.")
        
        if method is None:
            method = SOURCE_CONFIG['method']
        
        snr = SOURCE_CONFIG['snr']
        lambda2 = 1.0 / snr ** 2
        
        if self.verbose:
            print(f"Applying {method} inverse solution...")
        
        stcs = apply_inverse_epochs(
            epochs,
            self.inv,
            lambda2=lambda2,
            method=method,
            pick_ori='normal',
            verbose=False
        )
        
        if self.verbose:
            print(f"Computed {len(stcs)} source estimates")
        
        return stcs
    
    def localize_epochs(self,
                        epochs: mne.Epochs,
                        method: str = None) -> List[mne.SourceEstimate]:
        """
        Complete pipeline to compute source estimates from epochs.
        
        This is a convenience method that runs the full pipeline:
        1. Setup forward model (if needed)
        2. Compute inverse operator
        3. Apply inverse solution
        
        Args:
            epochs: MNE Epochs object
            method: Inverse method
            
        Returns:
            List of SourceEstimate objects
        """
        # Setup forward if needed
        if self.fwd is None:
            self.setup_forward_model(epochs)
        
        # Compute inverse operator
        self.compute_inverse_operator(epochs)
        
        # Apply inverse
        stcs = self.apply_inverse(epochs, method=method)
        
        return stcs


def create_forward_for_raw(raw: mne.io.Raw,
                           subjects_dir: Optional[Path] = None) -> mne.Forward:
    """
    Convenience function to create forward model for raw data.
    
    Args:
        raw: MNE Raw object
        subjects_dir: FreeSurfer subjects directory
        
    Returns:
        MNE Forward object
    """
    localizer = SourceLocalizer(subjects_dir=subjects_dir)
    localizer.setup_source_space()
    fwd = localizer.setup_forward_model(raw)
    
    return fwd
