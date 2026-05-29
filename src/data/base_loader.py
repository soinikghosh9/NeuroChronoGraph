"""
Base Data Loader Abstraction.

Standardizes interface for all EEG datasets (BIDS, Text, MAT).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mne
from pathlib import Path

class BaseDataset(ABC):
    """
    Abstract base class for all EEG datasets.
    """
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.subject_ids = []
        
    @abstractmethod
    def get_subject_ids(self) -> List[str]:
        """Return list of all subject IDs in the dataset."""
        pass
        
    @abstractmethod
    def get_subject_info(self, subject_id: str) -> Dict:
        """
        Return metadata for a subject.
        Must return dict with keys: 'subject_id', 'group', 'label', 'age', 'sex', 'mmse'.
        """
        pass
        
    @abstractmethod
    def load_raw(self, subject_id: str) -> mne.io.Raw:
        """
        Load MNE Raw object for a subject.
        Must be standardized to 19 channels (10-20) and 250Hz.
        """
        pass
        
    def summary(self) -> str:
        """Return a basic summary string."""
        return f"{self.__class__.__name__}: {len(self.subject_ids)} subjects."
