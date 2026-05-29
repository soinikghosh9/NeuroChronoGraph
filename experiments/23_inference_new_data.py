"""
Clinical Inference Script for NeuroChronoGraph.

This script enables inference on new EEG datasets for dementia classification.
Supports:
- Single subject prediction
- Batch prediction on new datasets
- Confidence scoring and uncertainty estimation
- Clinical reporting format

Usage:
    python 23_inference_new_data.py --input /path/to/preprocessed_data.pkl
    python 23_inference_new_data.py --input_dir /path/to/preprocessed_folder
    python 23_inference_new_data.py --input /path/to/data.pkl --output report.json
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional, Union
import pickle

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import DEVICE
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2


# =============== CONFIGURATION ===============
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "hierarchical_model_fold0.pt"
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
CLASS_DESCRIPTIONS = {
    'AD': "Alzheimer's Disease - progressive memory loss and cognitive decline",
    'FTD': "Frontotemporal Dementia - behavioral/language variant dementia",
    'CN': "Cognitively Normal - no significant cognitive impairment",
    'MCI': "Mild Cognitive Impairment - intermediate stage, may convert to dementia"
}


class HierarchicalDementiaClassifier:
    """
    Clinical inference wrapper for hierarchical dementia classification.

    Provides:
    - Easy loading of pretrained models
    - Single and batch prediction
    - Confidence scoring with uncertainty estimation
    - Clinical reporting in human-readable format
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = None):
        """
        Initialize the classifier.

        Args:
            checkpoint_path: Path to model checkpoint. Uses best fold if None.
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or DEVICE
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the pretrained model from checkpoint."""
        print(f"Loading model from: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Please train the model first using 18_train_hierarchical.py"
            )

        # Load weights first so we can detect stream-fusion checkpoint format
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        has_streams = any(k.startswith('stream_fusion.') for k in state_dict.keys())
        cfg = {'n_classes': 3, 'hidden_dim': 128, 'dropout': 0.0}
        if has_streams:
            cfg.update({
                'feature_streams': ['spectral', 'connectivity', 'complexity', 'microstate'],
                'stream_dim': 64,
            })
        self.model = create_neuro_chrono_graph_v2(cfg).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def _prepare_input(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare input data for model inference.

        Args:
            data: Dictionary containing preprocessed EEG data with keys:
                - 'epochs_data' or 'epochs': EEG epochs array [n_epochs, n_channels, n_times]
                - 'sfreq': Sampling frequency
                - 'band_features' (optional): Precomputed band power features
                - 'age' (optional): Subject age
                - 'sex' (optional): Subject sex (0=female, 1=male)
                - 'mmse' (optional): MMSE score

        Returns:
            Dictionary of tensors ready for model input
        """
        # Extract epochs
        if 'epochs_data' in data:
            epochs = data['epochs_data']
        elif 'epochs' in data:
            epochs = data['epochs'].get_data() if hasattr(data['epochs'], 'get_data') else data['epochs']
        else:
            raise ValueError("Input must contain 'epochs_data' or 'epochs' key")

        sfreq = data.get('sfreq', 256)

        # Ensure correct shape [batch, channels, time]
        if epochs.ndim == 3:
            # Average epochs or use first epoch
            x = np.mean(epochs, axis=0, keepdims=True)  # [1, channels, time]
        else:
            x = epochs[np.newaxis, ...]

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Prepare band features
        band_features = data.get('band_features', None)
        if band_features is None:
            # Compute simple band features from epochs
            band_features = self._compute_band_features(epochs, sfreq)

        band_tensors = {}
        for band_name, band_data in band_features.items():
            if isinstance(band_data, np.ndarray):
                bd = torch.tensor(band_data, dtype=torch.float32)
            else:
                bd = band_data.clone()

            # Ensure batch dimension
            if bd.ndim == 2:
                bd = bd.unsqueeze(0)
            band_tensors[band_name] = bd.to(self.device)

        # Clinical metadata (use defaults if not provided)
        age = data.get('age', 70) / 100.0  # Normalize
        sex = data.get('sex', 0.5)  # Default neutral
        mmse = data.get('mmse', 25) / 30.0  # Normalize

        metadata = torch.tensor([[age, sex, mmse]], dtype=torch.float32).to(self.device)
        clinical = {
            'age': metadata[:, 0],
            'sex': metadata[:, 1],
            'mmse': metadata[:, 2]
        }

        return {
            'x': x,
            'band_features': band_tensors,
            'clinical': clinical,
            'metadata': metadata
        }

    def _compute_band_features(self, epochs: np.ndarray, sfreq: float) -> Dict[str, np.ndarray]:
        """Compute band power features from epochs."""
        from scipy.signal import welch

        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # Average across epochs
        if epochs.ndim == 3:
            signal = epochs.mean(axis=0)  # [channels, time]
        else:
            signal = epochs

        n_ch, n_times = signal.shape
        nperseg = min(256, n_times)

        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, axis=-1)

        band_features = {}
        for band_name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = psd[:, mask].mean(axis=-1)  # [channels]
            # Create time series approximation [channels, time_approx]
            n_time_approx = min(100, n_times // 4)
            band_features[band_name] = np.tile(band_power[:, np.newaxis], (1, n_time_approx))

        return band_features

    def predict(self, data: Union[Dict, str, Path]) -> Dict:
        """
        Predict dementia class for a single subject.

        Args:
            data: Either:
                - Dictionary with preprocessed data
                - Path to .pkl file with preprocessed data

        Returns:
            Dictionary with prediction results:
                - 'predicted_class': Most likely class name
                - 'predicted_class_idx': Class index (0=AD, 1=FTD, 2=CN, 3=MCI)
                - 'probabilities': Dict of class probabilities
                - 'confidence': Confidence score (max probability)
                - 'hierarchical': Stage-wise predictions
                - 'clinical_interpretation': Human-readable interpretation
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            with open(data, 'rb') as f:
                data = pickle.load(f)

        # Prepare input
        inputs = self._prepare_input(data)

        # Run inference
        with torch.no_grad():
            outputs = self.model(
                inputs['x'],
                band_features=inputs['band_features'],
                clinical_data=inputs['clinical']
            )

        # Extract probabilities
        probs_screen = outputs['probs_screen'].cpu().numpy()[0]  # [2]: CN, Impaired
        probs_stage = outputs['probs_stage'].cpu().numpy()[0]    # [2]: MCI, Dementia
        probs_subtype = outputs['probs_subtype'].cpu().numpy()[0]  # [2]: AD, FTD

        # Compute flat 4-class probabilities using geometric mean chain rule
        eps = 1e-8
        p_cn = probs_screen[0]
        p_impaired = probs_screen[1]
        p_mci_given_imp = probs_stage[0]
        p_dem_given_imp = probs_stage[1]
        p_ad_given_dem = probs_subtype[0]
        p_ftd_given_dem = probs_subtype[1]

        # Depth-1: CN
        P_CN = p_cn

        # Depth-2: MCI (Impaired -> MCI)
        P_MCI = (p_impaired * p_mci_given_imp + eps) ** 0.5

        # Depth-3: AD, FTD (Impaired -> Dementia -> AD/FTD)
        P_AD = (p_impaired * p_dem_given_imp * p_ad_given_dem + eps) ** (1/3)
        P_FTD = (p_impaired * p_dem_given_imp * p_ftd_given_dem + eps) ** (1/3)

        # Normalize
        total = P_AD + P_FTD + P_CN + P_MCI
        probs_flat = np.array([P_AD, P_FTD, P_CN, P_MCI]) / total

        predicted_idx = int(np.argmax(probs_flat))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs_flat[predicted_idx])

        # Clinical interpretation
        if confidence > 0.7:
            confidence_level = "HIGH"
        elif confidence > 0.5:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        interpretation = self._generate_interpretation(
            predicted_class, confidence, confidence_level, probs_flat,
            probs_screen, probs_stage, probs_subtype
        )

        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_idx,
            'probabilities': {name: float(p) for name, p in zip(CLASS_NAMES, probs_flat)},
            'confidence': confidence,
            'confidence_level': confidence_level,
            'hierarchical': {
                'screening': {'CN': float(probs_screen[0]), 'Impaired': float(probs_screen[1])},
                'staging': {'MCI': float(probs_stage[0]), 'Dementia': float(probs_stage[1])},
                'subtyping': {'AD': float(probs_subtype[0]), 'FTD': float(probs_subtype[1])}
            },
            'clinical_interpretation': interpretation
        }

    def _generate_interpretation(self, predicted_class, confidence, confidence_level,
                                  probs_flat, probs_screen, probs_stage, probs_subtype) -> str:
        """Generate human-readable clinical interpretation."""
        lines = [
            "=" * 60,
            "CLINICAL INTERPRETATION REPORT",
            "=" * 60,
            "",
            f"PRIMARY CLASSIFICATION: {predicted_class}",
            f"Confidence: {confidence:.1%} ({confidence_level})",
            "",
            f"Description: {CLASS_DESCRIPTIONS[predicted_class]}",
            "",
            "HIERARCHICAL DECISION PATH:",
            f"  1. Screening (CN vs Impaired): {'Impaired' if probs_screen[1] > 0.5 else 'CN'} "
            f"({max(probs_screen):.1%})",
        ]

        if probs_screen[1] > 0.5:
            lines.append(
                f"  2. Staging (MCI vs Dementia): {'Dementia' if probs_stage[1] > 0.5 else 'MCI'} "
                f"({max(probs_stage):.1%})"
            )
            if probs_stage[1] > 0.5:
                lines.append(
                    f"  3. Subtyping (AD vs FTD): {'AD' if probs_subtype[0] > 0.5 else 'FTD'} "
                    f"({max(probs_subtype):.1%})"
                )

        lines.extend([
            "",
            "PROBABILITY DISTRIBUTION:",
            f"  AD:  {probs_flat[0]:.1%}",
            f"  FTD: {probs_flat[1]:.1%}",
            f"  CN:  {probs_flat[2]:.1%}",
            f"  MCI: {probs_flat[3]:.1%}",
            "",
            "CLINICAL NOTE:",
        ])

        if confidence_level == "LOW":
            lines.append("  This prediction has low confidence. Consider additional clinical")
            lines.append("  evaluation and neuroimaging before making diagnostic decisions.")
        elif predicted_class in ['AD', 'FTD']:
            lines.append("  This EEG pattern suggests dementia-related changes.")
            lines.append("  Recommend comprehensive neuropsychological evaluation and")
            lines.append("  neuroimaging (MRI/PET) for confirmation.")
        elif predicted_class == 'MCI':
            lines.append("  EEG shows patterns consistent with mild cognitive impairment.")
            lines.append("  Recommend follow-up assessment in 6-12 months to monitor")
            lines.append("  potential progression to dementia.")
        else:
            lines.append("  EEG patterns appear within normal limits for cognitive function.")
            lines.append("  Continue routine monitoring as clinically indicated.")

        lines.append("")
        lines.append("=" * 60)
        lines.append("DISCLAIMER: This is a research tool and should not replace")
        lines.append("clinical judgment or standard diagnostic procedures.")
        lines.append("=" * 60)

        return "\n".join(lines)

    def predict_batch(self, data_paths: List[Path], progress: bool = True) -> List[Dict]:
        """
        Predict dementia class for multiple subjects.

        Args:
            data_paths: List of paths to preprocessed data files
            progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(data_paths, desc="Processing subjects") if progress else data_paths

        for path in iterator:
            try:
                result = self.predict(path)
                result['subject_id'] = path.stem.replace('_preprocessed', '')
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'subject_id': path.stem.replace('_preprocessed', ''),
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)

        return results

    def generate_batch_report(self, results: List[Dict], output_path: Optional[Path] = None) -> str:
        """
        Generate summary report for batch predictions.

        Args:
            results: List of prediction results from predict_batch()
            output_path: Path to save JSON report (optional)

        Returns:
            Summary text
        """
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']

        # Count predictions
        class_counts = {cls: 0 for cls in CLASS_NAMES}
        confidence_levels = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0}

        for r in successful:
            class_counts[r['predicted_class']] += 1
            confidence_levels[r['confidence_level']] += 1

        lines = [
            "=" * 60,
            "BATCH PREDICTION SUMMARY",
            "=" * 60,
            "",
            f"Total subjects: {len(results)}",
            f"Successful: {len(successful)}",
            f"Failed: {len(failed)}",
            "",
            "PREDICTION DISTRIBUTION:",
            f"  AD:  {class_counts['AD']} ({class_counts['AD']/max(len(successful),1)*100:.1f}%)",
            f"  FTD: {class_counts['FTD']} ({class_counts['FTD']/max(len(successful),1)*100:.1f}%)",
            f"  CN:  {class_counts['CN']} ({class_counts['CN']/max(len(successful),1)*100:.1f}%)",
            f"  MCI: {class_counts['MCI']} ({class_counts['MCI']/max(len(successful),1)*100:.1f}%)",
            "",
            "CONFIDENCE DISTRIBUTION:",
            f"  HIGH:     {confidence_levels['HIGH']}",
            f"  MODERATE: {confidence_levels['MODERATE']}",
            f"  LOW:      {confidence_levels['LOW']}",
            "",
        ]

        if failed:
            lines.append("FAILED SUBJECTS:")
            for r in failed:
                lines.append(f"  - {r['subject_id']}: {r.get('error', 'Unknown error')}")

        summary = "\n".join(lines)

        # Save JSON if requested
        if output_path:
            report = {
                'summary': {
                    'total': len(results),
                    'successful': len(successful),
                    'failed': len(failed),
                    'class_counts': class_counts,
                    'confidence_levels': confidence_levels
                },
                'predictions': results
            }
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_path}")

        return summary


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(
        description='Run inference on new EEG data for dementia classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single subject prediction
  python 23_inference_new_data.py --input data/subject01_preprocessed.pkl

  # Batch prediction on directory
  python 23_inference_new_data.py --input_dir data/preprocessed/ --output results.json

  # Use specific checkpoint
  python 23_inference_new_data.py --input data/subject01.pkl --checkpoint model_fold2.pt
        """
    )

    parser.add_argument('--input', type=Path, help='Path to single preprocessed .pkl file')
    parser.add_argument('--input_dir', type=Path, help='Directory of preprocessed .pkl files')
    parser.add_argument('--output', type=Path, help='Output path for JSON report')
    parser.add_argument('--checkpoint', type=Path, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir must be provided")

    # Initialize classifier
    classifier = HierarchicalDementiaClassifier(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Run inference
    if args.input:
        # Single subject
        print(f"\nProcessing: {args.input}")
        result = classifier.predict(args.input)
        print(result['clinical_interpretation'])

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")

    else:
        # Batch processing
        pkl_files = list(args.input_dir.glob("*_preprocessed.pkl"))
        if not pkl_files:
            pkl_files = list(args.input_dir.glob("*.pkl"))

        if not pkl_files:
            print(f"No .pkl files found in {args.input_dir}")
            return

        print(f"\nFound {len(pkl_files)} files to process")
        results = classifier.predict_batch(pkl_files)

        # Generate report
        summary = classifier.generate_batch_report(
            results,
            output_path=args.output or args.input_dir / "batch_predictions.json"
        )
        print(summary)


if __name__ == "__main__":
    main()
