# NeuroChronoGraph

## Hierarchical Graph Neural Network for EEG-Based Dementia Differential Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Multi-Dataset](https://img.shields.io/badge/Datasets-5%20Sources-orange.svg)](https://openneuro.org/datasets/ds004504)

A comprehensive EEG analysis framework for **4-class dementia classification** (AD, FTD, CN, MCI) using a hierarchical three-stage architecture: Screening → Staging → Subtyping.

![Graphical Abstract](z<img width="3609" height="2607" alt="Figure1_graphical_abstract_page-0001" src="https://github.com/user-attachments/assets/782befe7-fe2b-4b3e-81fa-5e5beed66809" />
)

---

## 📊 Results Summary

| Metric | Development (5-Fold CV) | Hold-Out (Unbiased) |
|--------|------------------------|---------------------|
| **Accuracy** | 81.8% ± 0.9% | 81.2% |
| **Balanced Accuracy** | 83.7% ± 4.0% | 86.2% |
| **Cohen's Kappa** | 0.74 ± 0.05 | 0.74 |
| **Macro F1** | 0.81 ± 0.05 | 0.80 |

### Hierarchical Stage Performance

| Stage | Metric | CV Mean ± Std | Hold-Out |
|-------|--------|---------------|----------|
| **Screening** | Cohen's κ | 0.86 ± 0.03 | 0.87 |
| **Staging** | Balanced Acc | 92.3% ± 6.5% | 92.0% |
| **Subtyping** | Balanced Acc | 88.0% ± 3.5% | 77.1% |

### Class-Specific Performance (Hold-Out)

| Class | Sensitivity | Precision | F1-Score |
|-------|-------------|-----------|----------|
| AD | 62.8% | 91.1% | 0.74 |
| FTD | 91.4% | 70.6% | 0.80 |
| CN | 91.9% | 90.5% | 0.91 |
| MCI | 98.7% | 60.4% | 0.75 |

> **Note:** Development accuracy excludes collapsed folds (fold collapse recovery mechanism). Hold-out provides unbiased generalization estimate on 51 subjects.

---

## 🧠 Overview

NeuroChronoGraph is a brain-inspired AI system designed to analyze resting-state EEG and distinguish between Alzheimer's Disease, Frontotemporal Dementia, and healthy aging. The architecture explicitly mirrors biological brain mechanisms:

| Brain Mechanism | AI Component | Purpose |
|-----------------|--------------|---------|
| Cortical columns | Graph nodes | Local neural activity |
| White matter tracts | Graph edges | Connectivity pathways |
| Hierarchical processing | GNN layers | Progressive abstraction |
| Brain modules | Modular Transformer | Frontal/temporal/parietal/occipital |
| Cross-frequency coupling | Cross-Band Attention | θ-γ, α-β interactions |
| Synaptic plasticity | Adaptive graph learning | Task-relevant connections |
| Prefrontal modulation | Clinical FiLM | Context-dependent processing |

---

## ✨ Key Features

### Multi-Domain Feature Extraction
- **Spectral:** Power spectral density, band powers, alpha peak frequency, spectral slope
- **Complexity:** Sample entropy, multiscale entropy, Lempel-Ziv complexity, DFA
- **Connectivity:** wPLI, PLI (multiband, volume-conduction resistant)
- **Graph Theory:** Efficiency, clustering, modularity, small-world index
- **Microstates:** Duration, coverage, transitions (temporal brain dynamics)

### Brain-Inspired Architecture
- **Adaptive Graph Learning:** Combines prior connectivity with data-driven discovery
- **Cross-Band Attention:** Models frequency interactions (θ-γ, α-β coupling)
- **Modular Brain Transformer:** Processes brain regions hierarchically
- **Clinical Conditioning:** Age and MMSE modulate network representations
- **Uncertainty Estimation:** Evidential outputs provide confidence scores

### Rigorous Validation & Training
- **Hold-Out Test Set:** 51 subjects (10%) isolated before any model development
- **StratifiedGroupKFold:** 5-fold CV with balanced class distribution, no subject leakage
- **Fold Collapse Recovery:** Automatic retry mechanism (up to 3 attempts) when training collapse detected
- **Transparent Reporting:** Both CV and hold-out metrics with collapsed folds excluded

### Clinical Inference Pipeline
- **Easy Inference:** `HierarchicalDementiaClassifier` for single subject or batch prediction
- **Clinical Reports:** Human-readable interpretation with confidence scores
- **Robust to New Data:** Automatic feature computation for new datasets

---

## 📁 Project Structure

```
neurochrongraph/
├── main.py                           # Main entry point
├── src/
│   ├── config/config.py              # Central configuration
│   ├── data/
│   │   ├── dataset_factory.py        # Dataset orchestration
│   │   ├── bids_loader.py            # OpenNeuro loader
│   │   ├── alz_eeg_loader.py         # AlzEEG loader
│   │   └── mendeley_loader.py        # Mendeley loader
│   ├── models/
│   │   ├── v2/                       # NeuroChronoGraph V2
│   │   └── losses.py                 # Hierarchical Loss with Focal Loss
│   └── visualization/                # Plotting utilities
├── experiments/
│   ├── cli.py                        # ★ Interactive CLI Pipeline Runner
│   ├── run_all.py                    # Legacy batch runner
│   ├── 01_preprocess_all.py          # Phase 1: Preprocessing
│   ├── 18_train_hierarchical.py      # Phase 2: Core training & fold recovery
│   ├── 09_generate_publication...    # Phase 3: Publication plots
│   ├── 20_neuroscientific_ana...     # Phase 4: Core neuroscientific analysis
│   ├── 16_clinical_source_ana...     # Phase 4.5: Clinical source analysis
│   ├── 17_connectivity_visual...     # Phase 4.6: Connectome visualization
│   ├── visualize_microstates.py      # Phase 4.7: Microstate topographies
│   ├── 22_statistical_reporting.py   # Phase 5: Statistical tables
│   ├── 25_statistical_analysis.py    # Phase 6: Bootstrap CIs & subject-level
│   ├── 27_visualization_analysis.py  # Phase 7: t-SNE, PR curves
│   ├── 24_baseline_comparisons.py    # Phase 8: SVM, RF, XGBoost baselines
│   ├── 26_cross_dataset_valid...     # Phase 9: LODO generalization
│   ├── 28_feature_analysis.py        # Phase 9.5: Feature ablation & ranking
│   └── 23_inference_new_data.py      # Clinical inference script
├── outputs/
│   ├── results/v3_holdout_results/   # Results JSON
│   ├── figures/                      # Publication plots
│   └── checkpoints/                  # Model checkpoints
└── main.tex                          # Manuscript (LaTeX)
```

---

## 🏥 Hierarchical Classification Framework

To mimic clinically valid diagnostic workflows, NeuroChronoGraph employs a **3-Stage Hierarchical Architecture**:

### 1. Screening (Detection)
*   **Goal**: Distinguish **Cognitively Normal (CN)** subjects from those with any form of impairment.
*   **Classes**: Healthy vs. Impaired (AD + FTD + MCI).
*   **Clinical Relevance**: First-line exclusion of healthy individuals to prioritize resources.

### 2. Staging (Disease Progression)
*   **Goal**: Determine the severity of impairment, distinguishing prodromal stages from established dementia.
*   **Classes**: **Mild Cognitive Impairment (MCI)** vs. Dementia (AD/FTD).
*   **Medical Context**: MCI represents a critical "at-risk" transitional stage where intervention is most effective.
*   **Logic**: Trained specifically on the "Impaired" subset from Stage 1.

### 3. Subtyping (Differential Diagnosis)
*   **Goal**: Differentiate between specific dementia pathologies.
*   **Classes**: **Alzheimer's Disease (AD)** vs. **Frontotemporal Dementia (FTD)**.
*   **Medical Context**: These two conditions require vastly different care strategies (e.g., AChE inhibitors work for AD but may worsen FTD).
*   **Logic**: Trained specifically on the "Dementia" subset from Stage 2.

---

## 📊 Datasets & Cohort

The system integrates five distinct datasets to ensure robustness across recording devices and protocols:

| Dataset | Sample Types | N | Contribution |
| :--- | :--- | :--- | :--- |
| **OpenNeuro ds004504** | AD, FTD, CN | 88 | Primary source for FTD (BIDS format) |
| **OpenNeuro ds006036** | AD, FTD, CN | 88 | Replication cohort for ds004504 |
| **AlzEEG** | AD, CN | ~200 | Large AD/CN cohort for screening |
| **Mendeley Dataset** | AD, MCI, CN | ~150 | **Critical for Staging** (MCI samples) |
| **MCI Dataset** | MCI | 14 | Additional MCI samples |

**Total Cohort:** 509 subjects (266 AD, 46 FTD, 169 CN, 28 MCI)
**Protocol:** 19-Channel EEG (10-20 System), Resting State (Eyes Closed)
**Validation:** 458 development + 51 hold-out subjects

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/soinikghosh9/NeuroChronoGraph.git
cd NeuroChronoGraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
mne>=1.0.0
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=2.0.0
torch-geometric>=2.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pycrostates>=0.5.0
```

---

## 📈 Usage

### Training & Analysis Pipeline

NeuroChronoGraph features a comprehensive **13-phase analytical pipeline** orchestrated by a beautiful, interactive Command-Line Interface (CLI). 

```bash
# Launch the interactive minimalist CLI menu
python experiments/cli.py
```

The CLI provides a menu to run the full pipeline, specific analysis groups, or individual phases:

```bash
# Example non-interactive (CLI flag) usage:
python experiments/cli.py --full                 # Run all 13 phases
python experiments/cli.py --analysis-only        # Skip preprocessing & training
python experiments/cli.py --category neuroscience # Run Phases 4, 4.5, 4.6, 4.7
python experiments/cli.py --status               # View pipeline completion status
```

**Pipeline Phases:**
1. **Preprocessing:** Clean and extract features from 560 subjects
2. **Model Training:** Hierarchical GNN training (5-fold CV + holdout)
3. **Publication Plots:** Confusion matrices, ROC, performance highlights
4. **Neuroscience:** Attention, connectivity, microstates, clinical source
5. **Statistics:** Bootstrapping, DeLong AUCs, subject-level metrics
6. **Benchmarking:** SVM, RF, XGBoost, cross-dataset (LODO) validation
7. **Feature Analysis:** Domain ablation and class-specific discriminators

### Clinical Inference on New Data

```bash
# Single subject prediction
python experiments/23_inference_new_data.py --input path/to/subject_preprocessed.pkl

# Batch prediction on directory
python experiments/23_inference_new_data.py --input_dir path/to/preprocessed/ --output results.json

# Use specific checkpoint
python experiments/23_inference_new_data.py --input data.pkl --checkpoint outputs/checkpoints/hierarchical_model_fold0.pt
```

### Python API

```python
from experiments.inference_new_data import HierarchicalDementiaClassifier

# Initialize classifier
classifier = HierarchicalDementiaClassifier()

# Single prediction
result = classifier.predict("path/to/preprocessed.pkl")
print(result['predicted_class'])        # 'AD', 'FTD', 'CN', or 'MCI'
print(result['confidence'])             # Confidence score (0-1)
print(result['clinical_interpretation']) # Human-readable report

# Batch prediction
results = classifier.predict_batch(list_of_paths)
summary = classifier.generate_batch_report(results)
```

---

## 🔬 Key Biomarkers Discovered

### Alzheimer's Disease
- ↑ Theta/Alpha ratio (global slowing)
- ↓ Alpha peak frequency (< 9 Hz)
- ↓ Posterior connectivity (parietal-occipital wPLI)
- ↓ MS-D coverage (attention network dysfunction)

### Frontotemporal Dementia
- ↑ Frontal theta power
- ↓ Fronto-temporal connectivity
- Altered MS-C dynamics (salience network)
- ↑ Hub disruption index (frontal)

---

## 📄 Documentation

- **[Architecture Diagram](docs/architecture_diagram.md)**: Detailed model architecture with brain-AI correspondences
- **Manuscript**: The full paper is under peer review and is not included in this code repository (see the [Citation](#-citation) section).

---

## 📚 Citation

> **Status:** The manuscript associated with this repository is **currently under peer review at *Biomedical Signal Processing and Control* (Elsevier).** This entry will be updated with the final journal reference and DOI upon acceptance.

```bibtex
@unpublished{ghosh2026neurochronograph,
  title  = {NeuroChronoGraph: A Hierarchical Graph Neural Network for
            EEG-Based Differential Diagnosis of Alzheimer's Disease and
            Frontotemporal Dementia},
  author = {Ghosh, Soinik and Mourya, Anil Kumar and Srivastava, Mona and
            Sharma, Shiru and Sharma, Neeraj},
  year   = {2026},
  note   = {Manuscript under peer review}
}
```

### Dataset Citation

```bibtex
@dataset{miltiadous2023dataset,
  title={A dataset of EEG recordings from: Alzheimer's disease, 
         Frontotemporal dementia and Healthy subjects},
  author={Miltiadous, Andreas and Tzimourta, Katerina D. and others},
  year={2023},
  publisher={OpenNeuro},
  doi={10.18112/openneuro.ds004504.v1.0.5}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: May 29, 2026*
