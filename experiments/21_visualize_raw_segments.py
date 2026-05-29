"""
Visualize Raw EEG Segments.

Generates a grid of raw EEG plots:
- Rows: Datasets (OpenNeuro, Alz_EEG, Mendeley, etc.)
- Columns: Classes (CN, MCI, AD, FTD)

Purpose: Visual inspection of signal quality and potential artifacts/domain shifts.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import (
    DATA_ROOT, DEVICE, CHANNEL_NAMES, CLASS_MAPPING, SAMPLING_RATE
)
from src.data.dataset_factory import DatasetFactory

def visualize_raw_segments():
    print("Initializing Dataset Factory...")
    factory = DatasetFactory()
    datasets_paths = {
        'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
        'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
        'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
        'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
        'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset"
    }
    
    # Load all available datasets
    for name, path in datasets_paths.items():
        if path.exists():
            print(f"Adding dataset: {name}")
            factory.add_dataset(name, path)
        else:
            print(f"Skipping {name} (Not found)")

    # Create the unified dataset WITHOUT splitting first, 
    # but we need to access the underlying source info.
    # The factory returns a concatenated dataset.
    # We need to peek into the dataset to find samples.
    
    full_dataset, groups, labels = factory.create_torch_datasets(config={'n_times': 2000})
    
    # Create the unified dataset
    # We need to map class indices back to names
    IDX_TO_CLASS = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
    
    samples_found = {} # Key: (DatasetName, ClassName) -> Sample Data (x)
    
    print("Searching for samples...")
    
    # Iterate through the datasets in the ConcatDataset
    dataset_names = list(datasets_paths.keys())
    
    for ds_idx, sub_dataset in enumerate(full_dataset.datasets):
        # Use known name from our list
        if ds_idx < len(dataset_names):
            ds_name = dataset_names[ds_idx]
        else:
            ds_name = f"DS_{ds_idx}"
            
        print(f"Scanning {ds_name}...")
        
        # Optimization: Use metadata labels if available to find indices directly
        # EEGDatasetWrapper exposes .labels list
        if hasattr(sub_dataset, 'labels'):
            ds_labels = np.array(sub_dataset.labels)
            
            # Find one index for each target class
            for class_idx, class_str in IDX_TO_CLASS.items():
                # Find indices where label == class_idx
                matches = np.where(ds_labels == class_idx)[0]
                
                if len(matches) > 0:
                    # Pick the first valid non-NaN sample
                    for match_idx in matches[:10]: # Check first 10 matches
                        try:
                            sample = sub_dataset[match_idx]
                            x = sample['x']
                            
                            if torch.isnan(x).any() or torch.isinf(x).any():
                                continue # Skip corrupt samples
                                
                            key = (ds_name, class_str)
                            samples_found[key] = x
                            break # Found one!
                        except Exception as e:
                            print(f"Error loading {ds_name} idx {match_idx}: {e}")
                            continue
        else:
            # Fallback for datasets without exposed labels (slow scan)
            classes_needed = set(IDX_TO_CLASS.values())
            classes_found_in_ds = set()
            
            for i in range(min(len(sub_dataset), 500)): # Check first 500
                 if len(classes_found_in_ds) == len(classes_needed):
                     break
                 
                 try:
                     sample = sub_dataset[i]
                     x = sample['x']
                     y = int(sample['label'])
                     class_name = IDX_TO_CLASS.get(y, f"Unknown_{y}")
                     
                     if torch.isnan(x).any() or torch.isinf(x).any():
                         continue
                         
                     key = (ds_name, class_name)
                     if key not in samples_found:
                         samples_found[key] = x
                         classes_found_in_ds.add(class_name)
                 except:
                     continue
    
    # Plotting
    print(f"Found {len(samples_found)} unique (Dataset, Class) combinations.")
    
    # Determine grid size
    datasets = sorted(list(set(k[0] for k in samples_found.keys())))
    classes = ['CN', 'MCI', 'AD', 'FTD'] # Fixed order
    
    n_rows = len(datasets)
    n_cols = len(classes)
    
    if n_rows == 0:
        print("No valid datasets or samples found!")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Setup styling
    plt.style.use('seaborn-v0_8-paper')
    
    for r, ds_name in enumerate(datasets):
        for c, class_name in enumerate(classes):
            ax = axes[r, c]
            key = (ds_name, class_name)
            
            if key in samples_found:
                # Get data: [Channels, Time]
                data = samples_found[key].numpy()
                
                # Check bounds and NaNs again just in case
                if np.isnan(data).any() or np.isinf(data).any():
                    ax.text(0.5, 0.5, "Corrupted Data (NaN)", ha='center', va='center', color='red')
                else:
                    # Select channels with highest variance (most active)
                    # This handles sparse datasets (Mendeley with 4 chans) where fixed indices might hit zeros
                    stds = np.std(data, axis=1)
                    # Get indices of top 5 channels with signal
                    active_indices = np.argsort(stds)[::-1]
                    # Filter out purely flat channels if possible, but keep top 5
                    n_ch_to_plot = 5
                    channels_to_plot = active_indices[:n_ch_to_plot]
                    # Sort indices for logical ordering (e.g. 0, 1, 2, 3)
                    channels_to_plot = sorted(channels_to_plot) 
                    offset = 30 
                    
                    time_axis = np.arange(data.shape[1]) / 200.0 
                    
                    for i, ch_idx in enumerate(channels_to_plot):
                        if ch_idx < data.shape[0]:
                            # Normalize for visualization
                            sig = data[ch_idx]
                            
                            # Robust normalization
                            if np.std(sig) > 0:
                                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                            else:
                                sig = sig - np.mean(sig) # Flat line
                            
                            ax.plot(time_axis, sig + (i * offset), lw=0.8, label=f"Ch{ch_idx}")
                
                ax.set_title(f"{ds_name}\n{class_name}")
                ax.set_yticks([])
                if r == n_rows - 1:
                    ax.set_xlabel("Time (s)")
                else:
                    ax.set_xticks([])
                    
            else:
                ax.text(0.5, 0.5, "Not Available", ha='center', va='center', color='gray')
                ax.set_title(f"{ds_name}\n{class_name}")
                ax.axis('off')

    # Add suptitle
    fig.suptitle("Raw EEG Segments by Dataset and Class", fontsize=16)
    
    # Save
    out_dir = PROJECT_ROOT / "outputs" / "figures" / "data_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_eeg_samples_by_dataset_and_class.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    visualize_raw_segments()
