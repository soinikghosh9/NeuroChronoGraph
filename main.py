"""
NeuroChronoGraph: Hierarchical EEG Classification
Main Entry Point

Usage:
    python main.py --reset
    python main.py --steps 01 18
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Steps to run in order
# Dictionary mapping step ID to script path
STEPS = {
    "01": PROJECT_ROOT / "experiments" / "01_preprocess_all.py",
    "21": PROJECT_ROOT / "experiments" / "21_visualize_raw_segments.py",
    "18": PROJECT_ROOT / "experiments" / "18_train_hierarchical.py",
    "20": PROJECT_ROOT / "experiments" / "20_neuroscientific_analysis.py",
    "16": PROJECT_ROOT / "experiments" / "16_clinical_source_analysis.py",
    "17": PROJECT_ROOT / "experiments" / "17_connectivity_visualization.py",
    "22": PROJECT_ROOT / "experiments" / "22_statistical_reporting.py",
    "23": PROJECT_ROOT / "experiments" / "visualize_microstates.py",
    "09": PROJECT_ROOT / "experiments" / "09_generate_publication_plots.py"
}

# Step 21 (raw visualization) added after preprocessing for data inspection
ORDERED_STEPS = ["01", "21", "18", "20", "16", "17", "23", "22", "09"]

def reset_workflow():
    """Clear all outputs and caches for a fresh start."""
    print("\n[!] RESETTING WORKFLOW: Deleting all outputs and caches...")
    
    # 1. Delete Output Directories
    dirs_to_clear = [
        PROJECT_ROOT / "outputs" / "preprocessed",
        PROJECT_ROOT / "outputs" / "checkpoints",
        PROJECT_ROOT / "outputs" / "results",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "outputs" / "source_estimates",
        PROJECT_ROOT / "outputs" / "features",
        PROJECT_ROOT / "outputs" / "models"
    ]
    
    for d in dirs_to_clear:
        if d.exists():
            print(f"  - Deleting output dir: {d.name}...")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"    Error deleting {d}: {e}")
                
    # 2. Delete __pycache__ recursively
    print("  - Cleaning __pycache__ directories...")
    cleaned_count = 0
    for p in PROJECT_ROOT.rglob("__pycache__"):
        if p.is_dir():
            try:
                shutil.rmtree(p)
                cleaned_count += 1
            except Exception as e:
                print(f"    Error deleting {p}: {e}")
    print(f"    Cleaned {cleaned_count} cache directories.")
            
    print("[!] Reset complete.\n")

def run_step(step_id):
    """Run a single step in a separate process."""
    script_path = STEPS[step_id]
    print(f"\n" + "="*50)
    print(f"Running Step {step_id}: {script_path.name}")
    print("="*50 + "\n")
    
    if not script_path.exists():
        print(f"CRITICAL ERROR: Script not found: {script_path}")
        sys.exit(1)
        
    try:
        # Use subprocess to ensure clean process state (no stale modules)
        # Check=True will raise CalledProcessError if exit code != 0
        subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"\n>>> Step {step_id} COMPLETED SUCCESSFULLY.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Step {step_id} failed with exit code {e.returncode}")
        print("Check the output above for details.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n[!] Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR executing Step {step_id}: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NeuroChronoGraph Pipeline Orchestrator")
    parser.add_argument("--reset", action="store_true", help="Delete all existing outputs before running")
    parser.add_argument("--steps", nargs="+", choices=STEPS.keys(), help="Run specific steps only (e.g. 01 18)")
    
    args = parser.parse_args()
    
    print("==================================================")
    print("   NeuroChronoGraph: Full Research Pipeline")
    print("==================================================")
    
    if args.reset:
        reset_workflow()
        
    # Determine steps to run
    if args.steps:
        # Run specific steps in order provided (or should we sort them?)
        # Let's run them in the order user provided, trusting they know dependency.
        steps_to_run = args.steps
    else:
        # Run ALL steps in default order
        steps_to_run = ORDERED_STEPS
        
    for step_id in steps_to_run:
        run_step(step_id)
            
    print("\n==================================================")
    print("   Pipeline Finished Successfully")
    print("   Results: outputs/results/")
    print("   Figures: outputs/figures/")
    print("==================================================")

if __name__ == "__main__":
    main()
