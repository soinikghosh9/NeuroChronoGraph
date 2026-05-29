#!/usr/bin/env python
"""
NeuroChronoGraph — Interactive CLI Pipeline Runner

Beautiful terminal UI for running the full pipeline with
phase selection, progress tracking, and status overview.

Usage:
    python experiments/cli.py              # Interactive mode
    python experiments/cli.py --status     # Show pipeline status
    python experiments/cli.py --quick      # Quick figures-only run
"""

import subprocess
import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# ─── colorama for cross-platform ANSI colors ─────────────────────────────────
try:
    from colorama import init as colorama_init, Fore, Back, Style
    colorama_init(autoreset=True)
except ImportError:
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Back = Style = _Dummy()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ─── Pipeline Phases ─────────────────────────────────────────────────────────
# Each phase has a category for sub-menu grouping
PHASES = [
    {
        "id": "1", "script": "experiments/01_preprocess_all.py",
        "name": "Preprocessing",
        "desc": "Preprocess all EEG datasets (560 subjects)",
        "time": "~1-2 hrs", "flag": "preprocess", "cat": "core",
        "extra_args": ["--skip-existing"],
    },
    {
        "id": "2", "script": "experiments/18_train_hierarchical.py",
        "name": "Model Training",
        "desc": "Hierarchical GNN training (5-fold CV + holdout)",
        "time": "~4-8 hrs", "flag": "train", "cat": "core",
        "critical": True,
    },
    {
        "id": "3", "script": "experiments/09_generate_publication_plots.py",
        "name": "Publication Plots",
        "desc": "Confusion matrices, ROC, sensitivity, performance highlights",
        "time": "~2-3 min", "flag": "analysis", "cat": "visualization",
    },
    {
        "id": "4", "script": "experiments/20_neuroscientific_analysis.py",
        "name": "Neuroscientific Analysis",
        "desc": "Connectivity heatmaps, band coupling, attention, explainability",
        "time": "~15-30 min", "flag": "analysis", "cat": "neuroscience",
    },
    {
        "id": "4.5", "script": "experiments/16_clinical_source_analysis.py",
        "name": "Clinical Source Analysis",
        "desc": "Regional power, theta/alpha ratio, AP gradient, disease signatures",
        "time": "~10-20 min", "flag": "analysis", "cat": "neuroscience",
    },
    {
        "id": "4.6", "script": "experiments/17_connectivity_visualization.py",
        "name": "Connectivity Visualization",
        "desc": "Circular connectomes (AD/FTD/CN/MCI), connectivity matrices",
        "time": "~10-20 min", "flag": "analysis", "cat": "neuroscience",
    },
    {
        "id": "4.7", "script": "experiments/visualize_microstates.py",
        "name": "Microstate Analysis",
        "desc": "Microstate topographies & group comparisons",
        "time": "~15-30 min", "flag": "analysis", "cat": "neuroscience",
    },
    {
        "id": "5", "script": "experiments/22_statistical_reporting.py",
        "name": "Statistical Reporting",
        "desc": "Clinical stats and group statistics tables",
        "time": "~2-5 min", "flag": "analysis", "cat": "statistics",
    },
    {
        "id": "6", "script": "experiments/25_statistical_analysis.py",
        "name": "Statistical Analysis",
        "desc": "Bootstrap CIs, subject-level analysis, DeLong AUC, binomial test",
        "time": "~2-5 min", "flag": "analysis", "cat": "statistics",
    },
    {
        "id": "7", "script": "experiments/27_visualization_analysis.py",
        "name": "Advanced Visualizations",
        "desc": "t-SNE, calibration, precision-recall, fold performance",
        "time": "~3-5 min", "flag": "analysis", "cat": "visualization",
    },
    {
        "id": "8", "script": "experiments/24_baseline_comparisons.py",
        "name": "Baseline Comparisons",
        "desc": "SVM, RF, XGBoost, EEGNet benchmarks + McNemar's test",
        "time": "~2-4 hrs", "flag": "baseline", "cat": "benchmarking",
    },
    {
        "id": "9", "script": "experiments/26_cross_dataset_validation.py",
        "name": "Cross-Dataset Validation",
        "desc": "Leave-One-Dataset-Out (LODO) generalizability test",
        "time": "~6-10 hrs", "flag": "cross", "cat": "benchmarking",
    },
    {
        "id": "9.5", "script": "experiments/28_feature_analysis.py",
        "name": "Feature Analysis",
        "desc": "Feature ablation, importance ranking, class-specific discriminators",
        "time": "~10-20 min", "flag": "analysis", "cat": "statistics",
    },
]

# Analysis category definitions for the sub-menu
ANALYSIS_CATEGORIES = {
    "visualization": {
        "label": "Visualization & Figures",
        "desc":  "Publication plots, t-SNE, calibration, precision-recall",
        "icon":  "◆",
    },
    "neuroscience": {
        "label": "Neuroscience & Explainability",
        "desc":  "Connectivity, connectomes, microstates, clinical source",
        "icon":  "◆",
    },
    "statistics": {
        "label": "Statistics & Feature Analysis",
        "desc":  "Bootstrap CIs, DeLong AUC, feature ablation, reporting",
        "icon":  "◆",
    },
    "benchmarking": {
        "label": "Benchmarking & Generalization",
        "desc":  "Baseline models, cross-dataset LODO validation",
        "icon":  "◆",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

C = Fore.CYAN  # primary accent color

LOGO = rf"""
{C}  _  _                   ___ _                      ___               _   {Style.RESET_ALL}
{C} | \| |___ _  _ _ _ ___ / __| |_  _ _ ___ _ _  ___ / __|_ _ __ _ _ __| |_ {Style.RESET_ALL}
{C} | .` / -_) || | '_/ _ \ (__| ' \| '_/ _ \ ' \/ _ \ (_ | '_/ _` | '_ \ ' \{Style.RESET_ALL}
{C} |_|\_\___|\_,_|_| \___/\___|_||_|_| \___/_||_\___/\___|_| \__,_| .__/_||_|{Style.RESET_ALL}
{C}                                                                |_|       {Style.RESET_ALL}
"""

SUBTITLE = (
    f"  {Fore.WHITE}{Style.DIM}"
    "Hierarchical Graph Neural Network for EEG-Based Dementia Classification"
    f"{Style.RESET_ALL}"
)

VERSION_LINE = (
    f"  {Fore.WHITE}{Style.DIM}"
    "v3.0  •  4-class (AD / FTD / MCI / CN)  •  560 subjects  •  5 datasets"
    f"{Style.RESET_ALL}"
)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def render_logo():
    print(LOGO)
    print(SUBTITLE)
    print(VERSION_LINE)
    print()


def hline(char="═", width=72, color=C):
    print(f"{color}{char * width}{Style.RESET_ALL}")


def section_header(title, color=C):
    w = 72
    pad = (w - len(title) - 4) // 2
    rpad = w - pad - len(title) - 4
    print()
    hline("═", w, color)
    print(f"{color}║{' ' * pad}  {Style.BRIGHT}{title}{Style.RESET_ALL}{color}  {' ' * rpad}║{Style.RESET_ALL}")
    hline("═", w, color)


def prompt(msg, color=Fore.YELLOW):
    return input(f"\n{color}  ❯ {msg}{Style.RESET_ALL} ")


def info(msg):
    print(f"  {C}ℹ{Style.RESET_ALL}  {msg}")

def success(msg):
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL}  {msg}")

def warn(msg):
    print(f"  {Fore.YELLOW}⚠{Style.RESET_ALL}  {msg}")

def error(msg):
    print(f"  {Fore.RED}✗{Style.RESET_ALL}  {msg}")

def badge(ok):
    return (f"{Fore.BLACK}{Back.GREEN} PASS {Style.RESET_ALL}" if ok
            else f"{Fore.WHITE}{Back.RED} FAIL {Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

FLAG_COLORS = {
    "preprocess": Fore.BLUE,
    "train":      Fore.RED,
    "analysis":   Fore.GREEN,
    "baseline":   Fore.YELLOW,
    "cross":      Fore.MAGENTA,
}

def print_phase_table(phases=None, selected=None, show_index=True, compact=False):
    """Pretty-print a phase table. If phases is None, uses all PHASES."""
    if phases is None:
        phases = PHASES

    if not compact:
        hdr = f"  {'#':>3}  {'Phase':>5}  {'Name':<26}  {'Time':>10}  Description"
        print(f"\n{Fore.WHITE}{Style.DIM}{hdr}{Style.RESET_ALL}")
        hline("─", 88, Fore.WHITE + Style.DIM)

    for i, phase in enumerate(phases):
        idx = PHASES.index(phase) if phase in PHASES else i
        num = idx + 1
        pid = phase["id"]
        name = phase["name"]
        t = phase["time"]
        desc = phase["desc"]
        nc = FLAG_COLORS.get(phase["flag"], Fore.WHITE)
        crit = f" {Fore.RED}★{Style.RESET_ALL}" if phase.get("critical") else ""

        if selected is not None:
            mk = f"{Fore.GREEN}●{Style.RESET_ALL}" if idx in selected else f"{Fore.RED}○{Style.RESET_ALL}"
        elif show_index:
            mk = f"{C}{num:>2}{Style.RESET_ALL}"
        else:
            mk = "  "

        if compact:
            print(f"    {mk}  {nc}{Style.BRIGHT}{name:<26}{Style.RESET_ALL} {Fore.WHITE}{Style.DIM}{t}{Style.RESET_ALL}{crit}")
        else:
            print(
                f"  {mk}   {Fore.WHITE}{Style.DIM}{pid:>5}{Style.RESET_ALL}"
                f"  {nc}{Style.BRIGHT}{name:<26}{Style.RESET_ALL}"
                f"  {Fore.WHITE}{Style.DIM}{t:>10}{Style.RESET_ALL}"
                f"  {desc}{crit}"
            )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def check_pipeline_status():
    section_header("Pipeline Status", Fore.BLUE)

    preproc = list((PROJECT_ROOT / "outputs" / "preprocessed").glob("*.pkl"))
    n = len(preproc)
    print(f"\n  {badge(n >= 500)}  Preprocessed subjects: {Fore.WHITE}{n}{Style.RESET_ALL}/560")

    rj = RESULTS_DIR / "v3_holdout_results" / "results.json"
    ok = rj.exists()
    print(f"  {badge(ok)}  Training results: {'Found' if ok else 'Missing'}")
    if ok:
        try:
            d = json.load(open(rj))
            acc = d["holdout"]["accuracy"] * 100
            ns = d["holdout"]["n_subjects"]
            print(f"          Holdout accuracy: {Fore.GREEN}{acc:.1f}%{Style.RESET_ALL} (n={ns})")
        except Exception:
            pass

    sj = RESULTS_DIR / "statistical_analysis.json"
    ok2 = sj.exists()
    print(f"  {badge(ok2)}  Statistical analysis: {'Found' if ok2 else 'Missing'}")
    if ok2:
        try:
            sd = json.load(open(sj))
            sa = sd.get("subject_level", {}).get("accuracy", 0) * 100
            if sa > 0:
                print(f"          Subject-level accuracy: {Fore.GREEN}{sa:.1f}%{Style.RESET_ALL}")
        except Exception:
            pass

    # Figures
    dirs = {
        "Publication":    FIGURES_DIR / "publication",
        "Explainability": FIGURES_DIR / "explainability",
        "Statistics":     FIGURES_DIR / "statistics",
        "Root":           FIGURES_DIR,
    }
    print()
    for label, fdir in dirs.items():
        if fdir.exists():
            pngs = list(fdir.glob("*.png"))
            print(f"  {badge(len(pngs) > 0)}  {label}: {Fore.WHITE}{len(pngs)}{Style.RESET_ALL} figures")
        else:
            print(f"  {badge(False)}  {label}: missing")

    # Key CSVs/JSONs
    keys = [
        ("Baseline comparison",  RESULTS_DIR / "baseline_comparison.csv"),
        ("Cross-dataset val.",   RESULTS_DIR / "cross_dataset_validation.csv"),
        ("Feature analysis",     RESULTS_DIR / "feature_analysis.json"),
        ("Feature summary",      RESULTS_DIR / "feature_discriminative_summary.csv"),
    ]
    print()
    for label, fp in keys:
        ok = fp.exists()
        ts = ""
        if ok:
            ts = f" ({datetime.fromtimestamp(fp.stat().st_mtime).strftime('%m-%d %H:%M')})"
        print(f"  {badge(ok)}  {label}{ts}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  SCRIPT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase(phase, num=None, total=None):
    pid, name, script = phase["id"], phase["name"], phase["script"]
    full_path = PROJECT_ROOT / script
    if not full_path.exists():
        warn(f"Script not found: {script}")
        return False, 0

    tag = f"[{num}/{total}] " if num and total else ""
    print()
    hline("━", 72, C)
    print(f"  {C}▶{Style.RESET_ALL}  {tag}{Fore.WHITE}{Style.BRIGHT}Phase {pid}: {name}{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}{Style.DIM}{phase['desc']}{Style.RESET_ALL}")
    print(f"     {Fore.WHITE}{Style.DIM}Started: {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
    hline("━", 72, C)
    print()

    start = time.time()
    try:
        cmd = [sys.executable, str(full_path)]
        if phase.get("extra_args"):
            cmd.extend(phase["extra_args"])
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=86400)
        elapsed = time.time() - start
        if result.returncode == 0:
            success(f"Phase {pid} completed in {Fore.WHITE}{timedelta(seconds=int(elapsed))}{Style.RESET_ALL}")
            return True, elapsed
        else:
            error(f"Phase {pid} failed (exit {result.returncode}) after {timedelta(seconds=int(elapsed))}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        error(f"Phase {pid} timed out after {timedelta(seconds=int(elapsed))}")
        return False, elapsed
    except KeyboardInterrupt:
        elapsed = time.time() - start
        warn(f"Phase {pid} interrupted after {timedelta(seconds=int(elapsed))}")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        error(f"Phase {pid} error: {e}")
        return False, elapsed


def run_phases(phase_indices):
    results = []
    t0 = time.time()
    for i, idx in enumerate(phase_indices, 1):
        p = PHASES[idx]
        ok, el = run_phase(p, num=i, total=len(phase_indices))
        results.append((p, ok, el))
        if not ok and p.get("critical"):
            error("Training failed — cannot proceed with analysis.")
            break

    total = time.time() - t0
    section_header("Run Summary", Fore.GREEN if all(r[1] for r in results) else Fore.RED)
    for p, ok, el in results:
        print(f"  {badge(ok)}  Phase {p['id']:>5}: {p['name']:<26} {Fore.WHITE}{Style.DIM}{timedelta(seconds=int(el))}{Style.RESET_ALL}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\n  Total: {Fore.GREEN}{passed} passed{Style.RESET_ALL}, {Fore.RED}{failed} failed{Style.RESET_ALL}"
          f" in {Fore.WHITE}{timedelta(seconds=int(total))}{Style.RESET_ALL}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MENU SCREENS
# ═══════════════════════════════════════════════════════════════════════════════

def _confirm_and_run(indices):
    """Show selected phases and confirm."""
    print_phase_table(selected=set(indices), show_index=False)
    if prompt(f"Run {len(indices)} phase(s)? (y/n)", Fore.GREEN).lower() == "y":
        run_phases(indices)


def _skip_prompts():
    """Ask common skip questions, return (skip_baselines, skip_cross)."""
    sb = prompt("Skip baseline comparisons? (y/n)", Fore.YELLOW).strip().lower() == "y"
    sc = prompt("Skip cross-dataset validation? (y/n)", Fore.YELLOW).strip().lower() == "y"
    return sb, sc


def _apply_skips(indices, skip_base, skip_cross):
    if skip_base:
        indices = [i for i in indices if PHASES[i]["flag"] != "baseline"]
    if skip_cross:
        indices = [i for i in indices if PHASES[i]["flag"] != "cross"]
    return indices


def menu_full_run():
    section_header("Full Pipeline Run", Fore.BLUE)
    info(f"All 13 phases end-to-end.  Estimated: {Fore.WHITE}13-25 hours{Style.RESET_ALL}")
    print()
    warn("Long-running phases:")
    print(f"    Phase 2  (Training):      ~4-8 hrs")
    print(f"    Phase 8  (Baselines):     ~2-4 hrs")
    print(f"    Phase 9  (Cross-dataset): ~6-10 hrs")
    sb, sc = _skip_prompts()
    indices = _apply_skips(list(range(len(PHASES))), sb, sc)
    _confirm_and_run(indices)


def menu_analysis_only():
    section_header("Analysis Only", Fore.GREEN)
    info("Skip preprocessing & training. Requires existing results.json")
    rj = RESULTS_DIR / "v3_holdout_results" / "results.json"
    if not rj.exists():
        error("results.json not found! Run training (Phase 2) first.")
        return
    success("Found results.json")
    sb, sc = _skip_prompts()
    indices = [i for i, p in enumerate(PHASES) if p["flag"] not in ("preprocess", "train")]
    indices = _apply_skips(indices, sb, sc)
    _confirm_and_run(indices)


def menu_quick_analysis():
    section_header("Quick Analysis (Figures Only)", Fore.GREEN)
    info("Analysis-flagged phases only. Skips baselines & cross-validation.")
    rj = RESULTS_DIR / "v3_holdout_results" / "results.json"
    if not rj.exists():
        error("results.json not found! Run training first.")
        return
    indices = [i for i, p in enumerate(PHASES) if p["flag"] == "analysis"]
    _confirm_and_run(indices)


def menu_select_phases():
    section_header("Select Phases", Fore.MAGENTA)
    print_phase_table()
    print(f"  {Fore.WHITE}{Style.DIM}Enter numbers: 3 4 5  or  1,2,3  or  3-7{Style.RESET_ALL}")
    raw = prompt("Phases to run:")
    selected = set()
    for tok in raw.replace(",", " ").split():
        tok = tok.strip()
        if "-" in tok:
            try:
                a, b = tok.split("-", 1)
                for x in range(int(a), int(b) + 1):
                    if 1 <= x <= len(PHASES):
                        selected.add(x - 1)
            except ValueError:
                pass
        else:
            try:
                x = int(tok)
                if 1 <= x <= len(PHASES):
                    selected.add(x - 1)
            except ValueError:
                pass
    if not selected:
        warn("No valid phases selected.")
        return
    _confirm_and_run(sorted(selected))


def menu_single_phase():
    section_header("Run Single Phase", Fore.YELLOW)
    print_phase_table()
    raw = prompt("Enter phase number (1-13):")
    try:
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(PHASES):
            p = PHASES[idx]
            info(f"Selected: {Fore.WHITE}{Style.BRIGHT}{p['name']}{Style.RESET_ALL} — {p['desc']}")
            if prompt("Run now? (y/n)", Fore.GREEN).lower() == "y":
                run_phases([idx])
        else:
            error("Invalid phase number.")
    except ValueError:
        error("Please enter a number.")


# ─── ANALYSIS CATEGORY SUB-MENU ──────────────────────────────────────────────

def menu_analysis_by_category():
    """Sub-menu: pick an analysis category, then run those phases."""
    section_header("Analysis by Category", C)

    cats = list(ANALYSIS_CATEGORIES.items())
    for i, (key, meta) in enumerate(cats, 1):
        phases_in_cat = [p for p in PHASES if p.get("cat") == key]
        n = len(phases_in_cat)
        print(
            f"    {C}{Style.BRIGHT}[{i}]{Style.RESET_ALL}  {meta['icon']}  "
            f"{Fore.WHITE}{Style.BRIGHT}{meta['label']}{Style.RESET_ALL}"
            f"  {Fore.WHITE}{Style.DIM}({n} phases){Style.RESET_ALL}"
        )
        print(f"         {Fore.WHITE}{Style.DIM}{meta['desc']}{Style.RESET_ALL}")
        # List individual phases
        for p in phases_in_cat:
            nc = FLAG_COLORS.get(p["flag"], Fore.WHITE)
            print(f"           {nc}•{Style.RESET_ALL} {p['name']} {Fore.WHITE}{Style.DIM}[{p['id']}]{Style.RESET_ALL}")
        print()

    print(f"    {C}{Style.BRIGHT}[A]{Style.RESET_ALL}  Run ALL analysis categories")
    print(f"    {C}{Style.BRIGHT}[0]{Style.RESET_ALL}  Back to main menu")

    raw = prompt("Select category:").strip().lower()

    if raw == "0":
        return
    elif raw == "a":
        # All analysis
        indices = [i for i, p in enumerate(PHASES) if p.get("cat") in ANALYSIS_CATEGORIES]
        if indices:
            _confirm_and_run(indices)
        return

    try:
        ci = int(raw) - 1
        if 0 <= ci < len(cats):
            cat_key = cats[ci][0]
            indices = [i for i, p in enumerate(PHASES) if p.get("cat") == cat_key]
            if indices:
                phase_names = [PHASES[i]["name"] for i in indices]
                info(f"Category: {Fore.WHITE}{Style.BRIGHT}{cats[ci][1]['label']}{Style.RESET_ALL}")
                _confirm_and_run(indices)
            else:
                warn("No phases in this category.")
        else:
            error("Invalid selection.")
    except ValueError:
        error("Please enter a number or 'A'.")


# ═══════════════════════════════════════════════════════════════════════════════
#  NON-INTERACTIVE CLI MODE  (backward-compat with run_all.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli_mode():
    parser = argparse.ArgumentParser(description="NeuroChronoGraph Pipeline Runner", add_help=False)
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive UI")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--analysis-only", action="store_true", help="Skip preprocessing and training")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing only")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline comparisons")
    parser.add_argument("--skip-cross-validation", action="store_true", help="Skip cross-dataset validation")
    parser.add_argument("--quick", action="store_true", help="Quick analysis: figures only")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--category", type=str, default="",
                       help="Run a specific category: visualization, neuroscience, statistics, benchmarking")
    parser.add_argument("--phases", type=str, default="", help="Comma-separated phase numbers (e.g. 3,4,5)")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        return None  # go interactive

    render_logo()

    if args.help:
        parser.print_help()
        return 0
    if args.status:
        check_pipeline_status()
        return 0

    # Category mode
    if args.category:
        cat = args.category.lower()
        if cat not in ANALYSIS_CATEGORIES:
            error(f"Unknown category '{cat}'. Options: {', '.join(ANALYSIS_CATEGORIES.keys())}")
            return 1
        indices = [i for i, p in enumerate(PHASES) if p.get("cat") == cat]
        if not indices:
            error("No phases in this category.")
            return 1
        info(f"Running category: {ANALYSIS_CATEGORIES[cat]['label']}")
        print_phase_table(selected=set(indices), show_index=False)
        run_phases(indices)
        return 0

    # Phase selection
    if args.phases:
        indices = []
        for tok in args.phases.split(","):
            try:
                idx = int(tok.strip()) - 1
                if 0 <= idx < len(PHASES):
                    indices.append(idx)
            except ValueError:
                pass
        if not indices:
            error("No valid phase numbers.")
            return 1
    else:
        indices = list(range(len(PHASES)))
        if args.analysis_only:
            indices = [i for i in indices if PHASES[i]["flag"] not in ("preprocess", "train")]
        elif args.skip_preprocess:
            indices = [i for i in indices if PHASES[i]["flag"] != "preprocess"]
        if args.skip_baselines:
            indices = [i for i in indices if PHASES[i]["flag"] != "baseline"]
        if args.skip_cross_validation:
            indices = [i for i in indices if PHASES[i]["flag"] != "cross"]
        if args.quick:
            indices = [i for i in indices if PHASES[i]["flag"] == "analysis"]

    info(f"Running {len(indices)} phases...")
    print_phase_table(selected=set(indices), show_index=False)
    run_phases(indices)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════════════════

def main_menu():
    while True:
        clear_screen()
        render_logo()

        hline("═", 72, C)
        title = "MAIN MENU"
        pad = (72 - len(title) - 4) // 2
        rpad = 72 - pad - len(title) - 4
        print(f"{C}║{' ' * pad}  {Fore.WHITE}{Style.BRIGHT}{title}{Style.RESET_ALL}{C}  {' ' * rpad}║{Style.RESET_ALL}")
        hline("═", 72, C)

        items = [
            ("1", "Full Pipeline Run",         "All 13 phases end-to-end"),
            ("2", "Analysis Only",             "Skip preprocessing & training"),
            ("3", "Quick Analysis",            "Figures & stats only (fastest)"),
            ("4", "Analysis by Category",      "Visualization / Neuroscience / Statistics / Benchmarking"),
            ("5", "Select Phases",             "Choose specific phases to run"),
            ("6", "Run Single Phase",          "Pick and run one phase"),
            ("7", "Pipeline Status",           "Check existing outputs & results"),
            ("0", "Exit",                      ""),
        ]

        print()
        for key, label, desc in items:
            d = f" {Fore.WHITE}{Style.DIM}— {desc}{Style.RESET_ALL}" if desc else ""
            color = Fore.RED if key == "0" else C
            print(f"    {color}{Style.BRIGHT}[{key}]{Style.RESET_ALL}  {label}{d}")

        print()
        hline("─", 72, Fore.WHITE + Style.DIM)
        print(f"  {Fore.WHITE}{Style.DIM}CLI: python experiments/cli.py --analysis-only --skip-baselines{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}{Style.DIM}     python experiments/cli.py --category neuroscience{Style.RESET_ALL}")

        choice = prompt("Select option:").strip()

        actions = {
            "1": menu_full_run,
            "2": menu_analysis_only,
            "3": menu_quick_analysis,
            "4": menu_analysis_by_category,
            "5": menu_select_phases,
            "6": menu_single_phase,
            "7": check_pipeline_status,
        }

        if choice in actions:
            clear_screen()
            render_logo()
            actions[choice]()
            prompt("Press Enter to continue...", Fore.WHITE + Style.DIM)
        elif choice in ("0", "q", "exit", "quit"):
            print(f"\n  {C}Goodbye!{Style.RESET_ALL}\n")
            break
        else:
            warn("Invalid option. Choose 0-7.")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        result = run_cli_mode()
        if result is None:
            main_menu()
        else:
            sys.exit(result)
    except KeyboardInterrupt:
        print(f"\n\n  {Fore.YELLOW}Interrupted.{Style.RESET_ALL}\n")
        sys.exit(130)
