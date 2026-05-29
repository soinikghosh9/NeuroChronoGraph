"""
NeuroChronoGraph: Master End-to-End Pipeline Runner

Single command to run the ENTIRE pipeline from preprocessing through
training, analysis, visualization, and all publication-ready outputs.

Usage:
    python experiments/run_all.py                  # Full run (includes preprocessing)
    python experiments/run_all.py --skip-preprocess # Skip preprocessing (if already done)
    python experiments/run_all.py --analysis-only   # Skip training, run analysis only

Estimated time: 13-25 hours (full run with GPU)
"""

import os
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Strip ANSI colour/control sequences so tqdm bars don't pollute log files.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Match a tqdm progress line (description + percentage bar).
# We collapse consecutive progress updates into a single rewrite per phase.
_TQDM_RE = re.compile(r"^\s*([^:]+):\s*\d+%\|.*\|\s*\d+/\d+\b")

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = LOGS_DIR / f"run_{RUN_ID}"
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_LOG = RUN_LOG_DIR / "_summary.log"
LATEST_LINK = LOGS_DIR / "latest"


def _slog(msg: str) -> None:
    """Append a line to the master summary log AND echo to stdout."""
    print(msg, flush=True)
    with SUMMARY_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

SCRIPTS = [
    # (script_path, description, phase, skip_flag)
    ("experiments/01_preprocess_all.py", "Preprocessing EEG data", "Phase 1", "preprocess"),
    ("experiments/18_train_hierarchical.py", "Training hierarchical model (5-fold CV + holdout)", "Phase 2", "train"),
    ("experiments/09_generate_publication_plots.py", "Generating publication plots", "Phase 3", "analysis"),
    ("experiments/20_neuroscientific_analysis.py", "Neuroscientific analysis", "Phase 4", "analysis"),
    ("experiments/16_clinical_source_analysis.py", "Clinical source & regional power analysis", "Phase 4.5", "analysis"),
    ("experiments/17_connectivity_visualization.py", "Connectivity & connectome visualization", "Phase 4.6", "analysis"),
    ("experiments/visualize_microstates.py", "Microstate analysis & visualization", "Phase 4.7", "analysis"),
    ("experiments/22_statistical_reporting.py", "Statistical reporting", "Phase 5", "analysis"),
    ("experiments/25_statistical_analysis.py", "Bootstrap CIs & subject-level analysis", "Phase 6", "analysis"),
    ("experiments/27_visualization_analysis.py", "Publication visualizations", "Phase 7", "analysis"),
    ("experiments/24_baseline_comparisons.py", "Baseline model comparisons", "Phase 8", "baseline"),
    ("experiments/26_cross_dataset_validation.py", "Cross-dataset validation (LODO)", "Phase 9", "cross"),
    ("experiments/28_feature_analysis.py", "Feature importance & class-specific analysis", "Phase 9.5", "analysis"),
    ("experiments/29_feature_stream_ablation.py", "Feature-stream ablation (spectral/connectivity/complexity/microstate)", "Phase 9.6", "analysis"),
]


def run_script(script_path, description, phase):
    """Run a single script, tee its stdout+stderr to a per-phase log file."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        _slog(f"  WARNING: {script_path} not found, skipping")
        return False, 0, None

    safe_phase = phase.replace(" ", "_").replace(".", "p")
    safe_name = Path(script_path).stem
    log_file = RUN_LOG_DIR / f"{safe_phase}__{safe_name}.log"

    _slog(f"\n{'='*70}")
    _slog(f"  {phase}: {description}")
    _slog(f"  Script: {script_path}")
    _slog(f"  Log:    {log_file.relative_to(PROJECT_ROOT)}")
    _slog(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _slog(f"{'='*70}")

    start = time.time()
    try:
        cmd = [sys.executable, "-u", str(full_path)]
        if "01_preprocess" in script_path:
            cmd.append("--skip-existing")

        # Throttle tqdm in subprocesses: in non-TTY (piped) mode tqdm emits a
        # fresh full line per iteration AND per set_postfix, which floods logs.
        # Throttling to 30 s and disabling its leave-a-final-line keeps logs
        # tight while still showing periodic progress.
        env = os.environ.copy()
        env.setdefault("TQDM_MININTERVAL", "30")
        env.setdefault("TQDM_LEAVE", "False")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # Reproducibility: pin RNG seed and hash seed for every child phase.
        # Override at the shell level (e.g. ``set NCG_SEED=7``) to vary runs.
        env.setdefault("NCG_SEED", "42")
        env.setdefault("PYTHONHASHSEED", env["NCG_SEED"])
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        with log_file.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write(f"# Phase: {phase}\n")
            lf.write(f"# Script: {script_path}\n")
            lf.write(f"# Started: {datetime.now().isoformat()}\n")
            lf.write(f"# Cmd: {' '.join(cmd)}\n")
            lf.write("=" * 70 + "\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            # Read by character so '\r' (tqdm in-place updates) is treated as a
            # line break. Collapse consecutive progress-bar redraws for the same
            # tqdm 'desc' into a single rewriting line.
            buf = []
            last_progress_desc = None
            try:
                while True:
                    ch = proc.stdout.read(1)
                    if not ch:
                        break
                    if ch in ("\n", "\r"):
                        line = "".join(buf).rstrip()
                        buf.clear()
                        if not line:
                            continue
                        clean = _ANSI_RE.sub("", line)
                        m = _TQDM_RE.match(clean)
                        if m:
                            desc = m.group(1).strip()
                            if desc == last_progress_desc:
                                # Same bar updating — overwrite the previous
                                # progress line in stdout AND log instead of
                                # appending a new one.
                                sys.stdout.write("\r" + clean)
                                sys.stdout.flush()
                                continue
                            last_progress_desc = desc
                            sys.stdout.write("\n" + clean)
                            sys.stdout.flush()
                            lf.write(clean + "\n")
                            lf.flush()
                        else:
                            if last_progress_desc is not None:
                                sys.stdout.write("\n")
                                last_progress_desc = None
                            sys.stdout.write(clean + "\n")
                            sys.stdout.flush()
                            lf.write(clean + "\n")
                            lf.flush()
                    else:
                        buf.append(ch)
                if buf:
                    line = "".join(buf).rstrip()
                    if line:
                        clean = _ANSI_RE.sub("", line)
                        sys.stdout.write(clean + "\n")
                        lf.write(clean + "\n")
                proc.wait(timeout=86400)
            except subprocess.TimeoutExpired:
                proc.kill()
                elapsed = time.time() - start
                lf.write(f"\n# TIMED OUT after {elapsed:.1f}s\n")
                _slog(f"\n  X {phase} TIMED OUT after {timedelta(seconds=int(elapsed))}")
                _slog(f"     Log: {log_file.relative_to(PROJECT_ROOT)}")
                return False, elapsed, log_file

            elapsed = time.time() - start
            lf.write(f"\n# Finished: {datetime.now().isoformat()} (exit={proc.returncode}, elapsed={elapsed:.1f}s)\n")

        if proc.returncode == 0:
            _slog(f"\n  OK {phase} COMPLETED in {timedelta(seconds=int(elapsed))}")
            return True, elapsed, log_file
        else:
            _slog(f"\n  X {phase} FAILED (exit code {proc.returncode}) after {timedelta(seconds=int(elapsed))}")
            _slog(f"     Log: {log_file.relative_to(PROJECT_ROOT)}")
            return False, elapsed, log_file

    except Exception as e:
        elapsed = time.time() - start
        _slog(f"\n  X {phase} ERROR: {e}")
        return False, elapsed, log_file


def main():
    parser = argparse.ArgumentParser(description="NeuroChronoGraph: Run entire pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                       help="Skip preprocessing (Phase 1)")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Skip preprocessing and training, run analysis only (Phases 3-9.5)")
    parser.add_argument("--skip-cross-validation", action="store_true",
                       help="Skip cross-dataset validation (Phase 9, saves ~6-10 hours)")
    parser.add_argument("--skip-baselines", action="store_true",
                       help="Skip baseline comparisons (Phase 8, saves ~2-4 hours)")
    args = parser.parse_args()

    _slog("=" * 70)
    _slog("  NEUROCHRONOGRAPH: MASTER PIPELINE RUNNER")
    _slog(f"  Run ID:  {RUN_ID}")
    _slog(f"  Logs:    {RUN_LOG_DIR.relative_to(PROJECT_ROOT)}")
    _slog(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _slog("=" * 70)

    if args.analysis_only:
        _slog("  Mode: ANALYSIS ONLY (skipping preprocessing + training)")
    elif args.skip_preprocess:
        _slog("  Mode: SKIP PREPROCESSING")
    else:
        _slog("  Mode: FULL RUN")

    if args.skip_cross_validation:
        _slog("  Note: Cross-dataset validation SKIPPED")
    if args.skip_baselines:
        _slog("  Note: Baseline comparisons SKIPPED")

    # Maintain a 'latest' pointer to the current run's log directory.
    try:
        if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
            if LATEST_LINK.is_symlink() or LATEST_LINK.is_file():
                LATEST_LINK.unlink()
            else:
                import shutil
                shutil.rmtree(LATEST_LINK)
        try:
            LATEST_LINK.symlink_to(RUN_LOG_DIR, target_is_directory=True)
        except (OSError, NotImplementedError):
            # On Windows without dev-mode/admin, fall back to writing a pointer file.
            LATEST_LINK.write_text(str(RUN_LOG_DIR), encoding="utf-8")
    except Exception:
        pass

    results = []
    total_start = time.time()

    for script_path, description, phase, skip_flag in SCRIPTS:
        # Skip logic
        if args.skip_preprocess and skip_flag == "preprocess":
            _slog(f"\n  SKIPPING {phase}: {description}")
            continue
        if args.analysis_only and skip_flag in ("preprocess", "train"):
            _slog(f"\n  SKIPPING {phase}: {description}")
            continue
        if args.skip_cross_validation and skip_flag == "cross":
            _slog(f"\n  SKIPPING {phase}: {description}")
            continue
        if args.skip_baselines and skip_flag == "baseline":
            _slog(f"\n  SKIPPING {phase}: {description}")
            continue

        success, elapsed, log_file = run_script(script_path, description, phase)
        results.append((phase, description, success, elapsed, log_file))

        # If training fails, no point continuing
        if not success and skip_flag == "train":
            _slog("\n  CRITICAL: Training failed. Cannot proceed with analysis.")
            _slog(f"  See log: {log_file.relative_to(PROJECT_ROOT) if log_file else '(no log)'}")
            _slog("  Fix training issues and re-run.")
            break

    # Summary
    total_elapsed = time.time() - total_start
    _slog("\n\n" + "=" * 70)
    _slog("  PIPELINE SUMMARY")
    _slog("=" * 70)
    _slog(f"  Total time: {timedelta(seconds=int(total_elapsed))}")
    _slog(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _slog(f"  All logs:   {RUN_LOG_DIR.relative_to(PROJECT_ROOT)}")
    _slog("")

    for phase, desc, success, elapsed, log_file in results:
        status = "OK" if success else "X "
        log_rel = log_file.relative_to(PROJECT_ROOT) if log_file else "(no log)"
        _slog(f"  {status} {phase:12s} {desc:50s} {timedelta(seconds=int(elapsed))}  [{log_rel}]")

    succeeded = sum(1 for r in results if r[2])
    failed = sum(1 for r in results if not r[2])
    _slog(f"\n  {succeeded} succeeded, {failed} failed")

    if failed == 0:
        _slog("\n  ALL PHASES COMPLETED SUCCESSFULLY!")
        _slog("\n  Next steps:")
        _slog("  1. Check outputs/results/ for all generated data files")
        _slog("  2. Check outputs/figures/publication/ for all generated figures")
        _slog("  3. Fill TBD values in main.tex from the generated JSON/CSV files")
        _slog("  4. Compile main.tex to produce the final PDF")
    else:
        _slog(f"\n  Some phases failed. Per-phase logs in: {RUN_LOG_DIR.relative_to(PROJECT_ROOT)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
