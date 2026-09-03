"""
RC-ABM Ablation Runner — Phase 2: Diffusion OFF
================================================
Run this AFTER the main ensemble (runner.py) has completed.

This script:
  1. Temporarily disables spatial diffusion (DIFFUSION_ENABLED = False)
  2. Runs 3 ensemble iterations (sufficient for ablation comparison)
  3. Saves results to multi_runs/results_ablation/
  4. Restores DIFFUSION_ENABLED = True when done

The outputs are used by plot_diffusion_comparison.py to produce Fig. 10.
"""

import sys
from pathlib import Path

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parent.parent
SIM_DIR  = ROOT_DIR / "simulation"
for d in [str(ROOT_DIR), str(SIM_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

import config

# --- Ablation Configuration ---
NUM_RUNS_ABLATION = 3     # Fewer runs needed — comparison only, not primary results
DAYS_ABLATION     = 120   # Must match main ensemble for like-for-like comparison
EVAL_FREQ_ABLATION = 30

# ============================================================
# 1. Patch runner module globals before importing run_ensemble
# ============================================================
import runner as _runner

_runner.NUM_RUNS       = NUM_RUNS_ABLATION
_runner.DAYS           = DAYS_ABLATION
_runner.EVAL_FREQ      = EVAL_FREQ_ABLATION
_runner.RESULTS_SUBDIR = "results_ablation"  # separate folder — never mixed with main results

# 2. Disable diffusion in config
_original_diffusion = getattr(config, 'DIFFUSION_ENABLED', True)
config.DIFFUSION_ENABLED = False
print("=" * 60)
print("ABLATION MODE: DIFFUSION_ENABLED = False")
print(f"Runs: {NUM_RUNS_ABLATION} | Days: {DAYS_ABLATION} | EVAL_FREQ: {EVAL_FREQ_ABLATION}")
print(f"Results -> multi_runs/results_ablation/")
print("=" * 60 + "\n")

# 3. Run the ensemble
try:
    _runner.run_ensemble()
finally:
    # 4. Always restore diffusion, even if run fails
    config.DIFFUSION_ENABLED = _original_diffusion
    print(f"\nDIFFUSION_ENABLED restored to: {_original_diffusion}")
    print("Ablation complete. Run aggregate_ensemble_ablation.py next.")
