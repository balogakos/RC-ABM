"""
Aggregate Ablation Ensemble Results
=====================================
Run this after run_ablation.py to produce:
  - multi_runs/results_ablation/ensemble_centre_performance_ablation_mean.csv
  - multi_runs/results_ablation/ensemble_centre_performance_ablation_std.csv

These files are then loaded by plot_diffusion_comparison.py for Fig. 10.
"""

import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPT_DIR   = Path(__file__).resolve().parent    # multi_runs/
_PROJECT_ROOT = _SCRIPT_DIR.parent                  # Retail_ABM/

def aggregate_ablation_ensemble():
    results_dir = _SCRIPT_DIR / "results_ablation"

    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found. Run run_ablation.py first.")
        return

    perf_files = sorted(results_dir.glob("run_*_retail_centre_performance_*.csv"))
    print(f"Found {len(perf_files)} ablation centre performance files.")

    if len(perf_files) == 0:
        print("No files found. Exiting.")
        return

    perf_dfs = []
    for f in perf_files:
        df = pd.read_csv(f).set_index('Retail_Centre')
        # Sum across all trip-type × mode columns to get Total_Visits
        visit_cols = [c for c in df.columns if c not in ('Total_Revenue',)]
        df['Total_Visits'] = df[visit_cols].sum(axis=1)
        perf_dfs.append(df)

    df_concat = pd.concat(perf_dfs)
    perf_mean = df_concat.groupby('Retail_Centre').mean()
    perf_std  = df_concat.groupby('Retail_Centre').std()

    out_mean = results_dir / "ensemble_centre_performance_ablation_mean.csv"
    out_std  = results_dir / "ensemble_centre_performance_ablation_std.csv"

    perf_mean.to_csv(out_mean)
    perf_std.to_csv(out_std)

    print(f"Saved: {out_mean.name}")
    print(f"Saved: {out_std.name}")
    print("Done. Run plot_diffusion_comparison.py to produce Fig. 10.")

if __name__ == "__main__":
    aggregate_ablation_ensemble()
