import os
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_ensemble_results():
    """Reads all run parquets and prepares for statistical analysis."""
    results_dir = Path(__file__).resolve().parent / "results"
    run_files = list(results_dir.glob("run_*.parquet"))
    
    if not run_files:
        print(f"No run files found in {results_dir}")
        return
    
    print(f"Found {len(run_files)} runs. Consolidating...")
    
    all_runs_data = []
    for i, file_path in enumerate(run_files):
        df = pd.read_parquet(file_path)
        df['Run_ID'] = i + 1
        all_runs_data.append(df)
        
    full_df = pd.concat(all_runs_data, ignore_index=True)
    return full_df

if __name__ == "__main__":
    df = aggregate_ensemble_results()
    if df is not None:
        print(f"Consolidated {len(df)} total trip records across the ensemble.")
