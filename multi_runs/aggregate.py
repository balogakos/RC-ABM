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

def compute_dimensional_stats(df):
    """Calculates Mean and Std for Visits per Centre, Mode, and Type."""
    print("Computing multi-dimensional statistics (Modes & Trip Types)...")
    
    # 1. Base Visits per Centre per Run
    centre_run_visits = df.groupby(['Run_ID', 'Retail_Centre']).size().reset_index(name='Visits')
    
    # 2. Visits per Mode per Centre per Run
    mode_run_visits = df.groupby(['Run_ID', 'Retail_Centre', 'Transport_Mode']).size().unstack(fill_value=0)
    
    # 3. Visits per Trip Type per Centre per Run
    type_run_visits = df.groupby(['Run_ID', 'Retail_Centre', 'Trip_Type']).size().unstack(fill_value=0)
    
    # Combine them
    combined = pd.concat([mode_run_visits, type_run_visits], axis=1).fillna(0)
    combined['Total_Visits'] = combined.sum(axis=1)
    
    # 4. Aggregate across Runs (Mean & Std)
    stats = combined.groupby('Retail_Centre').agg(['mean', 'std'])
    
    # Flatten columns: 'walk_mean', 'walk_std', etc.
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns]
    return stats

if __name__ == "__main__":
    df = aggregate_ensemble_results()
    if df is not None:
        stats = compute_dimensional_stats(df)
        print(f"Aggregated stats for {len(stats)} Retail Centres.")
        print(stats.head())
