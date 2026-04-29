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
    
    # 4. Mean Quality Metrics per Centre per Run
    quality_run_stats = df.groupby(['Run_ID', 'Retail_Centre']).agg({
        'Travel_Time_Min': 'mean',
        'Utility_Score': 'mean'
    }).reset_index().set_index(['Run_ID', 'Retail_Centre'])
    
    # Combine them
    combined = pd.concat([mode_run_visits, type_run_visits, quality_run_stats], axis=1).fillna(0)
    
    # 5. Aggregate across Runs (Mean & Std)
    stats = combined.groupby('Retail_Centre').agg(['mean', 'std'])
    
    # Flatten columns: 'walk_mean', 'walk_std', etc.
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns]
    
    # Calculate Mean total visits separately as it was lost in concatenation logic above
    total_visits = combined.iloc[:, :len(mode_run_visits.columns) + len(type_run_visits.columns)].sum(axis=1)
    stats['Total_Visits_mean'] = total_visits.groupby('Retail_Centre').mean()
    stats['Total_Visits_std']  = total_visits.groupby('Retail_Centre').std()
    
    return stats

def save_final_summary(stats, filename="ensemble_summary.csv"):
    output_path = Path(__file__).resolve().parent / filename
    stats.to_csv(output_path)
    print(f"\nFinal ensemble summary saved to: {output_path}")

if __name__ == "__main__":
    df = aggregate_ensemble_results()
    if df is not None:
        stats = compute_dimensional_stats(df)
        save_final_summary(stats)
