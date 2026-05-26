import os
import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_ensemble():
    print("=" * 60)
    print("AGGREGATING ENSEMBLE RESULTS FROM RESULTS FOLDER")
    print("=" * 60)
    
    results_dir = Path(r"C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\multi_runs\results")
    outputs_root = Path(r"C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs")
    
    # 1. Back up and clean the main outputs folders
    print("\nStep 1: Cleaning up outputs directory and moving old files to backup...")
    backup_dir = outputs_root / "backup_old_runs"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for category in ["daily_summaries", "centre_performance", "utility_convergence"]:
        cat_dir = outputs_root / category
        if cat_dir.exists():
            for f in cat_dir.glob("*.csv"):
                # Move to backup
                shutil.move(str(f), str(backup_dir / f.name))
            print(f"  Cleaned and backed up: {category}/")
            
    # 2. Identify the run files in multi_runs/results/
    print("\nStep 2: Identifying run files in results folder...")
    run_daily_files = sorted(results_dir.glob("run_*_daily_summary_*.csv"))
    run_perf_files = sorted(results_dir.glob("run_*_retail_centre_performance_*.csv"))
    run_conv_files = sorted(results_dir.glob("run_*_utility_convergence_*.csv"))
    
    print(f"  Found {len(run_daily_files)} daily summaries")
    print(f"  Found {len(run_perf_files)} retail centre performance files")
    print(f"  Found {len(run_conv_files)} utility convergence files")
    
    if len(run_daily_files) == 0 or len(run_perf_files) == 0 or len(run_conv_files) == 0:
        print("Error: Could not find matching run files in multi_runs/results/")
        return
        
    # 3. Copy clean version of the 5 runs to outputs directory (without run_X_ prefix)
    # This enables plot_results.py in 'average' mode to load exactly these 5 runs.
    print("\nStep 3: Copying clean run files to outputs folder...")
    for idx, (daily_f, perf_f, conv_f) in enumerate(zip(run_daily_files, run_perf_files, run_conv_files)):
        # Extract base name (remove run_X_ prefix)
        daily_dest_name = daily_f.name.split("daily_summary_")[1]
        perf_dest_name = perf_f.name.split("retail_centre_performance_")[1]
        conv_dest_name = conv_f.name.split("utility_convergence_")[1]
        
        shutil.copy2(str(daily_f), str(outputs_root / "daily_summaries" / f"daily_summary_{daily_dest_name}"))
        shutil.copy2(str(perf_f), str(outputs_root / "centre_performance" / f"retail_centre_performance_{perf_dest_name}"))
        shutil.copy2(str(conv_f), str(outputs_root / "utility_convergence" / f"utility_convergence_{conv_dest_name}"))
        
        print(f"  Copied Run {idx+1} outputs to main outputs/ folders.")
        
    # 4. Compute explicit averaged ensemble files and save in results/
    print("\nStep 4: Computing explicit averaged files...")
    
    # 4.1 Daily Summaries Average
    daily_dfs = [pd.read_csv(f) for f in run_daily_files]
    df_daily_all = pd.concat(daily_dfs)
    numeric_cols_daily = df_daily_all.select_dtypes(include=[np.number]).columns.tolist()
    if 'Day' in numeric_cols_daily:
        numeric_cols_daily.remove('Day')
    daily_mean = df_daily_all.groupby('Day')[numeric_cols_daily].mean().reset_index()
    daily_std = df_daily_all.groupby('Day')[numeric_cols_daily].std().reset_index()
    
    daily_mean.to_csv(results_dir / "ensemble_daily_summary_mean.csv", index=False)
    daily_std.to_csv(results_dir / "ensemble_daily_summary_std.csv", index=False)
    print("  Saved: ensemble_daily_summary_mean.csv & ensemble_daily_summary_std.csv")
    
    # 4.2 Utility Convergence Average
    conv_dfs = [pd.read_csv(f) for f in run_conv_files]
    df_conv_all = pd.concat(conv_dfs)
    numeric_cols_conv = df_conv_all.select_dtypes(include=[np.number]).columns.tolist()
    if 'Day' in numeric_cols_conv:
        numeric_cols_conv.remove('Day')
    conv_mean = df_conv_all.groupby('Day')[numeric_cols_conv].mean().reset_index()
    conv_std = df_conv_all.groupby('Day')[numeric_cols_conv].std().reset_index()
    
    conv_mean.to_csv(results_dir / "ensemble_utility_convergence_mean.csv", index=False)
    conv_std.to_csv(results_dir / "ensemble_utility_convergence_std.csv", index=False)
    print("  Saved: ensemble_utility_convergence_mean.csv & ensemble_utility_convergence_std.csv")
    
    # 4.3 Retail Centre Performance Average
    perf_dfs = []
    for f in run_perf_files:
        df = pd.read_csv(f)
        # Ensure Retail_Centre is index/key
        df = df.set_index('Retail_Centre')
        perf_dfs.append(df)
        
    df_perf_concat = pd.concat(perf_dfs)
    perf_mean = df_perf_concat.groupby('Retail_Centre').mean()
    perf_std = df_perf_concat.groupby('Retail_Centre').std()
    
    perf_mean.to_csv(results_dir / "ensemble_centre_performance_mean.csv")
    perf_std.to_csv(results_dir / "ensemble_centre_performance_std.csv")
    print("  Saved: ensemble_centre_performance_mean.csv & ensemble_centre_performance_std.csv")
    
    # 5. Save the averaged performance file to outputs/centre_performance/ with a future timestamp
    # so that scripts like plot_hierarchy.py, plot_map.py, and run_pipeline.py pick it up as the latest file!
    avg_perf_path = outputs_root / "centre_performance" / "retail_centre_performance_99999999999999.csv"
    perf_mean.to_csv(avg_perf_path)
    print(f"  Saved averaged performance as latest anchor: {avg_perf_path.name}")
    
    print("\n" + "=" * 60)
    print("AGGREGATION AND OUTPUT ALIGNMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    aggregate_ensemble()
