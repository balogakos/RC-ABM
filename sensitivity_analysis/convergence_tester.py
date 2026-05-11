import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure simulation directory is in path for 'import config'
SIM_DIR = ROOT_DIR / "simulation"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import simulation.config as config
from simulation.core.simulation_engine import SimulationEngine
from multi_runs.runner import load_simulation_data

def run_convergence_analysis(n_agents=60000, total_days=600, eval_freq=30):
    """
    Runs a long-term simulation (600 days) to monitor the steady-state
    convergence of retail market behaviors.
    """
    results_dir = ROOT_DIR / "sensitivity_analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Sensitivity Analysis: 600-Day Convergence Test ---")
    print(f"Agents: {n_agents} | Target Days: {total_days} | Frequency: {eval_freq} days")
    
    # 1. Load Data
    consumers_df, utility_matrices, amenity_binary, tt_lookup = load_simulation_data(n_agents)
    engine = SimulationEngine(consumers_df, utility_matrices, amenity_binary, tt_lookup)
    
    # Trackers for behavior
    history = []
    prev_market_share = None
    prev_mode_share = None
    
    start_time = time.time()
    
    # 2. Run Simulation in chunks to measure stability
    for period in range(0, total_days, eval_freq):
        current_day_start = period + 1
        current_day_end = period + eval_freq
        
        print(f"\nDay {current_day_start} to {current_day_end}:")
        
        # Run for the period
        period_visits = engine.run(
            num_agents=n_agents,
            days=eval_freq,
            eval_freq=eval_freq
        )
        
        # 3. Analyze the output of this period
        if not period_visits:
            print("  Error: No visits generated in this period.")
            continue
            
        # Read the latest temp parquet
        df = pd.read_parquet(period_visits[0])
        
        # --- BEHAVIOR 1: Retail Market Share Stability ---
        market_counts = df['Retail_Centre'].value_counts(normalize=True)
        market_delta = 0
        if prev_market_share is not None:
            # Align indices and calculate absolute difference in share
            combined = pd.concat([prev_market_share, market_counts], axis=1).fillna(0)
            market_delta = np.abs(combined.iloc[:, 0] - combined.iloc[:, 1]).sum()
        prev_market_share = market_counts

        # --- BEHAVIOR 2: Transport Mode Stability ---
        mode_counts = df['Transport_Mode'].value_counts(normalize=True)
        mode_delta = 0
        if prev_mode_share is not None:
            combined_mode = pd.concat([prev_mode_share, mode_counts], axis=1).fillna(0)
            mode_delta = np.abs(combined_mode.iloc[:, 0] - combined_mode.iloc[:, 1]).sum()
        prev_mode_share = mode_counts

        # --- BEHAVIOR 3: Geodemographic Response ---
        # Measure the average utility score per subcluster to see if it's stabilising
        subcluster_stats = df.groupby('Geo_Subcluster')['Utility_Score'].mean()
        
        # 4. Record Metrics
        metrics = {
            'Period_End_Day': current_day_end,
            'Market_Share_Delta': market_delta,
            'Mode_Share_Delta': mode_delta,
            'Avg_Utility': df['Utility_Score'].mean(),
            'Total_Visits': len(df)
        }
        history.append(metrics)
        
        print(f"  Market Churn: {market_delta:.4f} | Mode Churn: {mode_delta:.4f} | Avg Utility: {metrics['Avg_Utility']:.3f}")
        
        # Clean up temp files
        for f in period_visits:
            f.unlink()
        period_visits[0].parent.rmdir()

    # 5. Final Report
    history_df = pd.DataFrame(history)
    output_path = results_dir / "convergence_metrics.csv"
    history_df.to_csv(output_path, index=False)
    
    print(f"\n--- Convergence Analysis Complete ---")
    print(f"Results saved to: {output_path}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    
    # Recommendation logic
    stable_period = history_df[history_df['Market_Share_Delta'] < 0.05]
    if not stable_period.empty:
        ideal_days = stable_period.iloc[0]['Period_End_Day']
        print(f"RECOMMENDATION: Model reaches behavioral stability around Day {ideal_days}.")
    else:
        print("RECOMMENDATION: Model has not fully converged. Consider running for more days.")

if __name__ == "__main__":
    # Run with 60k agents as suggested for long-term stability
    run_convergence_analysis(n_agents=60000, total_days=600, eval_freq=30)
