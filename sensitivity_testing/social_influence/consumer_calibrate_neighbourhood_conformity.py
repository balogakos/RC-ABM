import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sensitivity_testing.sensitivity_runner import SensitivityRunner

def run_calibration():
    print("=== Calibrating Neighbourhood Conformity ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # Values to test (0.0 = no influence, 1.0 = total conformity)
    test_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    results = []
    
    for val in test_values:
        print(f"Testing Conformity Strength: {val}...")
        visits = runner.run_experiment(
            days=20, 
            eval_freq=5, 
            param_overrides={'NEIGHBOURHOOD_CONFORMITY': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        metrics['conformity'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # Rationale: Higher conformity usually improves mean utility (agents picking the "best" local option)
    # but collapses HHI. We want the highest value where HHI stays below a threshold (e.g. 0.4).
    mask = df_res['hhi'] < 0.4
    if mask.any():
        best_row = df_res[mask].sort_values('mean_utility', ascending=False).iloc[0]
    else:
        best_row = df_res.sort_values('hhi', ascending=True).iloc[0]
        
    print(f"\nRecommended Value: {best_row['conformity']}")
    
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "conformity_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
