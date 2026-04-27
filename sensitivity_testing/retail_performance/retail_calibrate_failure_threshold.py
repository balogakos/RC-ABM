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
    print("=== Calibrating Retail Failure Threshold ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # Test different failure percentiles (0.05 = bottom 5%, 0.20 = bottom 20%)
    test_values = [0.05, 0.1, 0.15, 0.2]
    results = []
    
    for val in test_values:
        print(f"Testing Failure Threshold: {val}...")
        visits = runner.run_experiment(
            days=30, 
            eval_freq=10, 
            param_overrides={'RETAIL_FAILURE_THRESHOLD': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        # Custom metric: Count how many centres were "saved" (returned to top 50%)
        # For simplicity in this demo script, we use mean utility and diversity.
        metrics['failure_threshold'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # Selection logic: We want the threshold that maintains highest mean utility
    # without collapsing the HHI (indicating too many artificial rescues).
    df_res['score'] = df_res['mean_utility'] * (1 - df_res['hhi'])
    best_row = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\nRecommended Value: {best_row['failure_threshold']}")
    
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "retail_failure_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
