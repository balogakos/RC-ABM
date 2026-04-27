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
    print("=== Calibrating Spatial Diffusion Threshold (Visits) ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # Values to test
    test_values = [1, 2, 5, 10, 20]
    results = []
    
    for val in test_values:
        print(f"Testing Visit Threshold: {val}...")
        visits = runner.run_experiment(
            days=20, 
            eval_freq=5, 
            param_overrides={'DIFFUSION_THRESHOLD_VISITS': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        metrics['threshold'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # "Best" selection logic: 
    # We want a threshold that isn't so low it triggers on noise, but not so high it never triggers.
    # Metric: Diversity vs Mean Utility. 
    # High thresholds might lead to stagnant utility, very low might lead to premature monopolies.
    df_res['score'] = df_res['mean_utility'] / (df_res['hhi'] + 1e-6)
    best_row = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\nRecommended Value: {best_row['threshold']}")
    
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "diffusion_threshold_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
