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
    print("=== Calibrating Retail Intervention Boost ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # Test multipliers (1.0 = no help, 1.2 = massive subsidy)
    test_values = [1.0, 1.05, 1.1, 1.2]
    results = []
    
    for val in test_values:
        print(f"Testing Boost Multiplier: {val}...")
        visits = runner.run_experiment(
            days=40, # Longer period to see recovery
            eval_freq=10, 
            param_overrides={'RETAIL_INTERVENTION_BOOST': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        metrics['boost_multiplier'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # Score logic: Maximize utility gain per unit of HHI increase.
    # We want centres to recover (higher utility) without creating monopolies (higher HHI).
    df_res['score'] = df_res['mean_utility'] / (df_res['hhi'] + 1e-6)
    best_row = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\nRecommended Value: {best_row['boost_multiplier']}")
    
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "retail_boost_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
