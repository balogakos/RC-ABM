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
    print("=== Calibrating Spatial Diffusion Boost Multiplier ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # Values to test
    test_values = [1.0, 1.02, 1.05, 1.1, 1.2, 1.5]
    results = []
    
    for val in test_values:
        print(f"Testing Boost Multiplier: {val}...")
        # Note: Since the core script currently has a default 1.05, 
        # we need to make sure the core script is updated to use config.
        # For now, we override it in the runner if we've updated spatial_diffusion.py.
        
        visits = runner.run_experiment(
            days=20, 
            eval_freq=5, 
            param_overrides={'DIFFUSION_BOOST_MULTIPLIER': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        metrics['multiplier'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # "Best" selection logic: 
    # Balance between high utility (agents happy) and reasonable HHI (not a monopoly)
    # Score = Mean_Utility * (1 - HHI)
    df_res['score'] = df_res['mean_utility'] * (1 - df_res['hhi'])
    best_row = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\nRecommended Value: {best_row['multiplier']}")
    print(f"Rationale: This value achieved a balance of {best_row['mean_utility']:.3f} mean utility "
          f"while maintaining a diversity (1-HHI) of {1-best_row['hhi']:.3f}.")
    
    # Save results
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "diffusion_multiplier_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
