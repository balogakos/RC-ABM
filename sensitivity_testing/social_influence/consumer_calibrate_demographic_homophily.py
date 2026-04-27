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
    print("=== Calibrating Demographic Diffusion Weight (Homophily) ===")
    runner = SensitivityRunner(n_agents=2000)
    
    # 0.0 = pure spatial (neighbors), 1.0 = pure demographic (people like me)
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for val in test_values:
        print(f"Testing Demographic Weight: {val}...")
        visits = runner.run_experiment(
            days=20, 
            eval_freq=5, 
            param_overrides={'DEMOGRAPHIC_DIFFUSION_WEIGHT': val}
        )
        
        metrics = runner.calculate_metrics(visits)
        metrics['demographic_weight'] = val
        results.append(metrics)
    
    df_res = pd.DataFrame(results)
    print("\nResults:")
    print(df_res)
    
    # Recommendation: Usually a mid-range value (0.5) is most realistic.
    # We look for where mean utility is maximized while maintaining trip diversity.
    df_res['score'] = df_res['mean_utility'] * df_res['diversity']
    best_row = df_res.loc[df_res['score'].idxmax()]
    
    print(f"\nRecommended Value: {best_row['demographic_weight']}")
    
    output_file = Path(ROOT_DIR) / "outputs" / "sensitivity" / "homophily_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_file, index=False)
    print(f"Full results saved to {output_file}")

if __name__ == "__main__":
    run_calibration()
