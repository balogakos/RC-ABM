import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure simulation directory is in path for config imports
SIM_DIR = ROOT_DIR / "simulation"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import config
from simulation.core import paths
from simulation.core.simulation_engine import SimulationEngine

# --- Ensemble Configuration ---
NUM_RUNS    = 5      # Number of iterations to average ensemble uncertainty
DAYS        = 90    # 90-day run — long enough for hierarchy to stabilise
EVAL_FREQ   = 30    # Match paper (Section 2.2): evaluate every 30 ticks
SAMPLE_SIZE = 60000  # Agents sub-sampled per run for memory efficiency

def _clean_rc_id(x):
    s = str(x)
    return s[:-2] if s.endswith('.0') else s

def load_simulation_data(n_agents: int = None, run_id: int = 1) -> tuple:
    """
    Standalone loader for ensemble runs using trip-specific files.
    Returns (consumers_df, utility_matrices, amenity_binary, tt_lookup).
    """
    utility_dir = config.PROCESSED_DIR
    print(f"Loading 6 trip-specific utility datasets from {utility_dir}...")
    
    trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
    suffixes = ['_walk', '_drive', '_pt']
    
    utility_matrices = {}
    consumers_df = None

    sampled_households = None
    if n_agents:
        first_file = os.path.join(utility_dir, f'utility_scores_{trip_types[0]}.parquet')
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"Required utility dataset missing: {first_file}")
        print("Identifying unique households for random sampling...")
        all_households = pd.read_parquet(first_file, columns=['household'])['household'].unique()
        if n_agents < len(all_households):
            # Dynamic seed per run to ensure we get different random samples for each run
            np.random.seed(42 + run_id)
            sampled_households = np.random.choice(all_households, size=n_agents, replace=False).tolist()
            print(f"Randomly selected {n_agents} households out of {len(all_households)} (using seed {42 + run_id}).")
        else:
            print(f"Requested agents {n_agents} is >= total unique households {len(all_households)}. Loading all agents.")

    for trip_type in trip_types:
        file_path = os.path.join(utility_dir, f'utility_scores_{trip_type}.parquet')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
            
        if sampled_households is not None:
            df = pd.read_parquet(file_path, filters=[('household', 'in', sampled_households)])
        else:
            df = pd.read_parquet(file_path)
            
        # Extract matrices
        for suf in suffixes:
            mode = suf.lstrip('_')
            cols = [c for c in df.columns if c.endswith(suf)]
            if not cols: continue
            
            mat = df[cols].astype(np.float16)
            mat.columns = [_clean_rc_id(c[:-len(suf)]) for c in mat.columns]
            
            # Ensure index is clean and unique to prevent row multiplication during reindex
            # Force to string for consistent indexing across different utility datasets
            mat.index = df['household'].astype(str)
            if not mat.index.is_unique:
                mat = mat[~mat.index.duplicated(keep='first')]
                
            utility_matrices[f'{trip_type}_{mode}'] = mat.fillna(0)

        if trip_type == 'bulk':
            meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in suffixes)]
            consumers_df = df[meta_cols].copy()
            # Pre-normalize grocery mode probabilities to align with main.py
            grocery_prob_cols = ['prob_online', 'prob_bulk', 'prob_convenience']
            if all(c in consumers_df.columns for c in grocery_prob_cols):
                row_sums = consumers_df[grocery_prob_cols].sum(axis=1).replace(0, 1.0)
                consumers_df[grocery_prob_cols] = consumers_df[grocery_prob_cols].div(row_sums, axis=0)

            # Consistent with matrices
            consumers_df['household'] = consumers_df['household'].astype(str)
            if not consumers_df['household'].is_unique:
                consumers_df = consumers_df.drop_duplicates(subset='household', keep='first')


    # Load Amenity Binary
    import geopandas as gpd
    gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
    gdf['RC_ID'] = gdf['RC_ID'].apply(_clean_rc_id)
    gdf = gdf.set_index('RC_ID')
    
    amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services', 'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
    amenity_binary = {col: (gdf[col] > 0).astype(float) for col in amenity_cols if col in gdf.columns}
    amenity_binary = pd.DataFrame(amenity_binary)

    # Load Transport Times
    tt_df = pd.read_parquet(config.TRANSPORT_TIMES_PATH)
    tt_lookup = {}
    for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
        if col in tt_df.columns:
            tt_lookup[mode] = {str(k): v for k, v in tt_df[col].items() if v is not None}

    return consumers_df, utility_matrices, amenity_binary, tt_lookup

def run_single_iteration(run_id):
    """Worker function for a single simulation run."""
    import time
    print(f"--- [Run {run_id}] Initializing process... ---")
    
    # Set seed for reproducibility of stochastic choices in this run
    np.random.seed(100 + run_id)
    
    # Load data locally in each process (safer for multiprocessing)
    # Always load from the main directory, but sample randomly using SAMPLE_SIZE
    test_mode = getattr(config, 'TEST_MODE', False)
    n_agents = 100 if test_mode else SAMPLE_SIZE
    consumers_df, utility_matrices, amenity_binary, tt_lookup = load_simulation_data(n_agents, run_id=run_id)
    
    # Thresholding: Zero out extremely low utilities to speed up softmax choice logic in-place on the underlying numpy array
    THRESHOLD = np.float16(0.1)
    for mat in utility_matrices.values():
        arr = mat.values
        arr[arr < THRESHOLD] = np.float16(0.0)
        
    engine = SimulationEngine(consumers_df, utility_matrices, amenity_binary, tt_lookup)
        
    results_dir = Path(ROOT_DIR) / "multi_runs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"--- [Run {run_id}] Simulation started... ---")
    
    summary_files = engine.run(
        num_agents=len(consumers_df), 
        days=DAYS, 
        eval_freq=EVAL_FREQ,
        output_mode="summary"
    )
    
    if summary_files:
        copied_files = []
        for src_path in summary_files:
            dest_path = results_dir / f"run_{run_id}_{src_path.name}"
            shutil.copy2(src_path, dest_path)
            copied_files.append(dest_path)

        elapsed = time.time() - start_time
        print(f"--- [Run {run_id}] COMPLETE in {elapsed:.1f}s. Saved summary outputs to multi_runs/results/ ---")
        for p in copied_files:
            print(f"     {p.name}")
        return f"Run {run_id} Success"
    else:
        print(f"--- [Run {run_id}] FAILED (No summary files generated) ---")
        return f"Run {run_id} No Data"

def run_ensemble():
    """Main entry point for sequential ensemble runs."""
    import time
    print(f"====================================================")
    print(f"RETAIL ABM ENSEMBLE RUNNER (Sequential Mode)")
    print(f"Total Runs: {NUM_RUNS} | Days: {DAYS}")
    print(f"====================================================\n")
    
    overall_start = time.time()
    run_ids = list(range(1, NUM_RUNS + 1))
    
    print(f"Starting Sequential Ensemble: {NUM_RUNS} runs...")
    results = []
    for rid in run_ids:
        res = run_single_iteration(rid)
        results.append(res)
        
    total_time = time.time() - overall_start
    print(f"\n====================================================")
    print(f"ENSEMBLE PIPELINE COMPLETE")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Results Summary: {results}")
    print(f"====================================================\n")

if __name__ == "__main__":
    run_ensemble()
