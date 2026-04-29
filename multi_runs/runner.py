import os
import sys
import pandas as pd
import numpy as np
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

# --- Configuration ---
TEST_MODE = True    # SET THIS TO False FOR FINAL PAPER RESULTS (656k agents)
NUM_RUNS  = 5       # Number of iterations to average out uncertainty
DAYS      = 30
EVAL_FREQ = 10
PARALLEL  = True    # Run multiple simulations at once
NUM_CORES = 4       # How many CPU cores to use (Check your RAM!)

def _clean_rc_id(x):
    s = str(x)
    return s[:-2] if s.endswith('.0') else s

def load_simulation_data(n_agents=None):
    """Standalone loader for ensemble runs using trip-specific files."""
    print(f"Loading 6 trip-specific utility datasets from {config.UTILITY_DIR}...")
    
    trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
    suffixes = ['_walk', '_drive', '_pt']
    
    utility_matrices = {}
    consumers_df = None

    for trip_type in trip_types:
        file_path = os.path.join(config.UTILITY_DIR, f'utility_scores_{trip_type}.parquet')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
            
        df = pd.read_parquet(file_path)
        if n_agents and len(df) > n_agents:
            df = df.sample(n=n_agents, random_state=42)
            
        # Extract matrices
        for suf in suffixes:
            mode = suf.lstrip('_')
            cols = [c for c in df.columns if c.endswith(suf)]
            if not cols: continue
            
            mat = df[cols].astype(np.float32)
            mat.columns = [_clean_rc_id(c[:-len(suf)]) for c in mat.columns]
            
            # CRITICAL FIX: Ensure index is unique to prevent reindex row multiplication
            mat.index = df['household']
            if not mat.index.is_unique:
                mat = mat[~mat.index.duplicated(keep='first')]
                
            utility_matrices[f'{trip_type}_{mode}'] = mat.fillna(0)

        if trip_type == 'bulk':
            meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in suffixes)]
            consumers_df = df[meta_cols].copy()
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
    
    # Load data locally in each process (safer for multiprocessing)
    n_agents = 100 if TEST_MODE else None
    consumers_df, utility_matrices, amenity_binary, tt_lookup = load_simulation_data(n_agents)
    
    engine = SimulationEngine(consumers_df, utility_matrices, amenity_binary, tt_lookup)
    results_dir = Path(ROOT_DIR) / "multi_runs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"--- [Run {run_id}] Simulation started... ---")
    
    visits = engine.run(
        num_agents=len(consumers_df), 
        days=DAYS, 
        eval_freq=EVAL_FREQ
    )
    
    if visits:
        df_visits = pd.concat(visits, ignore_index=True)
        output_path = results_dir / f"run_{run_id}.parquet"
        df_visits.to_parquet(output_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"--- [Run {run_id}] COMPLETE in {elapsed:.1f}s. Saved to {output_path.name} ---")
        return f"Run {run_id} Success"
    else:
        print(f"--- [Run {run_id}] FAILED (No visits generated) ---")
        return f"Run {run_id} No Data"

def run_ensemble():
    """Main execution entry point."""
    import time
    from multiprocessing import Pool
    
    overall_start = time.time()
    run_ids = list(range(1, NUM_RUNS + 1))
    
    if PARALLEL:
        print(f"Starting Parallel Ensemble: {NUM_RUNS} runs on {NUM_CORES} cores...")
        with Pool(processes=NUM_CORES) as pool:
            results = pool.map(run_single_iteration, run_ids)
    else:
        print(f"Starting Sequential Ensemble: {NUM_RUNS} runs...")
        results = [run_single_iteration(rid) for rid in run_ids]
        
    total_time = time.time() - overall_start
    print(f"\n====================================================")
    print(f"ENSEMBLE PIPELINE COMPLETE")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Results Summary: {results}")
    print(f"====================================================\n")

if __name__ == "__main__":
    # Multiprocessing on Windows requires this check
    run_ensemble()
