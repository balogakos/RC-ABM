import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from simulation.core import paths
from simulation.core.simulation_engine import SimulationEngine

# --- Configuration ---
TEST_MODE = True  # Set to False for FULL simulation (656k agents)
NUM_RUNS  = 5
DAYS      = 30
EVAL_FREQ = 10

def _clean_rc_id(x):
    s = str(x)
    return s[:-2] if s.endswith('.0') else s

def load_simulation_data(n_agents=None):
    """Standalone loader for ensemble runs."""
    print(f"Loading utility datasets (Test Mode: {TEST_MODE})...")
    
    trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
    base_dir = paths.UTILITY_DIR
    if TEST_MODE:
        base_dir = base_dir / "testing"
    
    utility_matrices = {}
    consumers_df = None

    for trip_type in trip_types:
        file_path = base_dir / f'utility_scores_{trip_type}.parquet'
        df = pd.read_parquet(file_path)
        
        if n_agents and len(df) > n_agents:
            df = df.sample(n=n_agents, random_state=42)
            
        # Extract matrices
        suffixes = ['_walk', '_drive', '_pt']
        for suf in suffixes:
            mode = suf.lstrip('_')
            cols = [c for c in df.columns if c.endswith(suf)]
            if not cols: continue
            
            mat = df[cols].astype(np.float32)
            mat.columns = [_clean_rc_id(c[:-len(suf)]) for c in mat.columns]
            mat.index = df['household']
            
            key = f"{trip_type}_{mode}"
            utility_matrices[key] = mat.fillna(0)

        if consumers_df is None:
            meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in suffixes)]
            consumers_df = df[meta_cols].copy()

    # Load Amenity Binary
    amenity_path = base_dir / 'retail_centre_amenity_binary.parquet'
    amenity_binary = pd.read_parquet(amenity_path)
    amenity_binary.index = amenity_binary.index.astype(str)

    # Load Transport Times
    tt_df = pd.read_parquet(config.TRANSPORT_TIMES_PATH)
    tt_lookup = {}
    for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
        if col in tt_df.columns:
            tt_lookup[mode] = {str(k): v for k, v in tt_df[col].items() if v is not None}

    return consumers_df, utility_matrices, amenity_binary, tt_lookup

if __name__ == "__main__":
    # Test the loader
    n = 100 if TEST_MODE else None
    data = load_simulation_data(n_agents=n)
    print("Data loaded successfully.")
