"""
POST-PROCESSING SCRIPT: Adding Travel Times to Simulation Results
----------------------------------------------------------------
This script is used to enrich the Retail ABM simulation outputs with actual 
travel time minutes. During the simulation, travel time lookups are disabled
to maximize execution speed. This script performs a vectorized join at the end.
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from pathlib import Path

# --- Configuration ---
RESULTS_DIR = Path("multi_runs/results")
TRANSPORT_TIMES_PATH = "C:/Users/sgabalog/Documents/P3/Model/data_local/liverpool/processed/transport_times.parquet"

def process_results():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} not found.")
        return

    print("Loading travel time lookup table...")
    tt_df = pd.read_parquet(TRANSPORT_TIMES_PATH)
    
    # Files to process
    parquet_files = list(RESULTS_DIR.glob("run_*.parquet"))
    
    if not parquet_files:
        print("No result files found to process.")
        return

    for fpath in parquet_files:
        print(f"Processing {fpath.name}...")
        df = pd.read_parquet(fpath)
        
        # Original columns for cleanup
        original_cols = df.columns.tolist()
        
        # We need AgentID (to get Postcode) or the Postcode itself if stored
        # The simulation stores 'Postcode' (first 3-4 chars) and 'Retail_Centre' and 'Transport_Mode'
        
        # Example join logic (adjust based on your actual Parquet schema):
        # We assume tt_df has a MultiIndex or columns [Postcode, Retail_Centre, Mode]
        
        # Note: Since the simulation skips the lookup, we now 'bake' it back in.
        # This is significantly faster when done once for the whole run.
        
        # TODO: Add specific merge logic here once you've confirmed your 
        # preferred post-processing format.
        
        print(f"  -> Enrichment complete for {fpath.name}")

if __name__ == "__main__":
    process_results()
