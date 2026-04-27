import os
import sys
from pathlib import Path

# Fix ModuleNotFoundError: Ensure the project root is in sys.path
# This allows 'import simulation.core' to work when running main.py directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd

# Modular Imports
import config
from simulation.core import paths
from simulation.core.simulation_engine import SimulationEngine
from simulation.core.constants import TRANSPORT_MODES
import visualization

# --- Configuration Override ---
TEST_MODE = False

def _clean_rc_id(x):
    try:
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
    except Exception:
        return str(x)

def _preprocess_transport_times(df):
    if df is None or (hasattr(df, 'empty') and df.empty):
        return {}
    tt_lookup = {}
    for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
        if col not in df.columns:
            continue
        d = {}
        for postcode, val in df[col].items():
            if isinstance(val, dict):
                d[postcode] = {_clean_rc_id(k): float(v) for k, v in val.items()}
            elif val is not None:
                try:
                    d[postcode] = float(val)
                except Exception:
                    pass
        tt_lookup[mode] = d
    return tt_lookup

def _split_matrices(df, household_col='household'):
    transport_suffixes = ['_walk', '_drive', '_pt']
    mode_cols = {suf.lstrip('_'): [] for suf in transport_suffixes}
    meta_cols = []

    for col in df.columns:
        matched = False
        for suf in transport_suffixes:
            if col.endswith(suf):
                prefix = col[:-len(suf)]
                try:
                    float(prefix)
                    mode_cols[suf.lstrip('_')].append(col)
                    matched = True
                    break
                except (ValueError, TypeError):
                    pass
        
        if not matched:
            try:
                float(col)
                mode_cols['drive'].append(col)
                matched = True
            except (ValueError, TypeError):
                pass
                
        if not matched:
            meta_cols.append(col)

    meta_df = df[meta_cols].copy()
    mode_dfs = {}
    for mode, cols in mode_cols.items():
        if not cols:
            continue
        mat = df[cols].astype(np.float32)
        def _strip(c):
            for s in transport_suffixes:
                if c.endswith(s):
                    c = c[:-len(s)]
                    break
            return c[:-2] if c.endswith('.0') else c
        mat.columns = [_strip(c) for c in mat.columns]
        if household_col in df.columns:
            mat.index = df[household_col]
        mode_dfs[mode] = mat.fillna(0)

    return meta_df, mode_dfs

class RetailABMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail ABM Configuration")
        self.root.geometry("450x370")

        self._base_data = None
        self._loaded_n = None

        self.status_var = tk.StringVar(value="Configure settings and click Run Simulation.")
        ttk.Label(root, textvariable=self.status_var, wraplength=420).pack(pady=10)

        ttk.Label(root, text="Number of Agents to Simulate:").pack(pady=5)
        self.num_agents_var = tk.StringVar(value="1000")
        ttk.Entry(root, textvariable=self.num_agents_var).pack(pady=5)

        ttk.Label(root, text="Number of Days to Simulate:").pack(pady=5)
        self.days_var = tk.StringVar(value="30")
        ttk.Entry(root, textvariable=self.days_var).pack(pady=5)

        ttk.Label(root, text="Evaluation Frequency (Days):").pack(pady=5)
        self.eval_freq_var = tk.StringVar(value="10")
        ttk.Entry(root, textvariable=self.eval_freq_var).pack(pady=5)

        ttk.Button(root, text="Run Simulation", command=self.run_simulation).pack(pady=20)

    def log(self, message):
        print(message)
        self.status_var.set(message)
        self.root.update()

    def load_data(self, n_agents=None):
        utility_matrices = {}
        consumers = None
        trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
        
        base_utility_dir = paths.UTILITY_DIR
        if TEST_MODE:
            base_utility_dir = base_utility_dir / "testing"
            
        self.log(f"Loading utility datasets from {base_utility_dir}...")

        for trip_type in trip_types:
            file_path = base_utility_dir / f'utility_scores_{trip_type}.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
                
            if n_agents:
                _pf = pq.ParquetFile(file_path)
                df = next(_pf.iter_batches(batch_size=n_agents)).to_pandas()
            else:
                df = pd.read_parquet(file_path)
                
            meta_df, mode_dfs = _split_matrices(df)
            if trip_type == 'bulk':
                consumers = meta_df
                
            for tmode, mat in mode_dfs.items():
                utility_matrices[f'{trip_type}_{tmode}'] = mat
                
            import gc
            del df
            gc.collect()

        self.log("Loading retail centre amenity data...")
        gdf = gpd.read_file(paths.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        gdf['RC_ID'] = gdf['RC_ID'].apply(_clean_rc_id)
        gdf = gdf.set_index('RC_ID')

        amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services', 'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
        amenity_binary = {col: (gdf[col] > 0).astype(float) for col in amenity_cols if col in gdf.columns}

        self.log("Loading transport times lookup table...")
        transport_times = pd.DataFrame()
        if paths.TRANSPORT_TIMES_PATH.exists():
            transport_times = pd.read_parquet(paths.TRANSPORT_TIMES_PATH)
            if 'Postcode' in transport_times.columns:
                transport_times = transport_times.set_index('Postcode')

        tt_lookup = _preprocess_transport_times(transport_times)
        return consumers, utility_matrices, amenity_binary, tt_lookup

    def run_simulation(self):
        try:
            num_agents_req = int(self.num_agents_var.get())
            days = int(self.days_var.get())
            eval_freq = int(self.eval_freq_var.get())

            if self._base_data is None or self._loaded_n != num_agents_req:
                raw = self.load_data(n_agents=num_agents_req)
                self._base_data = raw
                self._loaded_n = num_agents_req

            consumers, base_matrices, amenity_binary, tt_lookup = self._base_data
            utility_matrices = {k: v.copy() for k, v in base_matrices.items()}

            # Thresholding
            THRESHOLD = 0.1
            for mat in utility_matrices.values():
                mat[mat < THRESHOLD] = 0.0

            # Run Engine
            engine = SimulationEngine(consumers, utility_matrices, amenity_binary, tt_lookup)
            all_visits = engine.run(num_agents_req, days, eval_freq, log_callback=self.log)

            # Results
            if all_visits:
                self.log("Saving results...")
                paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                total_visits_df = pd.concat(all_visits, ignore_index=True)
                output_path = paths.OUTPUT_DIR / f"visits_log_{int(time.time())}.parquet"
                total_visits_df.to_parquet(output_path)

                map_path = visualization.plot_visitation_map(str(output_path))
                messagebox.showinfo("Success", f"Simulation complete!\n{len(total_visits_df):,} visits saved.")
            else:
                messagebox.showwarning("Info", "No visits occurred.")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"{e}\n\n{traceback.format_exc()}")

def main():
    paths.ensure_dirs()
    root = tk.Tk()
    RetailABMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
