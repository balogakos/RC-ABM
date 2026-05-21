import os
import sys
from pathlib import Path

# Fix ModuleNotFoundError: Ensure the project root is in sys.path
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

def _clean_rc_id(x):
    try:
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
    except Exception:
        return str(x)

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
        base_utility_dir = config.UTILITY_DIR
        self.log("Loading 6 trip-specific datasets...")
        
        utility_matrices = {}
        consumers = None
        trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
            
        for trip_type in trip_types:
            file_path = os.path.join(base_utility_dir, f'utility_scores_{trip_type}.parquet')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required utility dataset missing: {file_path}")
            
            if n_agents:
                _pf = pq.ParquetFile(file_path)
                df = next(_pf.iter_batches(batch_size=n_agents)).to_pandas()
                df = df.iloc[:n_agents]
            else:
                df = pd.read_parquet(file_path)
            
            utility_cols = [c for c in df.columns if any(c.endswith(s) for s in ['_walk', '_drive', '_pt'])]
            if utility_cols:
                df[utility_cols] = df[utility_cols].astype(np.float16)
                
            suffixes = ['_walk', '_drive', '_pt']
            for suf in suffixes:
                mode = suf.lstrip('_')
                cols = [c for c in df.columns if c.endswith(suf)]
                if not cols: continue
                
                mat = df[cols]
                mat.columns = [_clean_rc_id(c[:-len(suf)]) for c in mat.columns]
                mat.index = df['household'].astype(str)
                if not mat.index.is_unique:
                    mat = mat[~mat.index.duplicated(keep='first')]
                utility_matrices[f'{trip_type}_{mode}'] = mat.fillna(0).astype(np.float16)

            if trip_type == 'bulk':
                meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in suffixes)]
                consumers = df[meta_cols].copy()
                
                # CRITICAL FIX: Pre-normalize grocery mode probabilities
                grocery_prob_cols = ['prob_online', 'prob_bulk', 'prob_convenience']
                if all(c in consumers.columns for c in grocery_prob_cols):
                    row_sums = consumers[grocery_prob_cols].sum(axis=1).replace(0, 1.0)
                    consumers[grocery_prob_cols] = consumers[grocery_prob_cols].div(row_sums, axis=0)

                consumers['household'] = consumers['household'].astype(str)
                if not consumers['household'].is_unique:
                    consumers = consumers.drop_duplicates(subset='household', keep='first')

        self.log("Loading retail centre amenity data...")
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        gdf['RC_ID'] = gdf['RC_ID'].apply(_clean_rc_id)
        gdf = gdf.set_index('RC_ID')
        amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services',
                        'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
        amenity_binary = {col: (gdf[col] > 0).astype(float) for col in amenity_cols if col in gdf.columns}
        amenity_binary = pd.DataFrame(amenity_binary)

        self.log("Loading transport times...")
        tt_lookup = {}
        tt_path = getattr(config, 'TRANSPORT_TIMES_PATH', '')
        if tt_path and os.path.exists(tt_path):
            tt_df = pd.read_parquet(tt_path)
            for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
                if col in tt_df.columns:
                    tt_lookup[mode] = {str(k): v for k, v in tt_df[col].items() if v is not None}

        return consumers, utility_matrices, amenity_binary, tt_lookup

    def run_simulation(self):
        try:
            num_agents_req = int(self.num_agents_var.get())
            days = int(self.days_var.get())
            eval_freq = int(self.eval_freq_var.get())

            self.log(f"Initializing simulation for {num_agents_req} agents...")
            raw = self.load_data(n_agents=num_agents_req)
            self._base_data = raw

            consumers, base_matrices, amenity_binary, tt_lookup = self._base_data
            utility_matrices = {k: v.copy() for k, v in base_matrices.items()}
            
            THRESHOLD = np.float16(0.1)
            for mat in utility_matrices.values():
                # Perform thresholding in-place on the underlying numpy array to prevent upcasting
                arr = mat.values
                arr[arr < THRESHOLD] = np.float16(0.0)

            engine = SimulationEngine(consumers, utility_matrices, amenity_binary, tt_lookup)
            summary_files = engine.run(num_agents_req, days, eval_freq, log_callback=self.log)

            if summary_files and len(summary_files) >= 2:
                daily_csv, centre_csv = summary_files[:2]
                self.log("Simulation complete!")
                msg = f"Simulation finished successfully.\n\n" \
                      f"Daily Summary: {os.path.basename(daily_csv)}\n" \
                      f"Centre Performance: {os.path.basename(centre_csv)}"
                if len(summary_files) == 3:
                    convergence_csv = summary_files[2]
                    msg += f"\nUtility Convergence & Distributions: {os.path.basename(convergence_csv)}"
                elif len(summary_files) == 4:
                    convergence_csv, distribution_csv = summary_files[2:4]
                    msg += f"\nUtility Convergence: {os.path.basename(convergence_csv)}" \
                           f"\nUtility Distributions: {os.path.basename(distribution_csv)}"
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showwarning("Info", "No visits occurred or summaries failed.")

        except Exception as e:
            import traceback
            self.log(f"Error: {e}")
            messagebox.showerror("Error", f"{e}\n\n{traceback.format_exc()}")

def main():
    paths.ensure_dirs()
    root = tk.Tk()
    RetailABMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
