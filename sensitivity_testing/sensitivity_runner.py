import os
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import geopandas as gpd
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
for p in [ROOT_DIR, ROOT_DIR / "simulation"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from simulation.core import paths
from simulation.core.simulation_engine import SimulationEngine

class SensitivityRunner:
    def __init__(self, n_agents=2000):
        self.n_agents = n_agents
        self.base_data = self._load_data(n_agents)
        
    def _load_data(self, n_agents):
        """Replicates data loading from main.py but in a headless way."""
        utility_matrices = {}
        trip_types = ['bulk', 'convenience', 'comparison', 'entertainment', 'food_drink', 'service']
        base_utility_dir = Path(config.PROCESSED_DIR)
        
        # 1. Load Utilities
        consumers = None
        for trip_type in trip_types:
            file_path = base_utility_dir / f'utility_scores_{trip_type}.parquet'
            _pf = pq.ParquetFile(file_path)
            df = next(_pf.iter_batches(batch_size=n_agents)).to_pandas()
            
            # Simple splitter (replicated from main.py)
            meta_cols = [c for c in df.columns if not any(c.endswith(s) for s in ['_walk', '_drive', '_pt'])]
            if trip_type == 'bulk':
                consumers = df[meta_cols].copy()
            
            for mode in ['walk', 'drive', 'pt']:
                cols = [c for c in df.columns if c.endswith(f'_{mode}')]
                if cols:
                    mat = df[cols].astype(np.float32)
                    mat.columns = [c[:-len(f'_{mode}')].replace('.0','') for c in mat.columns]
                    mat.index = df['household']
                    utility_matrices[f'{trip_type}_{mode}'] = mat.fillna(0)
        
        # 2. Load Amenities
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        gdf['RC_ID'] = gdf['RC_ID'].astype(str).str.replace('.0', '', regex=False)
        gdf = gdf.set_index('RC_ID')
        amenity_cols = ['Foodstore', 'Convenience Store', 'Retail', 'Cafe', 'Restaurant', 'Entertainment']
        amenity_binary = {col: (gdf[col] > 0).astype(float) for col in amenity_cols if col in gdf.columns}
        
        # 3. Load Transport
        tt_lookup = {}
        if os.path.exists(config.TRANSPORT_TIMES_PATH):
            df_trans = pd.read_parquet(config.TRANSPORT_TIMES_PATH)
            if 'Postcode' in df_trans.columns:
                df_trans = df_trans.set_index('Postcode')
            for col, mode in [('Walk', 'walk'), ('Drive', 'drive'), ('PT', 'pt')]:
                if col in df_trans.columns:
                    tt_lookup[mode] = df_trans[col].to_dict()

        return consumers, utility_matrices, amenity_binary, tt_lookup

    def run_experiment(self, days=20, eval_freq=5, param_overrides=None):
        """Runs a simulation with specific config overrides."""
        if param_overrides:
            for key, val in param_overrides.items():
                setattr(config, key, val)
        
        consumers, base_matrices, amenity_binary, tt_lookup = self.base_data
        # Deep copy matrices to avoid cross-experiment contamination
        utility_matrices = {k: v.copy() for k, v in base_matrices.items()}
        
        engine = SimulationEngine(consumers, utility_matrices, amenity_binary, tt_lookup)
        all_visits = engine.run(self.n_agents, days, eval_freq)
        
        if not all_visits:
            return pd.DataFrame()
            
        return pd.concat(all_visits, ignore_index=True)

    @staticmethod
    def calculate_metrics(visits_df):
        """Calculates performance metrics for calibration."""
        if visits_df.empty:
            return {"hhi": 1.0, "diversity": 0.0, "avg_utility": 0.0}
            
        # 1. HHI (Concentration) - Lower is better for diversity
        counts = visits_df['Retail_Centre'].value_counts()
        shares = counts / counts.sum()
        hhi = (shares**2).sum()
        
        # 2. Diversity Score (number of unique centres visited)
        unique_centres = visits_df['Retail_Centre'].nunique()
        
        # 3. Mean Utility
        mean_util = visits_df['Utility_Score'].mean()
        
        return {
            "hhi": hhi,
            "diversity": unique_centres,
            "mean_utility": mean_util,
            "total_visits": len(visits_df)
        }
