import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from .behaviours import performance_tracker, intervention_system
from simulation.core.paths import RETAIL_CENTRES_GPKG

def clean_rc_id(x):
    try:
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
    except Exception:
        return str(x)

class RetailManager:
    """
    Manages the retail centres, tracking their performance and applying interventions.
    """
    def __init__(self, amenity_binary):
        # Load retail centres GeoDataFrame
        self.retail_gdf = gpd.read_file(RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        self.retail_gdf['RC_ID'] = self.retail_gdf['RC_ID'].apply(clean_rc_id)
        self.retail_gdf = self.retail_gdf.set_index('RC_ID')

        self.amenity_binary = amenity_binary

        # --- Pre-compute spatial and size lookups (done once, reused every evaluation) ---
        # Full pairwise distance matrix (metres) — eliminates O(N²) per-evaluation distance calls
        centroids = np.array([
            (geom.centroid.x, geom.centroid.y)
            for geom in self.retail_gdf.geometry
        ])
        self._dist_matrix = cdist(centroids, centroids)  # (N, N) in metres
        self._centre_list = list(self.retail_gdf.index)  # ordered list matching dist_matrix rows
        self._centre_pos  = {rc: i for i, rc in enumerate(self._centre_list)}

        # POI sort index for fast size-peer lookup via searchsorted
        poi_col = 'Total_POI_' if 'Total_POI_' in self.retail_gdf.columns else None
        if poi_col:
            self._poi_series = self.retail_gdf[poi_col].fillna(0)
        else:
            amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services',
                            'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
            existing = [c for c in amenity_cols if c in self.retail_gdf.columns]
            self._poi_series = self.retail_gdf[existing].sum(axis=1)
        self._poi_sorted_idx = np.argsort(self._poi_series.values)  # ascending
        self._poi_sorted_vals = self._poi_series.values[self._poi_sorted_idx]
        self._poi_sorted_ids  = np.array(self._centre_list)[self._poi_sorted_idx]

        # Performance state
        self.underperformer_tracker = {}
        self.cumulative_boosts = {}
        self.death_spirals = {}  # Track permanent decline/death spiral state

    def evaluate_centres(self, visits_df, utility_matrices):
        """
        Runs the performance tracking and intervention policy.
        """
        # 1. Rank centres
        failed, participating = performance_tracker.rank_retail_centres(
            visits_df, self.retail_gdf, self.amenity_binary,
            self._dist_matrix, self._centre_list, self._centre_pos,
            self._poi_series, self._poi_sorted_idx,
            self._poi_sorted_vals, self._poi_sorted_ids
        )

        # 2. Apply interventions (including death spiral logic)
        messages = intervention_system.apply_intervention_policy(
            failed, participating, utility_matrices,
            self.underperformer_tracker, self.cumulative_boosts, self.death_spirals
        )

        return messages
