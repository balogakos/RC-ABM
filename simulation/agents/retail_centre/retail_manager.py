import geopandas as gpd
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
        
        # Performance state
        self.underperformer_tracker = {}
        self.cumulative_boosts = {}

    def evaluate_centres(self, visits_df, utility_matrices):
        """
        Runs the performance tracking and intervention policy.
        """
        # 1. Rank centres
        failed, participating = performance_tracker.rank_retail_centres(
            visits_df, self.retail_gdf, self.amenity_binary
        )
        
        # 2. Apply interventions
        messages = intervention_system.apply_intervention_policy(
            failed, participating, utility_matrices, 
            self.underperformer_tracker, self.cumulative_boosts
        )
        
        return messages
