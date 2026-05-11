import sys
import os
import pandas as pd
import numpy as np
import config
from .behaviours import consumption, shopping_logic, travel_choice, spatial_diffusion

# Geodemographic subcluster integration
_RETAIL_ABM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _RETAIL_ABM_ROOT not in sys.path:
    sys.path.insert(0, _RETAIL_ABM_ROOT)
try:
    from data_preprocessing.geodemographic.subcluster_priors import (
        assign_subcluster, apply_cluster_blend, ALL_SUBCLUSTERS
    )
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

class ConsumerPopulation:
    """
    Manages the state and behaviours of the entire consumer population.
    Uses vectorized operations on internal DataFrames for high performance.
    """
    def __init__(self, num_agents: int, attributes_df: pd.DataFrame):
        self.num_agents = num_agents
        # Initialize attributes
        if num_agents <= len(attributes_df):
            self.attributes = attributes_df.sample(n=num_agents, replace=False).reset_index(drop=True)
        else:
            self.attributes = attributes_df.sample(n=num_agents, replace=True).reset_index(drop=True)

        # Initialize dynamic state from attributes_df columns
        # Note: stock_level from data is used as Capacity. 
        # Initial Stock is randomized between 0 and Capacity.
        self.state_df = pd.DataFrame({
            'AgentID':            self.attributes['household'].values if 'household' in self.attributes.columns else range(num_agents),
            'Postcode':           self.attributes['Postcode'].values if 'Postcode' in self.attributes.columns else 'UNKNOWN',
            'Capacity':           self.attributes['stock_level'].values if 'stock_level' in self.attributes.columns else 100.0,
            'Consumption_Rate':   self.attributes['consumption_rate'].values if 'consumption_rate' in self.attributes.columns else 10.0,
            'Shopping_Threshold': self.attributes['shopping_threshold'].values if 'shopping_threshold' in self.attributes.columns else 20.0,
            'Last_Shop_Day':      -1,
            'Shopping_Mode':      None,
        })
        self.state_df.index = range(num_agents)  # Force a clean, unique range index

        # --- New Social Influence Traits ---
        c_min, c_max = getattr(config, 'CONFORMITY_RANGE', [0.1, 0.5])
        b_min, b_max = getattr(config, 'SOCIAL_OPENNESS_RANGE', [0.1, 1.0])
        
        self.state_df['Conformity_Coefficient'] = np.random.uniform(c_min, c_max, num_agents)
        self.state_df['Social_Openness']        = np.random.uniform(b_min, b_max, num_agents)

        # [LEGACY/FALLBACK] - Keeping for compatibility with old spatial_diffusion
        if getattr(config, 'RANDOMIZE_SOCIAL_ATTRIBUTES', False):
            # Randomized personality (0.0 to 1.0)
            self.state_df['Diffusion_Weight']    = np.random.uniform(0.0, 1.0, num_agents)
            self.state_df['Diffusion_Bandwidth'] = np.random.uniform(0.0, 1.0, num_agents)
            self.state_df['Conformity_Weight']   = 0.0 # Disabled daily echo chamber
        else:
            # Global defaults for all agents
            self.state_df['Diffusion_Weight']    = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.8)
            self.state_df['Diffusion_Bandwidth'] = getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5)
            self.state_df['Conformity_Weight']   = 0.0 # Disabled daily echo chamber

        # --- Geodemographic Subcluster Integration ---
        geo_enabled = getattr(config, 'GEODEMOGRAPHIC_ENABLED', True) and _GEO_AVAILABLE
        if geo_enabled:
            blend_min = getattr(config, 'CLUSTER_BLEND_MIN', 0.3)
            blend_max = getattr(config, 'CLUSTER_BLEND_MAX', 0.7)
            test_mode = getattr(config, 'TEST_MODE', True)

            # Assign subcluster label per agent
            # All agents MUST have a pre-assigned 'Geo_Subcluster' column from the data
            if 'Geo_Subcluster' in self.attributes.columns:
                pass
            else:
                # Critical fallback: If data hasn't been processed, default to 'none' and warn
                print("WARNING: 'Geo_Subcluster' column missing in agent data. Defaulting to 'none'.")
                self.attributes['Geo_Subcluster'] = 'none'

            # Per-agent blend weight drawn from U(blend_min, blend_max)
            self.attributes['Cluster_Blend_Weight'] = np.random.uniform(
                blend_min, blend_max, num_agents
            )

            # Blend NTS probabilities with cluster-level contextual priors
            self.attributes = apply_cluster_blend(
                self.attributes,
                subcluster_col='Geo_Subcluster',
                blend_weight_col='Cluster_Blend_Weight'
            )
        else:
            self.attributes['Geo_Subcluster']      = 'none'
            self.attributes['Cluster_Blend_Weight'] = 0.0
        
        # Starting stock randomized between 0 and Capacity
        self.state_df['Stock'] = np.random.uniform(0, self.state_df['Capacity'])

    def consume(self):
        """Daily stock reduction."""
        self.state_df = consumption.consume(self.state_df)

    def check_grocery_need(self):
        """Identifies agents needing to shop."""
        return shopping_logic.check_shopping_need(self.state_df)

    def choose_shopping_mode(self, shopping_mask):
        """Chooses between online, bulk, and convenience."""
        return shopping_logic.choose_mode(self.attributes, shopping_mask)

    def trigger_nts_trips(self):
        """Triggers non-grocery trips based on daily probabilities."""
        return shopping_logic.trigger_trips(self.attributes)

    def choose_destinations(self, mask, grocery_mode_series, utility_matrices, amenity_binary):
        """Selects destination and mode for grocery trips."""
        return travel_choice.choose_destination(
            self.state_df, mask, grocery_mode_series, utility_matrices, amenity_binary
        )

    def choose_nts_destinations(self, trip_type, mask, utility_matrices, amenity_binary):
        """Selects destination and mode for NTS trips."""
        return travel_choice.choose_destination_for_trip(
            trip_type, mask, self.attributes, utility_matrices, amenity_binary
        )

    def choose_chained_destinations(self, trip_list, shoppers_idx, grocery_modes, 
                                    utility_matrices, amenity_binary):
        """Selects single destination for multiple trips."""
        return travel_choice.choose_chained_destination(
            trip_list, shoppers_idx, grocery_modes, utility_matrices, amenity_binary, self.attributes
        )

    def replenish_stock(self, shopping_mask, grocery_modes):
        """Refills stock after shopping based on trip type."""
        self.state_df = consumption.update_stock_after_shop(self.state_df, shopping_mask, grocery_modes)

    def apply_social_influence(self, visits_df, utility_matrices):
        """Applies spatial diffusion (word-of-mouth)."""
        return spatial_diffusion.apply_spatial_diffusion_bonus(
            visits_df, self.attributes, utility_matrices
        )
