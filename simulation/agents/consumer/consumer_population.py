import pandas as pd
import numpy as np
import config
from .behaviours import consumption, shopping_logic, travel_choice, spatial_diffusion

class ConsumerPopulation:
    """
    Manages the state and behaviours of the entire consumer population.
    Uses vectorized operations on internal DataFrames for high performance.
    """
    def __init__(self, num_agents, attributes_df):
        self.num_agents = num_agents
        # Initialize attributes
        if len(attributes_df) < num_agents:
            self.attributes = attributes_df.sample(n=num_agents, replace=True).reset_index(drop=True)
        else:
            self.attributes = attributes_df.sample(n=num_agents, replace=False).reset_index(drop=True)

        # Initialize dynamic state from attributes_df columns
        self.state_df = pd.DataFrame({
            'AgentID':            self.attributes['household'].values if 'household' in self.attributes.columns else range(num_agents),
            'Postcode':           self.attributes['Postcode'].values if 'Postcode' in self.attributes.columns else 'UNKNOWN',
            'Stock':              self.attributes['stock_level'].values if 'stock_level' in self.attributes.columns else np.random.uniform(50, 100, num_agents),
            'Consumption_Rate':   self.attributes['consumption_rate'].values if 'consumption_rate' in self.attributes.columns else np.clip(np.random.normal(config.DAILY_CONSUMPTION_MEAN, config.DAILY_CONSUMPTION_STD, num_agents), 0.5, None),
            'Shopping_Threshold': self.attributes['shopping_threshold'].values if 'shopping_threshold' in self.attributes.columns else config.REORDER_THRESHOLD,
            'Last_Shop_Day':      -1,
            'Shopping_Mode':      None,
        })

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

    def replenish_stock(self, shopping_mask):
        """Refills stock after shopping."""
        self.state_df = consumption.update_stock_after_shop(self.state_df, shopping_mask)

    def apply_social_influence(self, visits_df, utility_matrices):
        """Applies spatial diffusion (word-of-mouth)."""
        return spatial_diffusion.apply_spatial_diffusion_bonus(
            visits_df, self.attributes, utility_matrices
        )
