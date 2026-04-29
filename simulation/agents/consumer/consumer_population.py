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

        # Agent-specific social behavior traits
        if getattr(config, 'RANDOMIZE_SOCIAL_ATTRIBUTES', False):
            # Randomized personality (0.0 to 1.0)
            self.state_df['Diffusion_Weight']    = np.random.uniform(0.0, 1.0, num_agents)
            self.state_df['Diffusion_Bandwidth'] = np.random.uniform(0.0, 1.0, num_agents)
            self.state_df['Conformity_Weight']   = np.random.uniform(0.0, 1.0, num_agents)
        else:
            # Global defaults for all agents
            self.state_df['Diffusion_Weight']    = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.8)
            self.state_df['Diffusion_Bandwidth'] = getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5)
            self.state_df['Conformity_Weight']   = getattr(config, 'NEIGHBOURHOOD_CONFORMITY', 0.2)
        
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
