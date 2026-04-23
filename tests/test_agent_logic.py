import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root and simulation folder to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))

from simulation import agent, config

class TestAgentLogic(unittest.TestCase):
    def setUp(self):
        # Override config for deterministic testing
        config.DAILY_CONSUMPTION_MEAN = 10.0
        config.DAILY_CONSUMPTION_STD = 0.0
        config.REORDER_THRESHOLD = 20.0
        config.MAX_STOCK_CAPACITY = 100.0
        
        self.n = 100
        self.consumers_df = pd.DataFrame({
            'household': range(self.n),
            'prob_bulk': 0.6,
            'prob_convenience': 0.3,
            'prob_online': 0.1,
            'prob_trip_comparison': 0.5,
            'prob_trip_service': 0.5,
            'prob_trip_entertainment': 0.5,
            'prob_trip_food_drink': 0.5,
            'Postcode': 'M1 1AA'
        })

    def test_stock_consumption(self):
        state_df, _ = agent.initialize_agent_state(self.n, self.consumers_df)
        state_df['Consumption_Rate'] = 10.0 # Force deterministic rate
        initial_stock = state_df['Stock'].copy()
        
        # Consume for 1 day
        state_df = agent.consume(state_df)
        expected_stock = initial_stock - 10.0
        
        pd.testing.assert_series_equal(state_df['Stock'], expected_stock)

    def test_shopping_threshold_trigger(self):
        state_df, _ = agent.initialize_agent_state(self.n, self.consumers_df)
        
        # Set stock just above threshold
        state_df['Stock'] = 21.0
        self.assertFalse(agent.check_shopping_need(state_df).any())
        
        # Set stock just below threshold
        state_df['Stock'] = 19.0
        self.assertTrue(agent.check_shopping_need(state_df).all())

    def test_stock_replenishment(self):
        state_df, _ = agent.initialize_agent_state(self.n, self.consumers_df)
        state_df['Stock'] = 10.0
        needs_shop = pd.Series([True] * self.n)
        
        state_df = agent.update_stock_after_shop(state_df, needs_shop)
        self.assertEqual(state_df['Stock'].iloc[0], 100.0)

    def test_trip_trigger_frequencies(self):
        # Set specific probabilities
        self.consumers_df['prob_trip_comparison'] = 1.0 # Always
        self.consumers_df['prob_trip_service'] = 0.0    # Never
        
        triggered = agent.trigger_trips(self.consumers_df)
        
        self.assertTrue(triggered['comparison'].all())
        self.assertFalse(triggered['service'].any())

if __name__ == '__main__':
    unittest.main()
