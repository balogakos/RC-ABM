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

class TestModelMechanics(unittest.TestCase):
    def test_visit_feedback_loops(self):
        # Initial utility is 1.0 for all
        matrix = pd.DataFrame(np.ones((10, 2)), index=range(10), columns=['C1', 'C2'])
        
        # 5 agents visit C1
        visited_agents = list(range(5))
        visited_centres = ['C1'] * 5
        
        agent.apply_feedback(matrix, visited_agents, visited_centres)
        
        # Check that values for those agents at C1 have changed from 1.0
        # Multiplier is random 0.95 or 1.05
        affected_vals = matrix.loc[0:4, 'C1']
        self.assertTrue((affected_vals != 1.0).all())
        self.assertTrue(((affected_vals == 1.05) | (affected_vals == 0.95)).all())
        
        # Check that non-visited centre C2 is unchanged
        self.assertTrue((matrix['C2'] == 1.0).all())

    def test_spatial_diffusion_bonus(self):
        # Create visits in one postcode
        visits_df = pd.DataFrame({
            'Postcode': ['M1 1AA'] * 10,
            'Retail_Centre': ['C1'] * 10,
            'Trip_Type': ['grocery'] * 10
        })
        
        consumers_sampled = pd.DataFrame({
            'household': range(20),
            'Postcode': ['M1 1AA'] * 20,
            'age_years': 30,
            'income': 30000
        })
        
        # Initial matrix
        matrix = pd.DataFrame(np.ones((20, 1)), index=range(20), columns=['C1'])
        matrices = {'average': matrix}
        
        agent.apply_spatial_diffusion_bonus(visits_df, consumers_sampled, matrices)
        
        # Agents in that postcode should have received a boost for C1
        # Default boost is +10% (1.10)
        self.assertTrue((matrices['average']['C1'] > 1.0).all())

if __name__ == '__main__':
    unittest.main()
