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

class TestDestinationChoice(unittest.TestCase):
    def setUp(self):
        self.centres = ['C1', 'C2', 'C3']
        self.agent_ids = range(500)
        
        # Identity matrix: Agent i prefers Centre i (mod 3)
        vals = np.zeros((500, 3))
        for i in range(500):
            vals[i, i % 3] = 10.0 # High utility for one specific centre
            
        self.utility_matrices = {
            'bulk_drive': pd.DataFrame(vals, index=self.agent_ids, columns=self.centres),
            'convenience_drive': pd.DataFrame(vals, index=self.agent_ids, columns=self.centres),
            'comparison_drive': pd.DataFrame(vals, index=self.agent_ids, columns=self.centres)
        }
        
        self.amenity_binary = {
            'Foodstore': pd.Series([1, 1, 1], index=self.centres),
            'Retail': pd.Series([1, 0, 0], index=self.centres), # Only C1 has retail
            'Entertainment': pd.Series([0, 1, 0], index=self.centres) # Only C2 has entertainment
        }

    def test_amenity_exclusion(self):
        # Even if C2/C3 have high utility, comparison trips MUST go to C1
        consumers = pd.DataFrame({'household': self.agent_ids})
        triggered = pd.Series([True] * 500)
        
        # We need a dict of matrices for comparison (simulation uses specific keys)
        matrices = {'comparison_drive': self.utility_matrices['comparison_drive']}
        
        dests, modes, scores = agent.choose_destination_for_trip(
            'comparison', triggered, consumers, matrices, self.amenity_binary
        )
        
        # All valid destinations must be C1
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C1').all())

    def test_softmax_probability_distribution(self):
        # Sample test: if C1 has utility 1.0 and C2 has 0.1
        # Patch BOTH the local config and the one imported by agent
        config.SOFTMAX_BETA = 1.0
        agent.config.SOFTMAX_BETA = 1.0
        
        # Create 1000 agents who all see C1= utility 1.0, C2= utility 0.1
        n_sample = 1000
        vals = np.zeros((n_sample, 2))
        vals[:, 0] = 1.0 # C1
        vals[:, 1] = 0.1 # C2
        centres = ['C1', 'C2']
        
        matrix = pd.DataFrame(vals, index=range(n_sample), columns=centres)
        amenities = {'Foodstore': pd.Series([1, 1], index=centres)}
        
        # Using internal sampling function or choose_destination
        # We'll use choose_destination_for_trip which calls the sampler
        triggered = pd.Series([True] * n_sample)
        consumers = pd.DataFrame({'household': range(n_sample)})
        
        dests, _, _ = agent.choose_destination_for_trip(
            'comparison', triggered, consumers, {'comparison_drive': matrix}, amenities
        )
        
        counts = dests.value_counts()
        # With Beta=1.0, exp(1.0)/exp(0.1) = 2.718/1.105 = 2.45
        # Ratio should be ~2.45
        ratio = counts.get('C1', 0) / counts.get('C2', 1)
        self.assertAlmostEqual(ratio, 2.459, delta=0.5)

if __name__ == '__main__':
    unittest.main()
