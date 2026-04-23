"""
Retail ABM - Test Suite

Tests for both simulation systems using synthetic data (no real files needed).
Run with: python tests.py

Tests are grouped by system:
  1. Grocery system (stock-based)
  2. NTS frequency trip system
  3. Amenity filter correctness
  4. Destination choice probabilities
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add project root and simulation folder to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "simulation"))

import config
import agent
config.DAILY_CONSUMPTION_MEAN = 5.0
config.DAILY_CONSUMPTION_STD  = 1.0
config.REORDER_THRESHOLD      = 20.0
config.MAX_STOCK_CAPACITY     = 100.0

import agent


# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic data
# ---------------------------------------------------------------------------

def make_agents(n=50, stock=50.0, consumption=5.0,
                prob_bulk=0.5, prob_convenience=0.3, prob_online=0.2,
                prob_comparison=0.5, prob_service=0.5,
                prob_entertainment=0.5, prob_food_drink=0.5):
    """Return (state_df, consumers_df) with synthetic values."""
    consumers = pd.DataFrame({
        'household':          range(n),
        'prob_bulk':          prob_bulk,
        'prob_convenience':   prob_convenience,
        'prob_online':        prob_online,
        'prob_trip_comparison':   prob_comparison,
        'prob_trip_service':      prob_service,
        'prob_trip_entertainment': prob_entertainment,
        'prob_trip_food_drink':   prob_food_drink,
    })
    state_df, consumers_sampled = agent.initialize_agent_state(n, consumers)
    # Override stock to a known value for deterministic tests
    state_df['Stock'] = stock
    state_df['Consumption_Rate'] = consumption
    return state_df, consumers_sampled


def make_utility_matrix(agent_ids, centre_ids, values=None):
    """Return a DataFrame(index=agent_ids, cols=centre_ids)."""
    if values is None:
        values = np.ones((len(agent_ids), len(centre_ids)))
    return pd.DataFrame(values, index=agent_ids, columns=centre_ids)


def make_amenity_binary(centre_ids, amenity_map):
    """
    amenity_map: {col_name: list_of_values_per_centre}
    Returns dict {col -> Series(index=centre_ids)}
    """
    return {col: pd.Series(vals, index=centre_ids)
            for col, vals in amenity_map.items()}


# ---------------------------------------------------------------------------
# 1. Grocery System Tests
# ---------------------------------------------------------------------------

class TestGrocerySystem(unittest.TestCase):

    def test_consumption_reduces_stock(self):
        """Stock decreases by consumption rate each day."""
        state_df, _ = make_agents(n=10, stock=50.0, consumption=5.0)
        state_df = agent.consume(state_df)
        self.assertTrue((state_df['Stock'] == 45.0).all(),
                        "Stock should be 50 - 5 = 45 after one consume step.")

    def test_stock_floored_at_zero(self):
        """Stock never goes negative."""
        state_df, _ = make_agents(n=10, stock=2.0, consumption=10.0)
        state_df = agent.consume(state_df)
        self.assertTrue((state_df['Stock'] >= 0).all(),
                        "Stock should be clipped at 0, not go negative.")

    def test_no_shopping_need_when_stock_high(self):
        """Agents above threshold should NOT need to shop."""
        state_df, _ = make_agents(n=20, stock=80.0)
        needs = agent.check_shopping_need(state_df)
        self.assertFalse(needs.any(),
                         "No agent should need shopping when stock is 80 (threshold=20).")

    def test_all_need_shopping_when_stock_low(self):
        """All agents below threshold should need to shop."""
        state_df, _ = make_agents(n=20, stock=5.0)
        needs = agent.check_shopping_need(state_df)
        self.assertTrue(needs.all(),
                        "All agents should need shopping when stock is 5 (threshold=20).")

    def test_stock_replenished_after_shop(self):
        """Stock returns to MAX_STOCK_CAPACITY after update_stock_after_shop."""
        state_df, _ = make_agents(n=10, stock=5.0)
        all_need = agent.check_shopping_need(state_df)
        state_df = agent.update_stock_after_shop(state_df, all_need)
        self.assertTrue((state_df['Stock'] == config.MAX_STOCK_CAPACITY).all(),
                        "Stock should be fully replenished after shopping.")

    def test_mode_choice_online_only(self):
        """When prob_online=1, all modes should be 'online'."""
        state_df, consumers = make_agents(
            n=100, stock=5.0,
            prob_online=1.0, prob_bulk=0.0, prob_convenience=0.0)
        needs = agent.check_shopping_need(state_df)
        modes = agent.choose_mode(consumers, needs)
        self.assertTrue((modes == 'online').all(),
                        "All modes should be 'online' when prob_online=1.")

    def test_mode_choice_bulk_only(self):
        """When prob_bulk=1, all modes should be 'bulk'."""
        state_df, consumers = make_agents(
            n=100, stock=5.0,
            prob_online=0.0, prob_bulk=1.0, prob_convenience=0.0)
        needs = agent.check_shopping_need(state_df)
        modes = agent.choose_mode(consumers, needs)
        self.assertTrue((modes == 'bulk').all(),
                        "All modes should be 'bulk' when prob_bulk=1.")


# ---------------------------------------------------------------------------
# 2. NTS Frequency Trip System Tests
# ---------------------------------------------------------------------------

class TestNTSTripSystem(unittest.TestCase):

    def test_trigger_trips_returns_all_types(self):
        """trigger_trips must return all configured trip types."""
        _, consumers = make_agents(n=20)
        triggered = agent.trigger_trips(consumers)
        # Filter for nts types (those with a prob_col)
        nts_types = {k for k, v in agent.TRIP_TYPE_CONFIG.items() if v['prob_col'] is not None}
        self.assertTrue(nts_types.issubset(set(triggered.keys())),
                         "trigger_trips must return all NTS trip types.")

    def test_zero_probability_never_triggers(self):
        """With prob=0, no trips should ever fire."""
        _, consumers = make_agents(
            n=200, prob_comparison=0.0, prob_service=0.0,
            prob_entertainment=0.0, prob_food_drink=0.0)
        triggered = agent.trigger_trips(consumers)
        for trip_type, mask in triggered.items():
            self.assertFalse(mask.any(),
                             f"{trip_type}: no trips should fire when prob=0.")

    def test_probability_one_always_triggers(self):
        """With prob=1, every agent should trigger every day."""
        _, consumers = make_agents(
            n=200, prob_comparison=1.0, prob_service=1.0,
            prob_entertainment=1.0, prob_food_drink=1.0)
        triggered = agent.trigger_trips(consumers)
        for trip_type, mask in triggered.items():
            if agent.TRIP_TYPE_CONFIG[trip_type]['prob_col'] is None:
                continue
            self.assertTrue(mask.all(),
                            f"{trip_type}: all agents should trigger when prob=1.")

    def test_multiple_types_fire_simultaneously(self):
        """With all probs=1, an agent can have all 4 trip types fire on the same day."""
        _, consumers = make_agents(
            n=50, prob_comparison=1.0, prob_service=1.0,
            prob_entertainment=1.0, prob_food_drink=1.0)
        triggered = agent.trigger_trips(consumers)
        # Every agent should appear in all NTS masks
        for trip_type, mask in triggered.items():
            if agent.TRIP_TYPE_CONFIG[trip_type]['prob_col'] is None:
                continue
            self.assertEqual(mask.sum(), 50,
                             f"All 50 agents should fire {trip_type}.")

    def test_trigger_rate_statistically_correct(self):
        """At prob=0.5, roughly 50% of agents should trigger (±5% tolerance)."""
        np.random.seed(99)
        _, consumers = make_agents(
            n=5000, prob_comparison=0.5, prob_service=0.5,
            prob_entertainment=0.5, prob_food_drink=0.5)
        triggered = agent.trigger_trips(consumers)
        for trip_type, mask in triggered.items():
            if agent.TRIP_TYPE_CONFIG[trip_type]['prob_col'] is None:
                continue
            rate = mask.mean()
            self.assertAlmostEqual(rate, 0.5, delta=0.05,
                                   msg=f"{trip_type} trigger rate {rate:.3f} not close to 0.5.")


# ---------------------------------------------------------------------------
# 3. Amenity Filter Tests
# ---------------------------------------------------------------------------

class TestAmenityFilters(unittest.TestCase):

    def setUp(self):
        """Common setup: 10 agents, 3 centres."""
        self.n = 10
        self.centres = ['C1', 'C2', 'C3']
        _, self.consumers = make_agents(self.n)
        # Agent IDs for utility matrix index
        self.agent_ids = self.consumers['household'].values
        self.utility_matrices = {
            'bulk':        make_utility_matrix(self.agent_ids, self.centres),
            'convenience': make_utility_matrix(self.agent_ids, self.centres),
        }
        self.all_triggered = pd.Series(True, index=self.consumers.index)

    def test_service_filter_only_valid_centres(self):
        """Service trips only land on centres with Personal & Professional Services = 1."""
        # C1=1, C2=0, C3=0
        amenity = make_amenity_binary(self.centres,
            {'Personal and Professional Services': [1, 0, 0]})
        dests, modes, scores = agent.choose_destination_for_trip(
            'service', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C1').all(),
                        "Service trips must only go to C1 (the only centre with Svcs=1).")

    def test_comparison_filter_only_retail_centres(self):
        """Comparison trips only land on centres with Retail = 1."""
        amenity = make_amenity_binary(self.centres,
            {'Retail': [0, 1, 0]})
        dests, modes, scores = agent.choose_destination_for_trip(
            'comparison', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C2').all(),
                        "Comparison trips must only go to C2 (the only centre with Retail=1).")

    def test_entertainment_filter(self):
        """Entertainment trips only land on centres with Entertainment = 1."""
        amenity = make_amenity_binary(self.centres,
            {'Entertainment': [0, 0, 1]})
        dests, modes, scores = agent.choose_destination_for_trip(
            'entertainment', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C3').all(),
                        "Entertainment trips must only go to C3 (only centre with Ent=1).")

    def test_food_drink_or_logic_cafe_only(self):
        """Food/Drink trip: centre with only Cafe=1 (Restaurant=0) is still valid."""
        amenity = make_amenity_binary(self.centres, {
            'Cafe':       [1, 0, 0],
            'Restaurant': [0, 0, 0],
        })
        dests, modes, scores = agent.choose_destination_for_trip(
            'food_drink', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C1').all(),
                        "C1 has Cafe=1, so food_drink trips should go there despite Restaurant=0.")

    def test_food_drink_or_logic_restaurant_only(self):
        """Food/Drink trip: centre with only Restaurant=1 (Cafe=0) is still valid."""
        amenity = make_amenity_binary(self.centres, {
            'Cafe':       [0, 0, 0],
            'Restaurant': [0, 1, 0],
        })
        dests, modes, scores = agent.choose_destination_for_trip(
            'food_drink', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C2').all(),
                        "C2 has Restaurant=1, so food_drink trips should go there despite Cafe=0.")

    def test_food_drink_both_zero_no_destination(self):
        """Food/Drink trip: if no centre has Cafe=1 OR Restaurant=1, no destination chosen."""
        amenity = make_amenity_binary(self.centres, {
            'Cafe':       [0, 0, 0],
            'Restaurant': [0, 0, 0],
        })
        dests, modes, scores = agent.choose_destination_for_trip(
            'food_drink', self.all_triggered, self.consumers,
            self.utility_matrices, amenity)
        self.assertTrue(dests.isna().all(),
                        "No destination should be chosen when all amenity values are 0.")

    def test_zero_utility_centre_never_chosen(self):
        """A centre with utility=0 is never chosen, even without amenity filter."""
        # C1 and C3 have utility=0, only C2=1
        vals = np.array([[0, 1, 0]] * self.n, dtype=float)
        utility_matrices = {
            'bulk':        make_utility_matrix(self.agent_ids, self.centres, vals),
            'convenience': make_utility_matrix(self.agent_ids, self.centres, vals),
        }
        amenity = make_amenity_binary(self.centres, {'Retail': [1, 1, 1]})
        dests, modes, scores = agent.choose_destination_for_trip(
            'comparison', self.all_triggered, self.consumers,
            utility_matrices, amenity)
        valid_dests = dests.dropna()
        self.assertTrue((valid_dests == 'C2').all(),
                        "Only C2 (utility=1) should be chosen; C1 and C3 have utility=0.")


# ---------------------------------------------------------------------------
# 4. Destination Choice Distribution Tests
# ---------------------------------------------------------------------------

class TestDestinationDistribution(unittest.TestCase):

    def test_proportional_choice(self):
        """
        Centre with 3x higher utility should be chosen ~3x more often.
        Uses 2000 agents to get statistical stability.
        """
        np.random.seed(42)
        n = 2000
        centres = ['Low', 'High']
        _, consumers = make_agents(n, prob_comparison=1.0)
        agent_ids = consumers['household'].values

        # High centre has 3x utility. Use log values so that exp(beta*util) yields 3:1 ratio
        vals = np.column_stack([np.full(n, np.log(1.0)), np.full(n, np.log(3.0))])
        utility_matrices = {
            'comparison_drive': make_utility_matrix(agent_ids, centres, vals),
        }
        amenity = make_amenity_binary(centres, {'Retail': [1, 1]})
        all_triggered = pd.Series(True, index=consumers.index)

        dests, modes, scores = agent.choose_destination_for_trip(
            'comparison', all_triggered, consumers, utility_matrices, amenity)

        counts = dests.value_counts()
        ratio = counts.get('High', 0) / counts.get('Low', 1)
        # Expect ratio ~3 ± 0.5
        self.assertAlmostEqual(ratio, 3.0, delta=0.5,
                               msg=f"Expected High/Low ratio ~3.0, got {ratio:.2f}")


# ---------------------------------------------------------------------------
# 5. Feedback Loop Tests
# ---------------------------------------------------------------------------

class TestFeedbackLoop(unittest.TestCase):

    def test_apply_feedback_updates_matrix(self):
        """apply_feedback modifies the matrix using random 1.05 / 0.95 multipliers."""
        agent_ids = [0, 1, 2]
        centres = ['C1', 'C2']
        matrix = make_utility_matrix(agent_ids, centres, np.ones((3, 2)))
        
        # Agents 0, 1 visit C1. Agent 2 visits C2.
        visited_agents = [0, 1, 2]
        visited_centres = ['C1', 'C1', 'C2']
        
        # Test original values are 1.0
        self.assertTrue((matrix.values == 1.0).all())
        
        agent.apply_feedback(matrix, visited_agents, visited_centres)
        
        # The values should now be either 1.05 or 0.95 (due to random multipliers)
        val0 = matrix.loc[0, 'C1']
        val1 = matrix.loc[1, 'C1']
        val2 = matrix.loc[2, 'C2']
        
        for val in [val0, val1, val2]:
            self.assertTrue(np.isclose(val, 1.05) or np.isclose(val, 0.95), 
                            f"Utility value was not updated correctly: {val}")
            
        # Agent 0 did not visit C2
        self.assertEqual(matrix.loc[0, 'C2'], 1.0)


# ---------------------------------------------------------------------------
# 6. Retail Centre Evaluation Tests
# ---------------------------------------------------------------------------

class TestRetailCentreEvaluation(unittest.TestCase):

    def setUp(self):
        import geopandas as gpd
        from shapely.geometry import Point
        
        # 3 centres forming a spatial cluster, C1 has very few visits
        # All have same size (100) so they are size peers.
        self.centres = ['C1', 'C2', 'C3', 'C4']
        self.retail_gdf = gpd.GeoDataFrame({
            'Total_POI_': [100, 100, 100, 500], # C4 is not a size peer
            'geometry': [
                Point(0, 0), Point(1, 0), Point(0, 1), # Cluster
                Point(100, 100) # Far away
            ]
        }, index=self.centres)
        
        # Build 1 day of sparse visits DataFrame
        # Trip type 'service'. C1=1 visit, C2=20 visits, C3=20 visits, C4=1 visit
        visits = []
        for c, count in zip(self.centres, [1, 20, 20, 1]):
            for _ in range(count):
                visits.append({'Day': 1, 'Retail_Centre': c, 'Trip_Type': 'service'})
        self.visits_df = pd.DataFrame(visits)
        
        # Initial Utility Matrix (average) with 1.0 for everyone
        self.agent_ids = [0, 1, 2] # Dummy agents
        self.utility_matrices = {
            'average': make_utility_matrix(self.agent_ids, self.centres, np.ones((3, 4)))
        }

    def test_evaluate_retail_centres_bottom_10_boosts_utility(self):
        """C1 is in the bottom 10% of size peers (1 vs 20,20) and spatial peers. Should be boosted."""
        
        # First failure: creates a strike
        tracker = {}
        agent.evaluate_retail_centres(self.visits_df, self.retail_gdf, self.utility_matrices, amenities, tracker=tracker)
        self.assertEqual(self.utility_matrices['average'].loc[0, 'C1'], 1.0) # No boost yet
        
        # Second consecutive failure: triggers boost
        agent.evaluate_retail_centres(self.visits_df, self.retail_gdf, self.utility_matrices, amenities, tracker=tracker)
        self.assertEqual(self.utility_matrices['average'].loc[0, 'C1'], 1.10)
        # C2, C3 are doing fine -> no boost
        self.assertEqual(self.utility_matrices['average'].loc[0, 'C2'], 1.0)
        self.assertEqual(self.utility_matrices['average'].loc[0, 'C3'], 1.0)
        
        # C4 is a size outlier (500) so it has no size peers. 
        # But it's in the spatial peers list of nobody close, and its closest peers are C1,C2,C3.
        # Let's just check C1 was explicitly boosted in the messages.
        self.assertTrue(any('C1' in msg for msg in messages), "C1 was not reported in the boost messages.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Retail ABM — Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
