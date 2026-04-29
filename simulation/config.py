"""
Retail ABM - Configuration

Single place for all file paths and model constants.
Edit this file to point to your data or adjust simulation parameters.

Agent Data Source
-----------------
All agents are now sourced from a single consolidated utility file:
  utility_scores_average.parquet

This contains one row per household with:
  - Household demographics (sex, age_years, nssec8, etc.)
  - NTS trip-frequency columns (prob_trip_comparison, etc.)
  - One column per (retail_centre × transport_mode) with pre-computed
    utility scores, averaged across bulk and convenience shopping modes.

When a grocery trip is triggered the simulation dynamically filters
centres by store type (Foodstore / Convenience Store) before selecting.
"""

import os

# --- Directories ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))   # simulation/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # Retail_ABM/
MODEL_DIR    = os.path.dirname(PROJECT_ROOT)                # Model/
DATA_LOCAL   = os.path.join(MODEL_DIR, "data_local", "liverpool")
INPUT_DIR    = os.path.join(DATA_LOCAL, "inputs")
PROCESSED_DIR = os.path.join(DATA_LOCAL, "processed")

# --- Simulation Mode ---
TEST_MODE = False  # Set to True to use files in 'Utility/testing/'

# --- Input: Utility Scores (also serve as agent data) ---
UTILITY_DIR = PROCESSED_DIR
if TEST_MODE:
    UTILITY_DIR = os.path.join(UTILITY_DIR, "testing")

# Single consolidated utility file (primary data source for all agents & trips)
UTILITY_SCORES_AVG = os.path.join(UTILITY_DIR, "utility_scores_average.parquet")

# --- Input: NTS ---
NTS_PATH = os.path.join(INPUT_DIR, "NTS", "Cleaned_NTS_Data.csv")

# --- Input: Transport Times (Postcode -> Walk/Drive/PT minutes) ---
TRANSPORT_TIMES_PATH = os.path.join(PROCESSED_DIR, "final_transport_times.parquet")

# --- Input: Retail Centres ---
RETAIL_CENTRES_GPKG = os.path.join(PROCESSED_DIR, "retail_centre_type_counts.gpkg")

# --- Output ---
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
DATA_LOCAL_DIR = DATA_LOCAL

# --- Model Constants ---
TICKS_PER_DAY          = 1     # One simulation step = one day

# --- Replenishment & Stock Logic ---
# Random refill ranges (multiplier of agent capacity)
CONVENIENCE_REFILL_RANGE = [0.3, 0.5]
BULK_REFILL_RANGE        = [0.8, 1.0]

# --- Demographic Diffusion Parameters ---
# Controls homophily: how strongly age + income similarity weights the spread of
# retail preferences between households.
#   0.0 = purely spatial (original behaviour — all neighbours treated equally)
#   1.0 = full demographic weighting (only similar households influence each other)
DEMOGRAPHIC_DIFFUSION_WEIGHT = 1.0

# Gaussian decay bandwidth in normalised [0, 1] age-income space.
# Smaller values → steeper drop-off (only near-identical demographics spread).
# Larger values  → gentler drop-off (moderate similarity still carries influence).
DEMOGRAPHIC_BANDWIDTH = 0.5

# --- Social Influence & Diffusion ---
# Minimum visits in a postcode sector to trigger a "trending" centre
DIFFUSION_THRESHOLD_VISITS = 3

# The utility boost applied to centres that are trending in a neighbourhood
DIFFUSION_BOOST_MULTIPLIER = 1.05

# Neighbourhood Conformity ("Echo Chamber" effect)
#   0.0 = no local influence
#   1.0 = total conformity (agents always pick the best local option)
NEIGHBOURHOOD_CONFORMITY = 0.0

# --- Retail Centre Performance & Intervention ---
# Percentile below which a centre is considered "failing" relative to peers
RETAIL_FAILURE_THRESHOLD = 0.10

# Multiplier applied to centre utilities during a welfare intervention
RETAIL_INTERVENTION_BOOST = 1.10

# Maximum cumulative multiplier a centre can receive from interventions
RETAIL_BOOST_CEILING = 1.30

# Weighting range for Size-Peers vs Spatial-Peers [Min_Weight, Max_Weight]
#   Higher value = judge more by size/POI count
#   Lower value  = judge more by nearby competitors
RETAIL_PEER_SIZE_WEIGHT = [0.3, 0.7]

# --- Destination Choice Temperature ---
# Controls how strongly agents prefer the highest-utility (centre × mode) option.
# Uses softmax: P(c,m) ∝ exp(SOFTMAX_BETA × utility)
#
#   Low  β (e.g. 1.0)  → soft preference — many alternatives compete
#   Mid  β (e.g. 5.0)  → clear preference for top options (recommended)
#   High β (e.g. 15.0) → near-deterministic — agent almost always picks the best
SOFTMAX_BETA = 5.0
