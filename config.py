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
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # Retail_ABM/
MODEL_DIR = os.path.dirname(BASE_DIR)                    # Model/

# --- Simulation Mode ---
TEST_MODE = False  # Set to True to use files in 'Utility/testing/'

# --- Input: Utility Scores (also serve as agent data) ---
UTILITY_DIR = os.path.join(MODEL_DIR, "Utility")
if TEST_MODE:
    UTILITY_DIR = os.path.join(UTILITY_DIR, "testing")

# Single consolidated utility file (primary data source for all agents & trips)
UTILITY_SCORES_AVG = os.path.join(UTILITY_DIR, "utility_scores_average.parquet")

# --- Input: NTS ---
NTS_PATH = r'C:\Users\sgabalog\Documents\P3\Model\Distance\Data\NTS\Cleaned_NTS_Data.csv'

# --- Input: Transport Times (Postcode -> Walk/Drive/PT minutes) ---
TRANSPORT_TIMES_PATH = os.path.join(MODEL_DIR, "Distance", "Results", "final_transport_times.parquet")

# --- Input: Retail Centres ---
RETAIL_CENTRES_GPKG = os.path.join(MODEL_DIR, "Retail Centre Data",
                                   "retail_centre_type_counts.gpkg")

# --- Output ---
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
DATA_LOCAL_DIR = os.path.join(BASE_DIR, "data_local")

# --- Model Constants ---
DAILY_CONSUMPTION_MEAN = 50.0   # Mean grocery units consumed per day
DAILY_CONSUMPTION_STD  = 2.0   # Std dev of daily consumption
TICKS_PER_DAY          = 1     # One simulation step = one day

MAX_STOCK_CAPACITY = 100.0     # Maximum grocery stock an agent can hold
REORDER_THRESHOLD  = 20.0      # Stock level below which a grocery trip is triggered

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

# --- Destination Choice Temperature ---
# Controls how strongly agents prefer the highest-utility (centre × mode) option.
# Uses softmax: P(c,m) ∝ exp(SOFTMAX_BETA × utility)
#
#   Low  β (e.g. 1.0)  → soft preference — many alternatives compete
#   Mid  β (e.g. 5.0)  → clear preference for top options (recommended)
#   High β (e.g. 15.0) → near-deterministic — agent almost always picks the best
SOFTMAX_BETA = 5.0
