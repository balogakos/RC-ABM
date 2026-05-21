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

EVAL_FREQ = 10
SOCIAL_SCALING_ALPHA = 0.05
CONFORMITY_RANGE = [0.1, 0.5]
SOCIAL_OPENNESS_RANGE = [0.1, 1.0]

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

# --- Social Influence & Diffusion Configuration ---
# If True, every agent is assigned a unique, random personality (0.0 to 1.0)
# for Diffusion Weight, Bandwidth, and Conformity. This removes the need 
# for global calibration and simulates a diverse population.
RANDOMIZE_SOCIAL_ATTRIBUTES = True

# [FALLBACK GLOBALS] - These are only used if RANDOMIZE_SOCIAL_ATTRIBUTES is False
# or if an agent is missing its specific trait.
DEMOGRAPHIC_DIFFUSION_WEIGHT = 0.8
DEMOGRAPHIC_BANDWIDTH        = 0.5
# NOTE: Neighbourhood Conformity is now handled by the new additive mechanism every 10 days.

# --- Geodemographic Subcluster Integration ---
# Toggle the entire feature on/off (False = ablation run, no cluster blending)
GEODEMOGRAPHIC_ENABLED = True

# Per-agent blend weight w_i is drawn from Uniform(CLUSTER_BLEND_MIN, CLUSTER_BLEND_MAX)
# at initialisation. Mean ≈ 0.5 → equal weight to NTS individual and cluster prior.
CLUSTER_BLEND_MIN    = 0.3
CLUSTER_BLEND_MAX    = 0.7

# Sigmoid sharpness applied when converting z-scores to probability priors.
# Higher γ = sharper differentiation between cluster priors.
CLUSTER_SIGMOID_GAMMA = 1.5


# --- Social Influence & Diffusion ---
# Threshold ratio: percentage of a postcode sector's population that must visit
# a centre for it to start "trending" (Word-of-Mouth effect).
DIFFUSION_POPULATION_RATIO = 0.05

# Hard floor for trending: minimum absolute visits required regardless of ratio.
# This prevents noise/outliers in very low-population (rural) sectors.
DIFFUSION_MIN_VISITS = 3

# The utility boost applied to centres that are trending in a neighbourhood
DIFFUSION_BOOST_MULTIPLIER = 1.05

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

# --- Social Behavior & Decay ---
SOCIAL_SCALING_ALPHA   = 0.05
SOCIAL_DECAY_FACTOR    = 0.75    # 0.75 = 25% decay per period. Ensures 180-day convergence.
SOCIAL_START_SIGMA     = 0.05    # Standard deviation of the initial random social reputation.
GEODEMOGRAPHIC_ENABLED = True
DISTANCE_SENSITIVITY   = 1.0
RETAIL_INTERVENTION    = 1.0

# --- Temporal Variability ---
# Controls the random daily fluctuation in trip probabilities (0.15 = +/- 15%)
PROBABILITY_VARIANCE   = 0.15
