"""
Retail ABM - Trip Frequency Assignment

Adds NTS-derived daily trip probability columns to the bulk utility scores file,
which is the primary agent data source for the simulation.

Input : utility_scores_bulk.parquet      (1 row per household, cols = demographics + retail centre IDs)
Output: utility_scores_bulk_with_trips.parquet  (same + prob_trip_* columns)

Columns added
-------------
    freq_comparison         Weekly comparison-shopping trips  (from PurposeCount_Comparison)
    freq_entertainment      Weekly entertainment trips        (from PurposeCount_Entertainment)
    freq_service            Weekly service trips              (from PurposeCount_Service)
    freq_food_drink         Weekly cafe/restaurant trips      (from PurposeCount_Food/Drink)
    prob_trip_comparison    freq / 7  (capped at 1.0)
    prob_trip_entertainment freq / 7
    prob_trip_service       freq / 7
    prob_trip_food_drink    freq / 7

Note: grocery (bulk/convenience) is handled by the stock system, not here.
Run this script ONCE before the main simulation, or let main.py auto-run it.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# NTS column → friendly name (grocery excluded — handled by stock system)
TRIP_COLUMNS = {
    'PurposeCount_Comparison':    'comparison',
    'PurposeCount_Entertainment': 'entertainment',
    'PurposeCount_Service':       'service',
    'PurposeCount_Food/Drink':    'food_drink',
}


def assign_frequencies(agents_path, output_path, nts_path=None, seed=42):
    """
    Loads the agent file, assigns NTS trip frequency columns, saves enriched file.

    Parameters
    ----------
    agents_path : str   Path to input parquet (utility_scores_bulk.parquet).
    output_path : str   Where to write the enriched parquet.
    nts_path    : str   Path to Cleaned_NTS_Data.csv (defaults to config.NTS_PATH).
    seed        : int   Random seed for reproducibility.
    """
    if nts_path is None:
        nts_path = config.NTS_PATH

    print(f"  Loading agents from {agents_path} ...")
    agents = pd.read_parquet(agents_path)

    print(f"  Loading NTS data from {nts_path} ...")
    nts = pd.read_csv(nts_path, usecols=list(TRIP_COLUMNS.keys())).dropna()
    print(f"  NTS rows: {len(nts):,}  |  Agent rows: {len(agents):,}")

    np.random.seed(seed)
    sampled_idx = np.random.randint(0, len(nts), size=len(agents))
    sampled_nts = nts.iloc[sampled_idx].reset_index(drop=True)

    for nts_col, name in TRIP_COLUMNS.items():
        agents[f'freq_{name}']      = sampled_nts[nts_col].values.astype(float)
        agents[f'prob_trip_{name}'] = (agents[f'freq_{name}'] / 7.0).clip(upper=1.0)

    print("  Assigned daily probabilities:")
    for name in TRIP_COLUMNS.values():
        col = f'prob_trip_{name}'
        print(f"    {col}: mean={agents[col].mean():.3f}, max={agents[col].max():.3f}")

    agents.to_parquet(output_path)
    print(f"  Saved to {output_path}")

    print("\n  === Average daily trip probabilities assigned (NTS-derived) ===")
    for name in TRIP_COLUMNS.values():
        col = f'prob_trip_{name}'
        print(f"  {name:<15} : mean daily prob = {agents[col].mean():.3f}  "
              f"(i.e. ~{agents[col].mean()*7:.2f} trips/week on average)")

    return agents


if __name__ == '__main__':
    print("--- Assigning NTS trip frequencies to bulk utility scores ---")
    if os.path.exists(config.UTILITY_SCORES_BULK):
        assign_frequencies(config.UTILITY_SCORES_BULK,
                           config.UTILITY_SCORES_BULK_WITH_TRIPS)
    else:
        print(f"ERROR: File not found: {config.UTILITY_SCORES_BULK}")
