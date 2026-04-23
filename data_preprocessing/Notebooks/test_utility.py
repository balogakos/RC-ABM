"""
Retail ABM - Gravity Model with Mode-Specific Distance Decay
"""

import pandas as pd
import numpy as np
import os
import ast
import gc
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
TEST_MODE = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 10000

CONSUMER_FILES = {
    'bulk':        'consumer_agents_bulk.parquet',
    'convenience': 'consumer_agents_convenience.parquet',
}
RETAIL_FILE = 'retail_centres_processed.parquet'

BULK_PARAM_FILE = os.path.join(BASE_DIR, 'Consumer_Agents', 'bulk_parameterised_consumer_agents_full.parquet')
CON_PARAM_FILE  = os.path.join(BASE_DIR, 'Consumer_Agents', 'con_parameterised_consumer_agents_full.parquet')

NTS_PATH = r'C:\Users\sgabalog\Documents\P3\Model\Distance\Data\NTS\Cleaned_NTS_Data.csv'

# --- IMPROVEMENT: Mode-Specific Friction Multipliers ---
# Higher = more distance sensitive (Walk > PT > Drive)
MODE_SENSITIVITY = {
    'Walk': 4.0,  # Steeper penalty for walking
    'PT': 1.5,    # Medium penalty for Public Transport
    'Drive': 1.0  # Baseline penalty for Driving
}

TRIP_PROFILES = {
    'bulk': {
        'consumer_file': 'bulk',
        'param_source': 'bulk',
        'amenities': ['Foodstore'],
        'distance_power': 3.0,
        'bonus_percentage': False
    },
    'convenience': {
        'consumer_file': 'convenience',
        'param_source': 'con',
        'amenities': ['Convenience Store'],
        'distance_power': 8.0, # High sensitivity for local convenience
        'bonus_percentage': False
    },
    'comparison': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Retail'],
        'distance_power': 1.5, # Lower sensitivity for flagship retail
        'bonus_percentage': True
    },
    'entertainment': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Entertainment'],
        'distance_power': 2.0,
        'bonus_percentage': True
    },
    'food_drink': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Cafe', 'Restaurant'],
        'distance_power': 2.0,
        'bonus_percentage': True
    },
    'service': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Personal and Professional Services'],
        'distance_power': 2.5,
        'bonus_percentage': True
    }
}

ALL_AMENITIES = ['Retail', 'Cafe', 'Personal and Professional Services', 'Restaurant', 'Entertainment', 'Foodstore', 'Convenience Store']
CONSUMER_META_COLS = ['sex', 'age_years', 'nssec8', 'pwkstat', 'salary_yearly', 'id', 'household', 'num_children', 'household_size', 'Postcode']

# ---------------------------------------------------------------------------

def apply_nts_enrichment(meta_df, nts_path, seed=42):
    """Enriches the agents dataframe with NTS daily trip probabilities."""
    if not os.path.exists(nts_path):
        return meta_df
        
    cols = {
        'PurposeCount_Comparison':    'comparison',
        'PurposeCount_Entertainment': 'entertainment',
        'PurposeCount_Service':       'service',
        'PurposeCount_Food/Drink':    'food_drink',
    }
    
    nts = pd.read_csv(nts_path, usecols=list(cols.keys())).dropna()
    
    np.random.seed(seed)
    sampled_idx = np.random.randint(0, len(nts), size=len(meta_df))
    sampled_nts = nts.iloc[sampled_idx].reset_index(drop=True)
    
    for nts_col, name in cols.items():
        meta_df[f'freq_{name}'] = sampled_nts[nts_col].values.astype(float)
        meta_df[f'prob_trip_{name}'] = (meta_df[f'freq_{name}'] / 7.0).clip(upper=1.0)
        
    return meta_df

def safe_eval(x):
    """Safely convert string to dict."""
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return {}
    return {}

def load_and_prep_retail_data():
    retail_path = os.path.join(BASE_DIR, RETAIL_FILE)
    if not os.path.exists(retail_path): return None

    df_rc = pd.read_parquet(retail_path)

    requested_cols = [
        'RC_ID', 'Classifica', 'Total_POI_',
        'Foodstore_scaled_class', 'Convenience Store_scaled_class',
        'Retail_scaled_class', 'Cafe_scaled_class',
        'Personal and Professional Services_scaled_class',
        'Restaurant_scaled_class', 'Entertainment_scaled_class',
        'Foodstore', 'Convenience Store', 'Retail', 'Cafe', 
        'Personal and Professional Services', 'Restaurant', 'Entertainment'
    ]
    available_cols = [c for c in requested_cols if c in df_rc.columns]
    df_rc = df_rc[available_cols].copy()

    rename_map = {
        'Foodstore_scaled_class': 'Foodstore_scaled',
        'Convenience Store_scaled_class': 'Convenience Store_scaled',
        'Retail_scaled_class': 'Retail_scaled',
        'Cafe_scaled_class': 'Cafe_scaled',
        'Personal and Professional Services_scaled_class': 'Personal and Professional Services_scaled',
        'Restaurant_scaled_class': 'Restaurant_scaled',
        'Entertainment_scaled_class': 'Entertainment_scaled',
    }
    df_rc.rename(columns={k: v for k, v in rename_map.items() if k in df_rc.columns}, inplace=True)
    df_rc['RC_ID'] = df_rc['RC_ID'].astype(str)
    
    rc_ids = df_rc['RC_ID'].tolist()
    return df_rc.set_index('RC_ID'), rc_ids

def get_weight_value(w_col_name, default_name, df_params):
    """Helper to extract parameter weights from df_params."""
    if w_col_name in df_params.columns:
        return df_params[w_col_name].values.reshape(-1, 1)
    if default_name in df_params.columns:
        return df_params[default_name].values.reshape(-1, 1)
    return np.full((len(df_params), 1), 0.5)

def process_trip_type(trip_name, profile, df_rc, rc_ids, param_data_dict):
    c_file = CONSUMER_FILES[profile['consumer_file']]
    parquet_path = os.path.join(BASE_DIR, c_file)
    if not os.path.exists(parquet_path): 
        return None, None

    print(f"\nProcessing {trip_name} utility (Gravity Model Logic)")

    df_params_full = param_data_dict.get(profile['param_source'])
    all_scores = []
    all_meta = []
    total_processed = 0

    parquet_file = pq.ParquetFile(parquet_path)
    batch_iter = parquet_file.iter_batches(batch_size=BATCH_SIZE)
    
    # Pre-calculate Mass based on POI counts
    total_poi_vals = df_rc.loc[rc_ids, 'Total_POI_'].fillna(1).values
    
    amenity_matrix = {}
    for am in profile['amenities']:
        col = f"{am}_scaled" if f"{am}_scaled" in df_rc.columns else am
        amenity_matrix[am] = df_rc.loc[rc_ids, col].fillna(0).values

    for i, batch in enumerate(batch_iter):
        df_chunk = batch.to_pandas()
        batch_hh = df_chunk['household'].values
        df_params = df_params_full.reindex(batch_hh) if df_params_full is not None else pd.DataFrame(index=batch_hh)

        # 1. Base Amenity Attractiveness Calculation
        amenity_sum = np.zeros((len(batch_hh), len(rc_ids)))
        for am in profile['amenities']:
            w_col = f"W_{am}" if am != 'Personal and Professional Services' else 'W_personal_proffesional'
            amenity_sum += get_weight_value(w_col, w_col, df_params) * amenity_matrix[am]

        # Apply concentration bonus if requested
        if profile['bonus_percentage']:
            absolute_sum = np.sum([df_rc.loc[rc_ids, am].fillna(0).values for am in profile['amenities'] if am in df_rc.columns], axis=0)
            concentration = absolute_sum / np.where(total_poi_vals == 0, 1, total_poi_vals)
            amenity_sum *= (0.1 + concentration)

        # Scale and add Gravity Pull (Total POI)
        amenity_score = amenity_sum / np.maximum(np.nanmax(amenity_sum, axis=1, keepdims=True), 1.0)
        attractiveness = amenity_score * np.log1p(total_poi_vals)

        # 2. IMPROVEMENT: Mode-Specific Gravity Decay
        batch_utilities = {}
        base_beta = 1.0 / profile['distance_power'] 
        
        for trans_mode in ['Walk', 'Drive', 'PT']:
            if trans_mode not in df_chunk.columns: continue
            
            # Extract duration matrix
            dicts = df_chunk[trans_mode].apply(safe_eval).tolist()
            dur_mat = pd.DataFrame.from_records(dicts, index=batch_hh).reindex(columns=rc_ids).values.astype(float)
            
            # Apply friction adjustment for this specific mode
            mode_beta = base_beta * MODE_SENSITIVITY.get(trans_mode, 1.0)
            dist_penalty = np.exp(-mode_beta * dur_mat)
            
            # Utility = Attractiveness (Quality + Mass) * Distance Decay
            batch_utilities[trans_mode] = attractiveness * dist_penalty

        # Normalize total utilities for the batch
        all_u = np.stack(list(batch_utilities.values()))
        u_max_total = np.maximum(np.nanmax(all_u, axis=(0, 2)).reshape(-1, 1), 1e-9)
        
        mode_scores = []
        for trans_mode, u_mat in batch_utilities.items():
            final_u = u_mat / u_max_total
            mode_df = pd.DataFrame(final_u, index=batch_hh, columns=[f"{c}_{trans_mode.lower()}" for c in rc_ids])
            mode_scores.append(mode_df)
            
        all_scores.append(pd.concat(mode_scores, axis=1))
        all_meta.append(df_chunk[[c for c in CONSUMER_META_COLS if c in df_chunk.columns]].set_index('household'))
        total_processed += len(df_chunk)

    return pd.concat(all_meta), pd.concat(all_scores)

def save_output(metadata_df, scores_df, filename):
    final_df = metadata_df.join(scores_df, how='left')
    output_path = os.path.join(BASE_DIR, filename)
    pq.write_table(pa.Table.from_pandas(final_df.reset_index()), output_path)
    print(f"Saved: {output_path}")

def main():
    res = load_and_prep_retail_data()
    if res is None: return
    df_rc, rc_ids = res

    df_params_bulk = pd.read_parquet(BULK_PARAM_FILE).set_index('household') if os.path.exists(BULK_PARAM_FILE) else None
    df_params_con = pd.read_parquet(CON_PARAM_FILE).set_index('household') if os.path.exists(CON_PARAM_FILE) else None
    
    param_data_dict = {'bulk': df_params_bulk, 'con': df_params_con, 'avg': df_params_bulk}

    for trip_name, profile in TRIP_PROFILES.items():
        meta, scores = process_trip_type(trip_name, profile, df_rc, rc_ids, param_data_dict)
        if scores is not None:
            enriched_meta = apply_nts_enrichment(meta, NTS_PATH)
            save_output(enriched_meta, scores, f'utility_scores_{trip_name}.parquet')

if __name__ == "__main__":
    main()
