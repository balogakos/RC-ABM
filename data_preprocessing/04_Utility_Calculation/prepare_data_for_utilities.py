"""
Retail ABM - Fast Vectorized Data Preparation & Utility Calculation
"""

import pandas as pd
import numpy as np
import os
import ast
import gc
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
TEST_MODE = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

BATCH_SIZE = 1000

CONSUMER_FILES = {
    'bulk':        os.path.join(PROCESSED_DIR, 'consumer_agents_bulk.parquet'),
    'convenience': os.path.join(PROCESSED_DIR, 'consumer_agents_convenience.parquet'),
}
RETAIL_FILE = os.path.join(PROCESSED_DIR, 'retail_centres_processed.parquet')

BULK_PARAM_FILE = os.path.join(INPUT_DIR, 'Consumer_Agents', 'bulk_parameterised_consumer_agents_full.parquet')
CON_PARAM_FILE  = os.path.join(INPUT_DIR, 'Consumer_Agents', 'con_parameterised_consumer_agents_full.parquet')

NTS_PATH = os.path.join(INPUT_DIR, 'NTS', 'Cleaned_NTS_Data.csv')

# Now we define our 6 trip profiles
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
        'distance_power': 10.0,
        'bonus_percentage': False
    },
    'comparison': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Retail'],
        'bonus_percentage': True,
        'size_weight': 1.5
    },
    'entertainment': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Entertainment'],
        'bonus_percentage': True,
        'size_weight': 1.0
    },
    'food_drink': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Cafe', 'Restaurant'],
        'bonus_percentage': True,
        'size_weight': 1.0,
        'average_amenities': True
    },
    'service': {
        'consumer_file': 'convenience',
        'param_source': 'avg',
        'amenities': ['Personal and Professional Services'],
        'bonus_percentage': True,
        'size_weight': 1.0,
        'average_amenities': True
    }
}

ALL_AMENITIES = ['Retail', 'Cafe', 'Personal and Professional Services', 'Restaurant', 'Entertainment', 'Foodstore', 'Convenience Store']
CONSUMER_META_COLS = ['sex', 'age_years', 'nssec8', 'pwkstat', 'salary_yearly', 'id', 'household', 'num_children', 'household_size', 'Postcode']

# ---------------------------------------------------------------------------

def apply_nts_enrichment(meta_df, nts_path, seed=42):
    """Enriches the agents dataframe with NTS daily trip probabilities."""
    if not os.path.exists(nts_path):
        print(f"  WARNING: NTS file not found at {nts_path}. Skipping enrichment.")
        return meta_df
        
    print(f"  Enriching agents with NTS trip probabilities from {nts_path}...")
    
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
    retail_path = RETAIL_FILE
    if not os.path.exists(retail_path): return None

    df_rc = pd.read_parquet(retail_path)

    # Amenity columns are now pre-scaled via log-transformation in the preprocessor.
    amenity_cols = [
        'Foodstore', 'Convenience Store', 'Retail', 'Cafe', 
        'Personal and Professional Services', 'Restaurant', 'Entertainment'
    ]
    
    requested_cols = ['RC_ID', 'Classifica', 'Total_POI_'] + amenity_cols
    available_cols = [c for c in requested_cols if c in df_rc.columns]
    df_rc = df_rc[available_cols].copy()

    # RC_ID must be string for consistent lookup and standardised to strip .0
    df_rc['RC_ID'] = df_rc['RC_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Store in standard order
    rc_ids = df_rc['RC_ID'].tolist()
    return df_rc.set_index('RC_ID'), rc_ids

def get_weight_value(w_col_name, default_name, df_params):
    """Helper to extract parameter weights from df_params."""
    if w_col_name in df_params.columns:
        return df_params[w_col_name].values.reshape(-1, 1)
    if default_name in df_params.columns:
        return df_params[default_name].values.reshape(-1, 1)
    if 'W_total' in df_params.columns:
        return df_params['W_total'].values.reshape(-1, 1)
    return np.full((len(df_params), 1), 0.5)

    c_file = CONSUMER_FILES[profile['consumer_file']]
    parquet_path = c_file
    if not os.path.exists(parquet_path): 
        print(f"File not found: {parquet_path}")
        return None, None

    print(f"\nProcessing {trip_name} utility dataset (Source: {c_file})")

    df_params_full = param_data_dict.get(profile['param_source'])

    all_scores = []
    all_meta = []
    total_processed = 0

    parquet_file = pq.ParquetFile(parquet_path)
    batch_iter = parquet_file.iter_batches(batch_size=BATCH_SIZE)
    if TEST_MODE:
        batch_iter = iter([next(batch_iter)])

    # Setup Amenities Matrices
    amenity_matrix = {}      # The scaled score logic
    absolute_matrix = {}     # Used for concentration bonus
    total_poi = df_rc['Total_POI_'].fillna(1).values if 'Total_POI_' in df_rc.columns else np.ones(len(rc_ids))
    total_poi[total_poi == 0] = 1 # Avoid division by zero
    
    for am in profile['amenities']:
        scaled_col = f"{am}_scaled"
        if scaled_col in df_rc.columns:
            amenity_matrix[am] = df_rc.loc[rc_ids, scaled_col].fillna(0).values
        elif am in df_rc.columns:
            amenity_matrix[am] = df_rc.loc[rc_ids, am].fillna(0).values
            
        if profile['bonus_percentage'] and am in df_rc.columns:
            absolute_matrix[am] = df_rc.loc[rc_ids, am].fillna(0).values

    # --- New Smoothed Concentration Pre-processing ---
    conc_multiplier = np.ones(len(rc_ids))
    if profile['bonus_percentage']:
        # Step 1: Determine smoothing parameters from the dataset
        all_total_poi = df_rc['Total_POI_'].dropna()
        K_smoothing = 0.1 * np.median(all_total_poi) if len(all_total_poi) > 0 else 1.0
        
        # Expected share for this specific amenity set
        total_target_all = 0
        for am in profile['amenities']:
            if am in df_rc.columns:
                total_target_all += df_rc[am].sum()
        total_poi_all = df_rc['Total_POI_'].sum()
        average_share = total_target_all / total_poi_all if total_poi_all > 0 else 0
        k_smoothing = K_smoothing * average_share
        
        # Step 2 & 3: Apply smoothed concentration and diminishing returns
        # absolute_sum is count of target amenities in each centre
        absolute_sum = np.zeros(len(rc_ids))
        for am in absolute_matrix:
            absolute_sum += absolute_matrix[am]
            
        # adjusted_concentration = (absolute_sum + k) / (total_poi + K)
        adj_conc = (absolute_sum + k_smoothing) / (total_poi + K_smoothing)
        
        # multiplier = 0.5 + 0.5 * sqrt(adjusted_concentration)
        # Slightly reduced the bonus weight as per instructions
        conc_multiplier = 0.5 + 0.5 * np.sqrt(adj_conc)


    for i, batch in enumerate(batch_iter):
        df_chunk = batch.to_pandas()
        batch_hh = df_chunk['household'].values
        
        df_params = pd.DataFrame(index=batch_hh)
        if df_params_full is not None:
            df_params = df_params_full.reindex(batch_hh)
            
        # Add required default weights if missing
        for col in ['W_entertainment', 'W_restaurant']:
            if col not in df_params.columns:
                df_params[col] = 0.5
                
        # 1. Calculate Baseline Amenity Sum (or Average) for the batch
        amenity_components = []
        
        for am in profile['amenities']:
            w_col = f"W_{am}"
            if am == 'Personal and Professional Services':
                w_col = 'W_personal_proffesional'
            w_am = get_weight_value(w_col, w_col, df_params)
            
            if am in amenity_matrix:
                amenity_components.append(w_am * amenity_matrix[am])

        if not amenity_components:
            amenity_sum = np.zeros((len(batch_hh), len(rc_ids)))
        elif profile.get('average_amenities', False):
            # Requirements 2 & 3: Use average of components; if all are 0, result is 0
            amenity_sum = np.mean(amenity_components, axis=0)
        else:
            amenity_sum = np.sum(amenity_components, axis=0)

        # Apply the pre-calculated concentration multiplier
        # Combined Total Count (Requirement 1) is already in conc_multiplier
        amenity_sum = amenity_sum * conc_multiplier

        # --- SIZE ATTRACTION ADDITION ---
        # Add a bonus based on Total POI count if size_weight is defined
        size_weight = profile.get('size_weight', 0.0)
        if size_weight > 0:
            # Normalize total_poi relative to the max in the dataset
            max_p = np.nanmax(total_poi)
            size_score = np.log1p(total_poi) / np.log1p(max_p)
            # Add size bonus directly to the amenity sum before normalization
            amenity_sum += size_weight * size_score

        # Normalize Amenity Score per agent to [0, 1]
        amenity_max = np.nanmax(amenity_sum, axis=1, keepdims=True)
        amenity_max[amenity_max == 0] = 1.0
        amenity_score = amenity_sum / amenity_max

        # Collect transport data matrix (now using final_transport_times)
        transport_path = os.path.join(PROCESSED_DIR, 'final_transport_times.parquet')
        if not os.path.exists(transport_path):
            print(f"Warning: Transport times not found at {transport_path}")
            return None, None
            
        # Optimization: only load column needed for current agent batch
        # For simplicity in this refactor, we load and reindex
        df_trans = pd.read_parquet(transport_path)
        
        # Merge batch postcodes with transport times
        batch_df = df_chunk[['Postcode']].reset_index().merge(df_trans[['Postcode', 'Drive']], on='Postcode', how='left')
        
        dicts = batch_df['Drive'].apply(safe_eval).tolist()
        
        # REMAP OLD IDs TO NEW LEADER IDs if mapping exists
        mapping_path = os.path.join(PROCESSED_DIR, 'centre_merging_map.parquet')
        mapping_dict = {}
        if os.path.exists(mapping_path):
            mapping = pd.read_parquet(mapping_path)
            # Standardise both sides of the mapping
            mapping['RC_ID'] = mapping['RC_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
            mapping['Leader_RC_ID'] = mapping['Leader_RC_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
            mapping_dict = mapping.set_index('RC_ID')['Leader_RC_ID'].to_dict()

        # Optimization: Map IDs at the dictionary level before creating the DataFrame
        # This is much faster and avoids the 'axis=1 groupby' deprecation warning.
        remapped_dicts = []
        for d in dicts:
            new_d = {}
            for k, v in d.items():
                k_str = str(k).replace('.0', '')
                leader = mapping_dict.get(k_str, k_str)
                # If we have multiple components mapping to one leader, take the minimum duration
                if leader not in new_d or v < new_d[leader]:
                    new_d[leader] = v
            remapped_dicts.append(new_d)

        # Create DataFrame and reindex to match simulation's retail centres
        dur_df = pd.DataFrame.from_records(remapped_dicts, index=batch_hh)
        dur_mat = dur_df.reindex(columns=rc_ids).values.astype(float)

        # 3. Calculate Final Utilities with Exponential Decay
        # h_drive is the half-life parameter for driving (minutes).
        h_drive = 5.0
        
        # t_im^min is the minimum travel time from agent i to any centre.
        with np.errstate(all='ignore'):
            t_im_min = np.nanmin(dur_mat, axis=1, keepdims=True)
        t_im_min[np.isnan(t_im_min)] = 0.0
        
        # exponential decay logic: T = exp( - (1.5 * ln(2))/h_m (t_ij - t_i^min))
        travel_indicator = np.exp(- (1.5 * np.log(2) / h_drive) * (dur_mat - t_im_min))
        travel_indicator[np.isnan(travel_indicator)] = 0.0
        
        # final utility: T * amenity_score
        u_mat = travel_indicator * amenity_score

        # Global Normalization: Scale scores for this agent to [0, 1]
        u_max = np.nanmax(u_mat, axis=1, keepdims=True)
        u_max[u_max == 0] = 1.0
        final_u = u_mat / u_max
        
        scores_df = pd.DataFrame(final_u, index=batch_hh, columns=rc_ids)
        
        # Keep only metadata safely
        base_cols = [c for c in CONSUMER_META_COLS if c in df_chunk.columns]
        meta_df = df_chunk[base_cols].set_index('household')
        meta_df = meta_df.join(df_params, lsuffix='_x', rsuffix='_y')
        
        meta_df = meta_df.loc[:, ~meta_df.columns.str.endswith('_y')]
        meta_df.columns = [c[:-2] if c.endswith('_x') else c for c in meta_df.columns]
        meta_df = meta_df[~meta_df.index.duplicated(keep='first')]

        all_scores.append(scores_df)
        all_meta.append(meta_df)
        total_processed += len(df_chunk)

        if i % 10 == 0:
            print(f"  Batch {i}: {total_processed:,} households processed")

    if not all_scores:
        return None, None

    print("Combining batches...")
    final_scores = pd.concat(all_scores)
    final_meta = pd.concat(all_meta)
    
    if final_scores.index.duplicated().any():
        final_scores = final_scores.groupby(level=0).max()
        final_meta = final_meta[~final_meta.index.duplicated(keep='first')]

    return final_meta, final_scores

def save_output(metadata_df, scores_df, filename):
    if metadata_df is None or scores_df is None: return
    
    metadata_df.columns = metadata_df.columns.astype(str)
    scores_df.columns = scores_df.columns.astype(str)
    final_df = metadata_df.join(scores_df, how='left')
    
    output_dir = os.path.join(PROCESSED_DIR, 'testing') if TEST_MODE else PROCESSED_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    try:
        final_table = pa.Table.from_pandas(final_df.reset_index())
        pq.write_table(final_table, output_path)
        print(f"Saved to {output_path} (Shape: {final_df.shape})")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

def main():
    print(f"Building Trip-Specific Datasets. TEST_MODE={TEST_MODE}, BATCH_SIZE={BATCH_SIZE}")

    res = load_and_prep_retail_data()
    if res is None: return
    df_rc, rc_ids = res

    # Load agent parameters
    print("Loading Agent Parameters...")
    df_params_bulk, df_params_con, df_params_avg = None, None, None
    if os.path.exists(BULK_PARAM_FILE):
        df_params_bulk = pd.read_parquet(BULK_PARAM_FILE).set_index('household')
    if os.path.exists(CON_PARAM_FILE):
        df_params_con = pd.read_parquet(CON_PARAM_FILE).set_index('household')
        
    if df_params_bulk is not None and df_params_con is not None:
        numeric_cols = df_params_bulk.select_dtypes(include=[np.number]).columns
        df_params_avg = df_params_bulk.copy()
        
        # Only average the numeric columns (weights, thresholds, etc.)
        common_numeric = [c for c in numeric_cols if c in df_params_con.columns]
        df_params_avg[common_numeric] = (df_params_bulk[common_numeric] + df_params_con[common_numeric]) / 2.0
    else:
        df_params_avg = df_params_bulk if df_params_bulk is not None else df_params_con

    param_data_dict = {
        'bulk': df_params_bulk,
        'con': df_params_con,
        'avg': df_params_avg
    }

    # Generate a dataset for each profile
    saved_paths = []
    last_meta_avg = None
    
    for trip_name, profile in TRIP_PROFILES.items():
        meta, scores = process_trip_type(trip_name, profile, df_rc, rc_ids, param_data_dict)
        if scores is not None:
            # We ONLY enrich the baseline consumer data with NTS trips before saving
            meta_enriched = apply_nts_enrichment(meta, NTS_PATH)
            
            output_name = f'utility_scores_{trip_name}.parquet'
            save_output(meta_enriched, scores, output_name)
            saved_paths.append(output_name)
            last_meta_avg = meta_enriched

    print("\nPIPELINE COMPLETE.")
    print("Generated files:")
    for path in saved_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    main()
