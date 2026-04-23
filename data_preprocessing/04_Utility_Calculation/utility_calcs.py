
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

BASE_DIR = PROCESSED_DIR
BULK_INPUT = 'test_consumer_agents_bulk_prepared.parquet'
CON_INPUT = 'test_consumer_agents_convenience_prepared.parquet'
OUTPUT_BULK = 'utility_scores_bulk.parquet'
OUTPUT_CON = 'utility_scores_convenience.parquet'
OUTPUT_AVG = 'utility_scores_average.parquet'

# Amenities for utility calculation (Weighted Sum)
# Excludes Foodstore and Convenience Store from this calculation
AMENITIES = [
    'Retail',
    'Cafe',
    'Personal and Professional Services',
    'Restaurant',
    'Entertainment'
]

# Helper: All column names that vary per Retail Centre (to exclude from metadata)
# Includes amenities, duration, mode, etc.
TRIP_SPECIFIC_COLS = [
    'Retail_Centre', 'Utility_Score', 'Duration', 'Duration_Scaled', 'Mode', 
    'Classifica', 'Total', 'Distance',
    'Foodstore', 'Convenience Store', 'Retail', 'Cafe', 
    'Personal and Professional Services', 'Restaurant', 'Entertainment'
]

def calculate_utility_column(df, mode_type):
    """
    Calculates utility score for each row and returns the Series.
    mode_type: 'bulk' or 'convenience'
    """
    print(f"Calculating utility for {mode_type}...")
    
    # helper to safely get column or 0
    def get_col(col_name):
        if col_name in df.columns:
            return df[col_name].fillna(0.0)
        else:
            return 0.0

    # 1. Base Utility: Distance
    # Use Duration_Scaled if available, else Duration
    w_dist_col = 'W_distance'
    for c in df.columns:
        if c.lower() == 'w_distance':
            w_dist_col = c
            break

    duration_col = 'Duration_Scaled' if 'Duration_Scaled' in df.columns else 'Duration'
    
    dist_score = get_col(duration_col) * get_col(w_dist_col)
    
    # 2. Amenity Scores (Weighted Sum)
    amenity_sum = 0.0
    
    col_overrides = {
        'Personal and Professional Services': 'W_personal_proffesional',
    }
    col_lower_map = {c.lower(): c for c in df.columns}

    for amenity in AMENITIES:
        w_col_name = None
        if amenity in col_overrides and col_overrides[amenity] in df.columns:
            w_col_name = col_overrides[amenity]
        else:
            target = f"W_{amenity}"
            if target in df.columns:
                w_col_name = target
            elif target.lower() in col_lower_map:
                w_col_name = col_lower_map[target.lower()]

        if w_col_name:
            weight = get_col(w_col_name)
        else:
            weight = get_col(f"W_{amenity}")

        # Use RAW amenity value (assuming input is count or binary)
        val = get_col(amenity)
        amenity_sum += val * weight

    # 3. Multiplicative Logic (Dist * Amenities)
    total_utility = dist_score * amenity_sum
    
    # 4. Binary Filter (Foodstore/Convenience Store)
    if mode_type == 'bulk':
        fs_val = get_col('Foodstore')
        total_utility *= (fs_val > 0).astype(float)
        
    elif mode_type == 'convenience':
        cs_val = get_col('Convenience Store')
        total_utility *= (cs_val > 0).astype(float)

    return total_utility

def process_dataframe(filename, mode_type):
    """
    Reads file, calculates utility, pivots (Household x Retail_Centre), and returns (Metadata_DF, Scores_DF).
    """
    input_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return None, None

    print(f"\nProcessing {filename} for {mode_type}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None, None

    print(f"Loaded {len(df)} rows.")

    # Calculate Utility
    df['Utility_Score'] = calculate_utility_column(df, mode_type)
    
    # Identify unique ID column (Household)
    id_col = 'household'
    if id_col not in df.columns:
        if 'id' in df.columns:
            id_col = 'id'
        else:
            print("Error: Could not find 'household' or 'id' column.")
            return None, None
            
    if 'Retail_Centre' not in df.columns:
        print("Error: 'Retail_Centre' column not found.")
        return None, None
        
    # --- Pivot: Scores ---
    # Index: Household, Columns: Retail_Centre, Values: Utility_Score
    # Aggfunc='max' handles duplicates (e.g. multi-mode trips to same center -> take best utility)
    print(f"Pivoting scores (Index={id_col})...")
    scores_df = df.pivot_table(index=id_col, columns='Retail_Centre', values='Utility_Score', aggfunc='max')
    
    # --- Extract Metadata ---
    # Keep columns that are NOT trip-specific (i.e. household attributes)
    # We drop columns that vary per trip
    
    # Build list of columns to drop: present in df AND present in TRIP_SPECIFIC_COLS
    cols_to_drop = [c for c in TRIP_SPECIFIC_COLS if c in df.columns]
    
    # Group by household and take first
    metadata_df = df.drop(columns=cols_to_drop).groupby(id_col).first()
    
    return metadata_df, scores_df

def save_output(metadata_df, scores_df, filename):
    """
    Joins Metadata + Scores and Saves.
    """
    if metadata_df is None or scores_df is None:
        return

    # Join on index (Household ID)
    final_df = metadata_df.join(scores_df, how='left') # Left join to keep all households in metadata

    output_path = os.path.join(BASE_DIR, filename)
    try:
        # Convert index to column if desired (standard parquet behavior often keeps index, but user might want 'household' as col)
        # reset_index makes 'household' a column again if it was index
        final_table = pa.Table.from_pandas(final_df.reset_index())
        pq.write_table(final_table, output_path)
        print(f"Saved to {output_path} (Shape: {final_df.shape})")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

def main():
    # 1. Process Bulk
    meta_bulk, scores_bulk = process_dataframe(BULK_INPUT, 'bulk')
    
    # 2. Process Convenience
    meta_con, scores_con = process_dataframe(CON_INPUT, 'convenience')
    
    if scores_bulk is None or scores_con is None:
        print("Error: Could not process inputs.")
        return

    # 3. Save Bulk Output
    print("\nSaving Bulk Output...")
    save_output(meta_bulk, scores_bulk, OUTPUT_BULK)
    
    # 4. Save Convenience Output
    print("\nSaving Convenience Output...")
    save_output(meta_con, scores_con, OUTPUT_CON)
    
    # 5. Calculate Average Utility
    print("\nCalculating Average Utility...")
    
    # Align DataFrames on index (Household) and Columns (Retail Centres)
    # 1. Align Columns: Find union of columns (retail centres)
    all_centres = sorted(list(set(scores_bulk.columns) | set(scores_con.columns)))
    
    # Reindex both to have all columns, fillna(0) for missing centres?
    # Utility usually 0 if not reachable.
    scores_bulk_aligned = scores_bulk.reindex(columns=all_centres, fill_value=0.0)
    scores_con_aligned = scores_con.reindex(columns=all_centres, fill_value=0.0)
    
    # Align Rows (Households)
    # Intersection or Union? Assume Union of households.
    all_households = sorted(list(set(scores_bulk.index) | set(scores_con.index)))
    scores_bulk_aligned = scores_bulk_aligned.reindex(index=all_households, fill_value=0.0)
    scores_con_aligned = scores_con_aligned.reindex(index=all_households, fill_value=0.0)
    
    # Calculate Average
    scores_avg = (scores_bulk_aligned + scores_con_aligned) / 2.0
    
    # Metadata for Average? 
    # Use Bulk metadata aligned to households.
    # If a household exists in Conv but not Bulk, we might miss metadata if we only use meta_bulk.
    # Better to combine metadata.
    # meta_bulk and meta_con should ideally have same columns.
    # Let's align meta_bulk to all_households.
    meta_avg = meta_bulk.reindex(index=all_households)
    # Fill missing from meta_con if possible?
    meta_avg = meta_avg.combine_first(meta_con.reindex(index=all_households))
    
    # Save Average Output
    print("Saving Average Output...")
    save_output(meta_avg, scores_avg, OUTPUT_AVG)

if __name__ == "__main__":
    main()
