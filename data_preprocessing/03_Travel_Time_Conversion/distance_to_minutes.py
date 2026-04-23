import pandas as pd
import numpy as np
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

WALK_RESULTS = os.path.join(PROCESSED_DIR, 'walking_results.parquet')
DRIVE_RESULTS = os.path.join(PROCESSED_DIR, 'driving_results.parquet')
PT_RESULTS = os.path.join(PROCESSED_DIR, 'pt_accessibility_results.parquet')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'final_transport_times.parquet')

WALK_SPEED_KMH = 4.0
DRIVE_SPEED_KMH = 20.0

# Conversion factors: meters * (60 / (speed * 1000)) = minutes
WALK_FACTOR = 60.0 / (WALK_SPEED_KMH * 1000.0)    # 0.015
DRIVE_FACTOR = 60.0 / (DRIVE_SPEED_KMH * 1000.0)  # 0.003

def load_and_convert(input_file, factor):
    if not os.path.exists(input_file):
        print(f"Skipping {input_file} (not found).")
        return None

    print(f"Processing {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Standardize Postcode column
    df['Postcode'] = df['source'].astype(str).str.strip()
    
    def process_row(d):
        if not isinstance(d, dict): return {}
        new_dict = {}
        for k, v in d.items():
            if v is not None and np.isfinite(v):
                mins = v * factor
                # Rounding to 2 decimal places
                new_dict[k] = round(mins, 2)
        return new_dict

    df['minutes_dict'] = df['nearest_targets'].apply(process_row)
    return df[['Postcode', 'minutes_dict']]

def load_and_convert_pt(input_file):
    if not os.path.exists(input_file):
        print(f"Skipping {input_file} (not found).")
        return None

    print(f"Processing {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Standardize Postcode column
    df['Postcode'] = df['postcode'].astype(str).str.strip()
    
    # Calculate total time (vectorized operations are fast)
    df['total_time_mins'] = (
        df['postcode_to_pt_distance'] * WALK_FACTOR +
        df['pt_network_distance'] * DRIVE_FACTOR +
        df['retail_to_pt_distance'] * WALK_FACTOR
    )
    df['total_time_mins'] = df['total_time_mins'].round(2)
    
    # Group results per postcode into dictionaries (High-performance zip-loop)
    print("Grouping 9.7M PT results into dictionaries...")
    results_dict = {}
    
    # Zip is much faster than iterrows or groupby.apply for this scale
    postcodes = df['Postcode'].values
    retails = df['retail_centre'].astype(str).values
    times = df['total_time_mins'].values
    
    for pcd, rc, t in zip(postcodes, retails, times):
        if not np.isfinite(t): continue
        if pcd not in results_dict:
            results_dict[pcd] = {}
        results_dict[pcd][rc] = t
        
    # Convert to DataFrame
    print("Constructing PT dataframe...")
    pt_rows = []
    for pcd, d in results_dict.items():
        pt_rows.append({'Postcode': pcd, 'PT': d})
    
    return pd.DataFrame(pt_rows)

def main():
    print("Starting Specialized Transport Time Conversion (Walk, Drive, PT)...")
    
    # 1. Process Walk
    walk_data = load_and_convert(WALK_RESULTS, WALK_FACTOR)
    if walk_data is not None:
        walk_data = walk_data.rename(columns={'minutes_dict': 'Walk'})
        
    # 2. Process Drive
    drive_data = load_and_convert(DRIVE_RESULTS, DRIVE_FACTOR)
    if drive_data is not None:
        drive_data = drive_data.rename(columns={'minutes_dict': 'Drive'})
        
    # 3. Process PT
    pt_data = load_and_convert_pt(PT_RESULTS)
        
    # 4. Merge
    print("Merging modes...")
    dfs = [df for df in [walk_data, drive_data, pt_data] if df is not None]
    
    if not dfs:
        print("Error: No data found to convert.")
        return
        
    combined = dfs[0]
    for next_df in dfs[1:]:
        combined = pd.merge(combined, next_df, on='Postcode', how='outer')

    # 5. Final Data Integrity Check
    # Ensure any NaN/None resulting from outer merge is replaced with an empty dict
    for col in ['Walk', 'Drive', 'PT']:
        if col in combined.columns:
            combined[col] = combined[col].apply(lambda x: x if isinstance(x, dict) else {})

    # Save results
    print(f"Saving combined results to {OUTPUT_FILE}...")
    combined.to_parquet(OUTPUT_FILE, index=False)
    
    # Verify no None/NaN in result
    print("\n--- Final Verification ---")
    for col in ['Walk', 'Drive', 'PT']:
        if col in combined.columns:
            null_count = combined[col].isna().sum()
            print(f"Column '{col}': {null_count} None/NaN values found.")
            
    print("Conversion complete!")

if __name__ == "__main__":
    main()
