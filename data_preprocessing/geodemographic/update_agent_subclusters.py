import pandas as pd
import glob
import os
from pathlib import Path

def update_subclusters_in_data():
    """
    Joins the postcode-to-subcluster mapping onto all utility parquet files
    to permanently save geodemographic assignments.
    """
    # Paths
    base_dir = Path(__file__).parent
    mapping_csv = base_dir / "postcode_subcluster_map.csv"
    data_dir = Path(r"c:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\processed")
    
    if not mapping_csv.exists():
        print(f"Error: Mapping file not found at {mapping_csv}")
        return

    print(f"Loading mapping from {mapping_csv}...")
    mapping_df = pd.read_csv(mapping_csv)
    # Standardize column name for the join
    mapping_df = mapping_df.rename(columns={'Subcluster': 'Geo_Subcluster'})
    mapping_df['Postcode'] = mapping_df['Postcode'].astype(str).str.strip().str.upper()

    # Find all utility parquet files
    parquet_files = glob.glob(str(data_dir / "utility_scores_*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    for fpath in parquet_files:
        print(f"Processing {os.path.basename(fpath)}...")
        df = pd.read_parquet(fpath)
        
        # Ensure Postcode column exists and is formatted correctly
        if 'Postcode' not in df.columns:
            print(f"  Skipping: 'Postcode' column missing in {os.path.basename(fpath)}")
            continue
            
        df['Postcode'] = df['Postcode'].astype(str).str.strip().str.upper()
        
        # Drop existing Geo_Subcluster if it exists (to avoid duplicates)
        if 'Geo_Subcluster' in df.columns:
            df = df.drop(columns=['Geo_Subcluster'])
            
        # Merge
        updated_df = df.merge(mapping_df, on='Postcode', how='left')
        
        # Fill missing with 'none' or a default if necessary
        updated_df['Geo_Subcluster'] = updated_df['Geo_Subcluster'].fillna('none')
        
        # Save back
        updated_df.to_parquet(fpath, index=False)
        print(f"  Successfully updated {len(updated_df)} rows.")

    print("\nAll datasets updated with Geo_Subcluster labels.")

if __name__ == "__main__":
    update_subclusters_in_data()
