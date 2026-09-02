import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path

def aggregate_retail_performance(input_path, output_path=None):
    """
    Reads large simulation results in batches and aggregates performance metrics
    per Retail Centre.
    """
    if not os.path.exists(input_path):
        print(f"Error: Results file not found at {input_path}")
        return

    print(f"Starting batch aggregation for: {os.path.basename(input_path)}")
    
    # Initialize a list to hold the batch-level aggregations
    # We will aggregate within each batch first to keep the intermediate size small
    batch_summaries = []
    
    # Open the parquet file for streaming
    pf = pq.ParquetFile(input_path)
    
    total_rows = pf.metadata.num_rows
    batch_size = 200000  # Process 200k rows at a time
    processed_rows = 0

    print(f"Total rows to process: {total_rows:,}")

    # Iterate through batches
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        
        # 1. Basic counts and averages
        # Group by Retail_Centre and aggregate numeric columns
        summary = df.groupby('Retail_Centre').agg({
            'AgentID': 'count',
            'Utility_Score': 'sum',
            'Utility_Modifier': 'sum'
        }).rename(columns={'AgentID': 'Total_Visits'})
        
        # 2. Trip Type Breakdown (Pivoted)
        if 'Trip_Type' in df.columns:
            trip_counts = df.pivot_table(
                index='Retail_Centre', 
                columns='Trip_Type', 
                values='AgentID', 
                aggfunc='count', 
                fill_value=0
            )
            summary = summary.join(trip_counts, how='left')

        # 3. Transport Mode Breakdown (Pivoted)
        if 'Transport_Mode' in df.columns:
            mode_counts = df.pivot_table(
                index='Retail_Centre', 
                columns='Transport_Mode', 
                values='AgentID', 
                aggfunc='count', 
                fill_value=0
            )
            summary = summary.join(mode_counts, how='left', rsuffix='_mode')

        # 4. Geodemographic Breakdown (Pivoted)
        if 'Geo_Subcluster' in df.columns:
            geo_counts = df.pivot_table(
                index='Retail_Centre', 
                columns='Geo_Subcluster', 
                values='AgentID', 
                aggfunc='count', 
                fill_value=0
            )
            summary = summary.join(geo_counts, how='left', rsuffix='_cluster')

        batch_summaries.append(summary)
        
        processed_rows += len(df)
        if processed_rows % (batch_size * 5) == 0 or processed_rows == total_rows:
            print(f"  Processed {processed_rows:,} / {total_rows:,} rows...")

    # Final Aggregation of all batches
    print("\nFinalising global aggregation...")
    final_df = pd.concat(batch_summaries).groupby(level=0).sum()
    
    # Calculate final averages
    final_df['Avg_Utility_Score'] = final_df['Utility_Score'] / final_df['Total_Visits']
    final_df['Avg_Utility_Modifier'] = final_df['Utility_Modifier'] / final_df['Total_Visits']
    final_df = final_df.drop(columns=['Utility_Score', 'Utility_Modifier'])

    # --- AUTOMATIC METADATA MERGE ---
    print("Enriching results with Retail Centre metadata (POIs)...")
    try:
        import geopandas as gpd
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation'))
        import config
        retail_data = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')

        # Clean IDs for matching
        def clean_id(x):
            s = str(x).strip()
            return s[:-2] if s.endswith('.0') else s

        # Calculate POI metrics
        amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services', 
                        'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
        existing_cols = [col for col in amenity_cols if col in retail_data.columns]
        retail_data['Total_POIs'] = retail_data[existing_cols].sum(axis=1)
        
        # Prepare for merge
        id_col = next((c for c in ['RC_ID', 'rc_id', 'ID'] if c in retail_data.columns), 'RC_ID')
        retail_data[id_col] = retail_data[id_col].apply(clean_id)
        
        # Align simulation IDs
        final_df.index = [clean_id(i) for i in final_df.index]
        final_df.index.name = 'RC_ID'
        
        # Merge
        meta_df = retail_data[[id_col, 'Total_POIs'] + existing_cols].set_index(id_col)
        final_df = final_df.join(meta_df, how='left')
        print(f"  Successfully attached metadata for {final_df['Total_POIs'].notna().sum()} centres.")
    except Exception as e:
        print(f"  Warning: Could not merge metadata. Error: {e}")

    # Save results
    if output_path is None:
        output_path = input_path.replace('.parquet', '_aggregated.csv')
    
    final_df.to_csv(output_path)
    print(f"Success! Final enriched data saved to: {output_path}")
    return final_df

if __name__ == "__main__":
    target_file = r"C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\multi_runs\results\run_2.parquet"
    aggregate_retail_performance(target_file)
