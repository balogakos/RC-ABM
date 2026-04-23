import pandas as pd
import numpy as np
import os
import geopandas as gpd

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

utility_dir = PROCESSED_DIR
retail_path = os.path.join(PROCESSED_DIR, 'retail_centre_counts.parquet')

def process_data():
    print("Starting data processing...")
    
    # --- 1. PROCESS RETAIL CENTRE DATA ---
    # We now load from GeoPackage to get boundaries for spatial merging
    gpkg_path = os.path.join(PROCESSED_DIR, 'retail_centre_type_counts.gpkg')
    mapping_out = os.path.join(utility_dir, 'centre_merging_map.parquet')
    
    if not os.path.exists(gpkg_path):
        print(f"Error: Could not find retail centre GPKG at {gpkg_path}")
        return

    # Load from the counts layer
    gdf_retail = gpd.read_file(gpkg_path, layer='retail_centre_counts')
    print(f"Loaded retail data: {len(gdf_retail)} centres")

    # Standardise IDs to string
    gdf_retail['RC_ID'] = gdf_retail['RC_ID'].astype(str).replace(r'\.0$', '', regex=True)

    # --- 2. SPATIAL MERGING ---
    print("Performing spatial merge (touching or within 50m)...")
    
    # Ensure metric projection for accurate buffering (EPSG:27700 is standard for UK)
    original_crs = gdf_retail.crs
    if gdf_retail.crs is None or not gdf_retail.crs.is_projected:
        print("  Warning: CRS is not projected. Using EPSG:27700 for buffering.")
        gdf_retail = gdf_retail.to_crs(epsg=27700)
    
    # Identify clusters: buffer by 25m (bridging 50m gaps), then find connected components
    buffered = gdf_retail.copy()
    buffered['geometry'] = buffered.geometry.buffer(25)
    
    # Use unary_union and explode to get individual cluster polygons
    clusters_geom = buffered.unary_union
    clusters_gdf = gpd.GeoDataFrame(geometry=list(clusters_geom.geoms if hasattr(clusters_geom, 'geoms') else [clusters_geom]), crs=gdf_retail.crs)
    clusters_gdf['cluster_id'] = range(len(clusters_gdf))
    
    # Join original centres to clusters
    gdf_retail = gpd.sjoin(gdf_retail, clusters_gdf, how='left', predicate='intersects')
    
    # Determine the 'Leader' for each cluster (largest centre by Total_POI_)
    # Note: total_poi_col is handle dynamically below, assuming it's available.
    total_poi_col = [c for c in gdf_retail.columns if c.startswith('Total_POI_')][0]
    
    # Aggregate mapping
    mapping = []
    merged_records = []
    
    for cid, group in gdf_retail.groupby('cluster_id'):
        if len(group) == 1:
            leader = group.iloc[0].copy()
            mapping.append({'RC_ID': leader['RC_ID'], 'Leader_RC_ID': leader['RC_ID']})
            merged_records.append(leader)
            continue
            
        leader_idx = group[total_poi_col].idxmax()
        leader = group.loc[leader_idx].copy()
        
        # Mapping for all components to the leader
        for _, row in group.iterrows():
            mapping.append({'RC_ID': row['RC_ID'], 'Leader_RC_ID': leader['RC_ID']})
            
        # Sum amenity counts logic
        amenities = [
            'Foodstore', 'Convenience Store', 'Retail', 'Cafe',
            'Personal Services', 'Professional Services', 'Restaurant', 'Entertainment',
            total_poi_col
        ]
        for am in amenities:
            if am in group.columns:
                leader[am] = group[am].sum()
        
        merged_records.append(leader)

    df_retail = pd.DataFrame(merged_records)
    df_mapping = pd.DataFrame(mapping)
    
    # Save mapping for visualization
    df_mapping.to_parquet(mapping_out)
    print(f"  Merged {len(gdf_retail)} centres into {len(df_retail)} mega-centres.")
    print(f"  Mapping saved to: {mapping_out}")

    # --- 3. COMBINE SERVICES AND CONTINUE ---
    print("Aggregating services and scaling...")
    service_cols = ['Personal Services', 'Professional Services']
    available_services = [c for c in service_cols if c in df_retail.columns]
    df_retail['Personal and Professional Services'] = df_retail[available_services].sum(axis=1)

    # Remove 'Active' column/POIs and original service columns
    cols_to_drop = ['Active', 'Active_POIs'] + available_services
    df_retail = df_retail.drop(columns=[c for c in cols_to_drop if c in df_retail.columns], errors='ignore')

    # Ensure the main POI column is named 'Total_POI_' for the utility script
    if total_poi_col != 'Total_POI_':
        df_retail = df_retail.rename(columns={total_poi_col: 'Total_POI_'})
        total_poi_col = 'Total_POI_'
        
    df_retail['Classifica'] = pd.qcut(df_retail[total_poi_col], 10, labels=False, duplicates='drop')

    # Scaling within Classifica groups
    columns_to_scale = [
        'Foodstore', 'Convenience Store', 'Retail', 'Cafe',
        'Personal and Professional Services', 'Restaurant', 'Entertainment'
    ]

    first_binary_cols = ['Foodstore', 'Convenience Store']
    
    for col in columns_to_scale:
        if col in df_retail.columns:
            # 1. Apply log transformation to dampen outliers
            log_vals = np.log1p(df_retail[col])
            
            # 2. Scale within Classifica groups: result = log(1+count) / log(1+max_in_group)
            # This ensures that as long as count > 0, the final value is > 0.
            g_max_log = df_retail.groupby('Classifica')[col].transform(lambda x: np.log1p(x.max()))
            
            # Replace original column with scaled version
            df_retail[col] = (log_vals / g_max_log).fillna(0)
            
            # Replace original column with scaled version
            df_retail[col] = (log_vals / g_max_log).fillna(0)

    # Scale Total_POI_ globally (0 to 1)
    if total_poi_col in df_retail.columns:
        min_val = df_retail[total_poi_col].min()
        max_val = df_retail[total_poi_col].max()
        range_val = max_val - min_val
        
        new_col_name = f'{total_poi_col}_scaled'
        if range_val != 0:
            df_retail[new_col_name] = (df_retail[total_poi_col] - min_val) / range_val
        else:
            df_retail[new_col_name] = 0.0
        print(f"Created globally scaled column: {new_col_name}")

    # Drop columns that aren't needed for simulation or cause parquet errors
    cols_to_drop = ['geometry', 'cluster_id', 'index_right']
    df_retail = df_retail.drop(columns=[c for c in cols_to_drop if c in df_retail.columns])
    
    # Save processed retail centre data
    retail_out = os.path.join(utility_dir, 'retail_centres_processed.parquet')
    df_retail.to_parquet(retail_out)
    print(f"Saved retail data to: {retail_out}")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    process_data()
