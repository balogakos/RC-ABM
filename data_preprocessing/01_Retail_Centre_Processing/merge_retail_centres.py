import geopandas as gpd
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

INPUT_GPKG = os.path.join(PROCESSED_DIR, "retail_centre_type_counts.gpkg")
LAYER_NAME = 'retail_centre_counts'
OUTPUT_GPKG = os.path.join(PROCESSED_DIR, "retail_centre_type_counts_merged.gpkg")

# Distance threshold for "2 H3 squares" (Resolution 10 is ~130m diameter)
# We will merge if the gap between centres is <= DISTANCE_THRESHOLD
DISTANCE_THRESHOLD = 10

AMENITY_COLS = [
    'Cafe', 'Convenience Store', 'Entertainment', 'Foodstore',
    'Personal Service', 'Professional Services', 'Restaurant', 'Retail',
    'Total_POI_', 'Dining_and', 'Leisure_Tr', 'Community_', 'Active', 'H3_cells'
]

def merge_retail_centres():
    if not os.path.exists(INPUT_GPKG):
        print(f"Error: Input file not found at {INPUT_GPKG}")
        return

    print(f"Loading retail centres from {INPUT_GPKG}...")
    gdf = gpd.read_file(INPUT_GPKG, layer=LAYER_NAME)
    print(f"Loaded {len(gdf)} records.")

    # 1. Spatial Clustering
    print(f"Clustering centres within {DISTANCE_THRESHOLD}m of each other...")
    
    # We buffer by half the threshold so that if the gap is <= threshold, the buffers touch/overlap
    buffered = gdf.copy()
    buffered['geometry'] = buffered.geometry.buffer(DISTANCE_THRESHOLD / 2.0)

    # Use spatial join to find all intersecting buffered geometries
    # This finds which centres should belong to the same cluster
    adj = gpd.sjoin(buffered, buffered, how='inner', predicate='intersects')
    
    # Build an adjacency matrix for connected components
    row = adj.index.values
    col = adj['index_right'].values
    data = np.ones(len(row))
    graph = csr_matrix((data, (row, col)), shape=(len(gdf), len(gdf)))
    
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    gdf['cluster_id'] = labels
    print(f"Identified {n_components} unique clusters.")

    # 2. Aggregation
    print("Aggregating amenity counts and geometries...")
    
    # Define aggregation rules
    agg_rules = {col: 'sum' for col in AMENITY_COLS if col in gdf.columns}
    # For RC_ID, we'll keep the one with the most POIs as the primary ID
    
    # Sort by POI count so the first record in each group is the "biggest" centre
    gdf = gdf.sort_values(by=['cluster_id', 'Total_POI_'], ascending=[True, False])
    
    # Group by cluster
    def aggregate_geometries(geoms):
         return geoms.unary_union

    # Actually, we aggregate counts separately
    merged_counts = gdf.groupby('cluster_id')[list(agg_rules.keys())].sum()
    
    # Join back the primary metadata (RC_ID, category_n, etc. from the largest centre in cluster)
    primary_metadata = gdf.groupby('cluster_id').first()[['RC_ID', 'category_n', 'centre_typ', 'LocalAutho']]
    
    # Dissolve geometries and apply a "Closing" operation to fill gaps
    print("Bridging gaps in geometries (Spatial Closure)...")
    # Buffer out, Union, then Buffer back in
    merged_geoms = gdf.dissolve(by='cluster_id').geometry
    # Use half the threshold to bridge the exact gap used for clustering
    bridge_dist = DISTANCE_THRESHOLD / 2.0
    merged_geoms = merged_geoms.buffer(bridge_dist).buffer(-bridge_dist)
    
    # Combine everything
    merged_gdf = gpd.GeoDataFrame(
        pd.concat([primary_metadata, merged_counts], axis=1),
        geometry=merged_geoms,
        crs=gdf.crs
    )

    # Recalculate score (simple average or sum?) — keeping it as sum of POIs for now
    # If the original 'score' was normalized, it might need recalculation.
    
    print(f"Final centre count: {len(merged_gdf)}")
    
    # 3. Save
    print(f"Saving to {OUTPUT_GPKG}...")
    merged_gdf.to_file(OUTPUT_GPKG, layer=LAYER_NAME, driver="GPKG")
    print("Merge complete.")

if __name__ == "__main__":
    merge_retail_centres()
