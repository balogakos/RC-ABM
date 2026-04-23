
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
import os
import pickle
import time

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LSOA_PATH = os.path.join(INPUT_DIR, 'Liverpool Boundary', 'CA boundary.shp')
POSTCODE_PATH = os.path.join(INPUT_DIR, 'Liverpool Postcode', 'postcode_CA_new.shp')
RETAIL_PATH = os.path.join(INPUT_DIR, 'Liverpool Retail Centres', 'Liverpool_new_retail_centres_01-26(version).shp')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'walking_results.parquet')
CACHE_FILE = os.path.join(CACHE_DIR, 'walking_network_cache.pkl')
TEST_MODE = False  # Set to True to only process the first 100 postcodes

PROJECTED_CRS = 'EPSG:27700'
NETWORK_CRS = 'EPSG:4326'  # Original graph CRS

def connect_points_to_graph(G, points_gdf, id_col, label_prefix, target_crs):
    """
    Connects points from a GeoDataFrame to the nearest nodes in the graph G.
    Uses cKDTree for fast nearest neighbor lookup.
    """
    print(f"Connecting {label_prefix} to network...")
    
    # helper to ensure we work with a consistent projection for distance calculation
    # Ideally we project everything to local grid (27700) for distance, but G is likely 4326 initially.
    # The graph G from ox.graph_from_polygon is usually unprojected (lat/lon).
    # We need to project the points to match the graph's coordinate system for matching.
    
    # Get graph node coordinates
    node_ids = list(G.nodes())
    node_coords = np.array([(data['x'], data['y']) for node, data in G.nodes(data=True)])
    
    # Build KDTree for fast lookup
    tree = cKDTree(node_coords)
    
    # Extract point coordinates (ensure they are in the same CRS as the graph nodes)
    # We need points in the same CRS as the graph 'x' and 'y'
    points_gdf_proj = points_gdf.to_crs(target_crs)
    
    # Use centroids in case of Polygons (safe for Points too)
    point_coords = np.array([(p.centroid.x, p.centroid.y) for p in points_gdf_proj.geometry])
    
    # Query nearest nodes
    # k=1 returns (distances, indices)
    distances, indices = tree.query(point_coords, k=1)
    
    # Add nodes and edges to G
    count = 0
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        # Point ID
        if id_col in points_gdf.columns:
            p_id = str(points_gdf.iloc[i][id_col]).strip()
        else:
            p_id = f"{label_prefix}_{i}"
            
        nearest_node_id = node_ids[idx]
        
        # Add the point as a new node
        # Use projected coords if available in original GDF, or just use the ones we extracted
        # Here we add it with its original coords (lat/lon)
        pt = points_gdf_proj.geometry.iloc[i]
        # Handle Polygons by using centroid for the node position
        cx, cy = pt.centroid.x, pt.centroid.y
        G.add_node(p_id, x=cx, y=cy, type=label_prefix)
        
        # Add bidirectional edge with length of 0 (or actual dist)
        # Using 0 approximates that they are "on" the network for this analysis, 
        # or we could project to calculate meters. Notebook logic used simple Euclidean check.
        # We will add an edge with 'length' = 0 to effectively snap it.
        if nearest_node_id != p_id:
            G.add_edge(p_id, nearest_node_id, length=0)
            G.add_edge(nearest_node_id, p_id, length=0)
            count += 1
            
    print(f"Connected {count} {label_prefix} points to the network.")
    return G

def from_nx_to_igraph(Mg, weight_attr='length'):
    print("Converting NetworkX graph to igraph...")
    # Mapping to integers
    mapping = {node: i for i, node in enumerate(Mg.nodes())}
    inverse_mapping = {i: node for node, i in mapping.items()}
    
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(mapping))
    
    edges = []
    weights = []
    
    for u, v, data in Mg.edges(data=True):
        if weight_attr in data:
            edges.append((mapping[u], mapping[v]))
            weights.append(data[weight_attr])
        else:
            # Handle edges without length (shouldn't happen for network edges if projected correctly)
             edges.append((mapping[u], mapping[v]))
             weights.append(0)

    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights
    
    return ig_graph, mapping

def run_analysis():
    # --- Caching Logic ---
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached network from {CACHE_FILE}...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                G, retail, postcodes = pickle.load(f)
            print("Loaded data from cache successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding network...")
            G = None
    else:
        G = None

    if G is None:
        # 1. Load Data
        print("Loading shapefiles...")
        lsoa = gpd.read_file(LSOA_PATH)
        postcodes = gpd.read_file(POSTCODE_PATH)
        retail = gpd.read_file(RETAIL_PATH)
        
        # 2. Build Road Network
        print("Building walking network from polygon...")
        boundary_polygon = lsoa.geometry.union_all()
        
        # Ensure boundary is 4326 for OSMnx
        if lsoa.crs != "EPSG:4326":
            boundary_s = gpd.GeoSeries([boundary_polygon], crs=lsoa.crs).to_crs("EPSG:4326")
            boundary_polygon = boundary_s.iloc[0]
            
        # CHANGED: network_type="walk"
        G = ox.graph_from_polygon(boundary_polygon, network_type="walk")
        print(f"Initial graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # 3. Project Graph (for accurate length calculations on road edges)
        # The notebook projects, simplifies, then re-adds points.
        
        print(f"Projecting graph to {PROJECTED_CRS}...")
        G = ox.project_graph(G, to_crs=PROJECTED_CRS)
        
        # Ensure all edges have length (OSMnx projects usually add it, but safety check)
        for u, v, data in G.edges(data=True):
            if 'length' not in data or pd.isna(data.get('length')):
                data['length'] = 0

        # 4. Prepare Data (Clip and Project)
        
        # Ensure Boundary is in Projected CRS for clipping
        if lsoa.crs != PROJECTED_CRS:
            lsoa_proj = lsoa.to_crs(PROJECTED_CRS)
        else:
            lsoa_proj = lsoa
        
        boundary_shape = lsoa_proj.geometry.union_all()
        
        # Clip Retail Centers to Boundary (User Request)
        print("Clipping Retail Centers to boundary...")
        retail = retail.to_crs(PROJECTED_CRS)
        retail = gpd.clip(retail, boundary_shape)
        
        if len(retail) == 0:
            print("Warning: No retail centers found within the boundary after clipping.")
            return

        # Clip Postcodes to Boundary (Good practice if we clip retail)
        print("Clipping Postcodes to boundary...")
        postcodes = postcodes.to_crs(PROJECTED_CRS)
        postcodes = gpd.clip(postcodes, boundary_shape)
        
        if len(postcodes) == 0:
            print("Warning: No postcodes found within the boundary after clipping.")
            return

        # 5. Connect Retail Centres (Using Centroids)
        # connect_points_to_graph now handles the geometry to point conversion internally
        G = connect_points_to_graph(G, retail, id_col='RC_ID', label_prefix='RC', target_crs=PROJECTED_CRS)

        # 6. Connect Postcodes (Using Centroids)
        G = connect_points_to_graph(G, postcodes, id_col='postcode', label_prefix='Postcode', target_crs=PROJECTED_CRS)
        
        # Save to Cache
        print(f"Saving constructed network to {CACHE_FILE}...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump((G, retail, postcodes), f)

    # 6. Prepare for Routing (Convert to igraph)
    # igraph is much faster for shortest path calculations on large graphs
    ig_graph, mapping = from_nx_to_igraph(G, weight_attr='length')
    
    # 7. Calculate Pathways
    print("Calculating shortest paths...")
    
    # Identify Source and Target Node Indices in igraph
    # Sources: Postcodes
    # Targets: Retail Centres (RC)
    
    # We need to find the node IDs in G that correspond to Postcodes and RCs
    # We tagged them with 'type' in connect_points_to_graph for easier ID, 
    # or we can check if the ID exists in the dataframes.
    
    pcd_ids = [str(row['postcode']).strip() for idx, row in postcodes.iterrows() if str(row['postcode']).strip() in mapping]
    rc_ids = [str(row['RC_ID']).strip() for idx, row in retail.iterrows() if str(row['RC_ID']).strip() in mapping]
    
    if not pcd_ids:
        print("Error: No postcode nodes found in graph.")
        return
    if not rc_ids:
        print("Error: No retail centre nodes found in graph.")
        return

    source_indices = [mapping[pid] for pid in pcd_ids]
    target_indices = [mapping[rid] for rid in rc_ids]
    
    print(f"Sources (Postcodes): {len(source_indices)}")
    print(f"Targets (Retail Centres): {len(target_indices)}")
    
    print(f"Sources (Postcodes): {len(source_indices)}")
    print(f"Targets (Retail Centres): {len(target_indices)}")
    
    # Chunk processing for progress tracking
    chunk_size = 1000
    total_sources = len(source_indices)
    results = []
    
    print(f"Starting shortest path calculations (Batch size: {chunk_size})...")
    start_time = time.time()
    
    for i in range(0, total_sources, chunk_size):
        chunk_source_indices = source_indices[i : i + chunk_size]
        chunk_pcd_ids = pcd_ids[i : i + chunk_size]
        
        # Calculate distances for this chunk
        dists_matrix = ig_graph.distances(source=chunk_source_indices, target=target_indices, weights='weight')
        
        # Process results
        for j, row_dists in enumerate(dists_matrix):
            # Filter and sort (keep top 20)
            valid_pairs = []
            for k, dist in enumerate(row_dists):
                if np.isfinite(dist):
                    valid_pairs.append((rc_ids[k], dist))
            
            valid_pairs.sort(key=lambda x: x[1])
            # Ensure keys are strings for Parquet compatibility
            nearest_dict = {str(rc): d for rc, d in valid_pairs}
            results.append({
                'source': chunk_pcd_ids[j],
                'nearest_targets': nearest_dict
            })
            
        # Progress Update
        processed = min(i + chunk_size, total_sources)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining_items = total_sources - processed
        eta = remaining_items / rate if rate > 0 else 0
        
        print(f"Processed {processed}/{total_sources} sources. Elapsed: {elapsed:.1f}s. ETA: {eta:.1f}s")
        
    # 8. Save Results
    print("Creating DataFrame...")
    results_df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    results_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    run_analysis()
