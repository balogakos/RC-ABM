
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
PT_STOPS_PATH = os.path.join(INPUT_DIR, 'PT Stops', 'naptan_CA.shp')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'pt_accessibility_results.parquet')
CACHE_FILE = os.path.join(CACHE_DIR, 'pt_network_cache.pkl')
TEST_MODE = False  # Set to True to only process the first 100 postcodes

PROJECTED_CRS = 'EPSG:27700'
NETWORK_CRS = 'EPSG:4326'

def connect_points_to_graph(G, points_gdf, id_col, label_prefix, target_crs):
    print(f"Connecting {label_prefix} to network...")
    node_ids = list(G.nodes())
    node_coords = np.array([(data['x'], data['y']) for node, data in G.nodes(data=True)])
    tree = cKDTree(node_coords)
    points_gdf_proj = points_gdf.to_crs(target_crs)
    point_coords = np.array([(p.centroid.x, p.centroid.y) for p in points_gdf_proj.geometry])
    distances, indices = tree.query(point_coords, k=1)
    
    count = 0
    mapping = {}
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        if id_col in points_gdf.columns:
            p_id = str(points_gdf.iloc[i][id_col]).strip()
        else:
            p_id = f"{label_prefix}_{i}"
            
        nearest_node_id = node_ids[idx]
        pt = points_gdf_proj.geometry.iloc[i]
        cx, cy = pt.centroid.x, pt.centroid.y
        G.add_node(p_id, x=cx, y=cy, type=label_prefix)
        
        if nearest_node_id != p_id:
            G.add_edge(p_id, nearest_node_id, length=0)
            G.add_edge(nearest_node_id, p_id, length=0)
            count += 1
        mapping[p_id] = p_id
            
    print(f"Connected {count} {label_prefix} points to the network.")
    return G, mapping

def from_nx_to_igraph(Mg, weight_attr='length'):
    print("Converting NetworkX graph to igraph...")
    mapping = {node: i for i, node in enumerate(Mg.nodes())}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(mapping))
    edges = []
    weights = []
    for u, v, data in Mg.edges(data=True):
        if weight_attr in data:
            edges.append((mapping[u], mapping[v]))
            weights.append(data[weight_attr])
        else:
            edges.append((mapping[u], mapping[v]))
            weights.append(0)
    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights
    return ig_graph, mapping

def run_analysis():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached networks from {CACHE_FILE}...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                G_walk = cache_data['G_walk']
                G_drive = cache_data['G_drive']
                retail = cache_data['retail']
                postcodes = cache_data['postcodes']
                pt_stops = cache_data['pt_stops']
                walk_mapping = cache_data['walk_mapping']
                drive_mapping = cache_data['drive_mapping']
            print("Loaded dual-networks from cache successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding networks...")
            G_walk = None
    else:
        G_walk = None

    if G_walk is None:
        print("Loading shapefiles...")
        lsoa = gpd.read_file(LSOA_PATH)
        postcodes = gpd.read_file(POSTCODE_PATH)
        retail = gpd.read_file(RETAIL_PATH)
        pt_stops = gpd.read_file(PT_STOPS_PATH)
        
        boundary_polygon = lsoa.geometry.union_all()
        if lsoa.crs != "EPSG:4326":
            boundary_s = gpd.GeoSeries([boundary_polygon], crs=lsoa.crs).to_crs("EPSG:4326")
            boundary_polygon = boundary_s.iloc[0]
            
        print("Building WALKING network...")
        G_walk = ox.graph_from_polygon(boundary_polygon, network_type="walk")
        G_walk = ox.project_graph(G_walk, to_crs=PROJECTED_CRS)

        print("Building DRIVING network...")
        G_drive = ox.graph_from_polygon(boundary_polygon, network_type="drive")
        G_drive = ox.project_graph(G_drive, to_crs=PROJECTED_CRS)
        
        for u, v, data in G_walk.edges(data=True):
            if 'length' not in data or pd.isna(data.get('length')): data['length'] = 0
        for u, v, data in G_drive.edges(data=True):
            if 'length' not in data or pd.isna(data.get('length')): data['length'] = 0

        lsoa_proj = lsoa.to_crs(PROJECTED_CRS)
        boundary_shape = lsoa_proj.geometry.union_all()
        
        retail = retail.to_crs(PROJECTED_CRS)
        retail = gpd.clip(retail, boundary_shape)
        postcodes = postcodes.to_crs(PROJECTED_CRS)
        postcodes = gpd.clip(postcodes, boundary_shape)
        pt_stops = pt_stops.to_crs(PROJECTED_CRS)
        pt_stops = gpd.clip(pt_stops, boundary_shape)

        # Snap Entities to WALK network for pedestrian access/egress
        G_walk, rc_walk_mapping = connect_points_to_graph(G_walk, retail, id_col='RC_ID', label_prefix='RC', target_crs=PROJECTED_CRS)
        G_walk, pcd_walk_mapping = connect_points_to_graph(G_walk, postcodes, id_col='postcode', label_prefix='Postcode', target_crs=PROJECTED_CRS)
        G_walk, pt_walk_mapping = connect_points_to_graph(G_walk, pt_stops, id_col='ATCOCode', label_prefix='PT', target_crs=PROJECTED_CRS)

        # Snap PT stops to DRIVE network for inter-stop routing
        G_drive, pt_drive_mapping = connect_points_to_graph(G_drive, pt_stops, id_col='ATCOCode', label_prefix='PT', target_crs=PROJECTED_CRS)

        print(f"Saving constructed dual-network to {CACHE_FILE}...")
        cache_data = {
            'G_walk': G_walk, 'G_drive': G_drive,
            'retail': retail, 'postcodes': postcodes, 'pt_stops': pt_stops,
            'walk_mapping': G_walk.nodes(), # We use node names directly
            'drive_mapping': G_drive.nodes()
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        walk_mapping = G_walk.nodes()
        drive_mapping = G_drive.nodes()

    ig_walk, walk_mapping_idx = from_nx_to_igraph(G_walk, weight_attr='length')
    ig_drive, drive_mapping_idx = from_nx_to_igraph(G_drive, weight_attr='length')
    
    pcd_ids = [str(row['postcode']).strip() for idx, row in postcodes.iterrows() if str(row['postcode']).strip() in walk_mapping_idx]
    rc_ids = [str(row['RC_ID']).strip() for idx, row in retail.iterrows() if str(row['RC_ID']).strip() in walk_mapping_idx]
    pt_ids = [str(ptid).strip() for ptid in pt_stops['ATCOCode'] if str(ptid).strip() in walk_mapping_idx and str(ptid).strip() in drive_mapping_idx]
    
    if TEST_MODE:
        pcd_ids = pcd_ids[:100]

    print("Step 1: Postcodes to nearest PT stops (WALK)")
    pcd_indices = [walk_mapping_idx[pid] for pid in pcd_ids]
    pt_walk_indices = [walk_mapping_idx[ptid] for ptid in pt_ids]
    pcd_to_pt_matrix = ig_walk.distances(source=pcd_indices, target=pt_walk_indices, weights='weight')
    
    pcd_nearest_pt = {}
    for i, pcd_id in enumerate(pcd_ids):
        dists = pcd_to_pt_matrix[i]
        valid = [(pt_ids[j], d) for j, d in enumerate(dists) if np.isfinite(d)]
        if valid:
            valid.sort(key=lambda x: x[1])
            pcd_nearest_pt[pcd_id] = (valid[0][0], valid[0][1])

    print("Step 2: Retail Centres to nearest PT stops (WALK)")
    rc_indices = [walk_mapping_idx[rid] for rid in rc_ids]
    rc_to_pt_matrix = ig_walk.distances(source=rc_indices, target=pt_walk_indices, weights='weight')
    
    rc_nearest_pt = {}
    for i, rc_id in enumerate(rc_ids):
        dists = rc_to_pt_matrix[i]
        valid = [(pt_ids[j], d) for j, d in enumerate(dists) if np.isfinite(d)]
        if valid:
            valid.sort(key=lambda x: x[1])
            rc_nearest_pt[rc_id] = (valid[0][0], valid[0][1])

    print("Step 3: Point-to-Point through PT network (DRIVE)")
    unique_pcd_pts = list(set([v[0] for v in pcd_nearest_pt.values()]))
    unique_rc_pts = list(set([v[0] for v in rc_nearest_pt.values()]))
    
    pcd_pt_drive_indices = [drive_mapping_idx[ptid] for ptid in unique_pcd_pts]
    rc_pt_drive_indices = [drive_mapping_idx[ptid] for ptid in unique_rc_pts]
    
    full_pt_matrix = ig_drive.distances(source=pcd_pt_drive_indices, target=rc_pt_drive_indices, weights='weight')
    
    pt_dist_lookup = {}
    for i, pt_start in enumerate(unique_pcd_pts):
        for j, pt_end in enumerate(unique_rc_pts):
            pt_dist_lookup[(pt_start, pt_end)] = full_pt_matrix[i][j]

    print("Step 4: Combining results...")
    results = []
    for pcd_id in pcd_ids:
        if pcd_id not in pcd_nearest_pt: continue
        start_pt, d1 = pcd_nearest_pt[pcd_id]
        for rc_id in rc_ids:
            if rc_id not in rc_nearest_pt: continue
            end_pt, d2 = rc_nearest_pt[rc_id]
            d_net = pt_dist_lookup.get((start_pt, end_pt), float('inf'))
            if np.isfinite(d_net):
                results.append({
                    'postcode': pcd_id,
                    'retail_centre': rc_id,
                    'postcode_to_pt_distance': d1,
                    'retail_to_pt_distance': d2,
                    'pt_network_distance': d_net
                })

    print("Saving results...")
    pd.DataFrame(results).to_parquet(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    run_analysis()
