"""
Retail ABM - Visualization Module

This script generates maps and charts from the simulation results.
It specifically creates a map of Retail Centres colored by the number of visits they received.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import warnings

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

def clean_id(x):
    """
    Standardizes ID to string (removes .0 if float).
    """
    try:
        if pd.isna(x):
            return "Unknown"
        s = str(x)
        if s.endswith('.0'):
            return s[:-2]
        return s
    except:
        return str(x)

def plot_visitation_map(visits_file, transport_mode=None):
    """
    Generates a map of retail centres colored by visit count.

    Parameters
    ----------
    transport_mode : str or None
        If 'walk', 'drive', or 'pt', only trips by that mode are counted.
        If None (default), all modes are aggregated together.
    """
    mode_label = f" ({transport_mode})" if transport_mode else ""
    print(f"Generating map for {visits_file}... (mode={transport_mode or 'all'})")

    # 1. Load Visits
    try:
        visits_df = pd.read_parquet(visits_file)
    except Exception as e:
        print(f"Error reading visits file: {e}")
        return

    if visits_df.empty:
        print("Visits file is empty. No map generated.")
        return

    # Filter by transport mode if requested
    if transport_mode and 'Transport_Mode' in visits_df.columns:
        visits_df = visits_df[visits_df['Transport_Mode'] == transport_mode]
        if visits_df.empty:
            print(f"No visits found for transport mode '{transport_mode}'.")
            return
            
    # --- MAPPING MERGED RESULTS BACK TO INDIVIDUAL CENTRES ---
    mapping_path = os.path.join(config.UTILITY_DIR, 'centre_merging_map.parquet')
    if os.path.exists(mapping_path):
        print("  Applying centre merging map...")
        mapping = pd.read_parquet(mapping_path)
        # Ensure string IDs
        mapping['RC_ID'] = mapping['RC_ID'].astype(str)
        mapping['Leader_RC_ID'] = mapping['Leader_RC_ID'].astype(str)
        
        # Aggregate Visits by the Leader ID (which is what main.py logs)
        stats = visits_df.groupby('Retail_Centre').size().reset_index(name='Visit_Count')
        stats['Retail_Centre'] = stats['Retail_Centre'].astype(str)
        
        # Map back to all original IDs
        # Join: Original RC_ID gets the count of its Leader
        stats = mapping.merge(stats, left_on='Leader_RC_ID', right_on='Retail_Centre', how='left')
        stats['Visit_Count'] = stats['Visit_Count'].fillna(0)
        stats['clean_id'] = stats['RC_ID']
    else:
        # Fallback to standard aggregation if no map exists
        stats = visits_df.groupby('Retail_Centre').size().reset_index(name='Visit_Count')
        stats['clean_id'] = stats['Retail_Centre'].apply(clean_id)
        
    print(f"Aggregated visits for {len(stats)} original centre polygons.")

    # 2. Load Geometry (Retail Centres)
    if not os.path.exists(config.RETAIL_CENTRES_GPKG):
        print(f"Error: Retail Centre GPKG not found at {config.RETAIL_CENTRES_GPKG}")
        return

    try:
        # Layer 'retail_centre_counts' has the geometry and 'RC_ID'
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
    except Exception as e:
        print(f"Error reading GPKG: {e}")
        return
        
    # Clean ID for join
    # RC_ID might be float/int/str
    if 'RC_ID' not in gdf.columns:
        print("Error: 'RC_ID' column not found in GPKG.")
        print("Available columns:", gdf.columns)
        return

    gdf['clean_id'] = gdf['RC_ID'].apply(clean_id)
    
    # 3. Join
    # Left join on GeoDataFrame to keep geometry
    # We want ALL retail centres or just visited ones? 
    # Usually better to show all (grey for 0 visits) or just visited.
    # Let's show all, but color 0 as distinct or low.
    
    merged = gdf.merge(stats, on='clean_id', how='left')
    merged['Visit_Count'] = merged['Visit_Count'].fillna(0)
    
    print(f"Merged data. Max visits to a centre: {merged['Visit_Count'].max()}")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot base (all centres) in light grey
    merged.plot(ax=ax, color='lightgrey', edgecolor='grey', linewidth=0.1)
    
    # Plot visited centres with color scale
    visited = merged[merged['Visit_Count'] > 0]
    if not visited.empty:
        visited.plot(
            column='Visit_Count',
            ax=ax,
            legend=True,
            legend_kwds={'label': "Number of Visits", 'orientation': "vertical"},
            cmap='viridis',
            edgecolor='none',
            # markersize for points? These are polygons.
        )
    
    ax.set_title(
        f"Retail Centre Visitation Density{mode_label}\n"
        f"(Total Visits: {len(visits_df)})", fontsize=15)
    ax.set_axis_off()
    
    # Save
    timestamp = int(time.time())
    output_filename = f"visitation_map_{timestamp}.png"
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Map saved to {output_path}")
    return output_path


def create_pydeck_map(visits_file, as_2d=False, transport_mode=None):
    """
    Generates an interactive PyDeck map of retail centres colored by visit count.
    If as_2d is True, the map is flat. Otherwise, it is extruded in 3D.

    Parameters
    ----------
    transport_mode : str or None
        If 'walk', 'drive', or 'pt', only trips by that mode are counted.
        If None (default), all modes are aggregated together.
    """
    print(f"Generating interactive map for {visits_file}... (mode={transport_mode or 'all'})")

    try:
        visits_df = pd.read_parquet(visits_file)
    except Exception as e:
        print(f"Error reading visits file: {e}")
        return None

    if visits_df.empty:
        print("Visits file is empty. No map generated.")
        return None

    # Filter by transport mode if requested
    if transport_mode and 'Transport_Mode' in visits_df.columns:
        visits_df = visits_df[visits_df['Transport_Mode'] == transport_mode]
        if visits_df.empty:
            print(f"No visits found for transport mode '{transport_mode}'.")
            return None

    # --- MAPPING MERGED RESULTS BACK TO INDIVIDUAL CENTRES ---
    mapping_path = os.path.join(config.UTILITY_DIR, 'centre_merging_map.parquet')
    if os.path.exists(mapping_path):
        print("  Applying centre merging map...")
        mapping = pd.read_parquet(mapping_path)
        mapping['RC_ID'] = mapping['RC_ID'].astype(str)
        mapping['Leader_RC_ID'] = mapping['Leader_RC_ID'].astype(str)
        
        stats = visits_df.groupby('Retail_Centre').size().reset_index(name='Visit_Count')
        stats['Retail_Centre'] = stats['Retail_Centre'].astype(str)
        
        stats = mapping.merge(stats, left_on='Leader_RC_ID', right_on='Retail_Centre', how='left')
        stats['Visit_Count'] = stats['Visit_Count'].fillna(0)
        stats['clean_id'] = stats['RC_ID']
    else:
        stats = visits_df.groupby('Retail_Centre').size().reset_index(name='Visit_Count')
        stats['clean_id'] = stats['Retail_Centre'].apply(clean_id)
    
    if not os.path.exists(config.RETAIL_CENTRES_GPKG):
        print(f"Error: Retail Centre GPKG not found at {config.RETAIL_CENTRES_GPKG}")
        return None
        
    try:
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
    except Exception as e:
        print(f"Error reading GPKG: {e}")
        return None
        
    if 'RC_ID' not in gdf.columns:
        print("Error: 'RC_ID' column not found in GPKG.")
        return None
        
    gdf['clean_id'] = gdf['RC_ID'].apply(clean_id)
    merged = gdf.merge(stats, on='clean_id', how='left')
    merged['Visit_Count'] = merged['Visit_Count'].fillna(0)
    
    # Needs to be WGS84 for PyDeck / Mapbox
    merged = merged.to_crs(epsg=4326)
    
    # Filter empty geometries
    merged = merged[~merged.geometry.is_empty]
    
    # Calculate color based on visits
    max_visits = merged['Visit_Count'].max()
    if max_visits < 1: max_visits = 1
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    
    def get_color(val):
        if val == 0:
            return [200, 200, 200, 100] # Light gray for zero visits
        rgba = cmap(val / max_visits)
        return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 200]
        
    merged['fill_color'] = merged['Visit_Count'].apply(get_color)
    # Give a small base elevation just to render correctly, scale visits for height
    merged['elevation'] = merged['Visit_Count'] * 5 + 10
    
    import pydeck as pdk
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        merged,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=not as_2d,
        wireframe=True,
        get_elevation="elevation" if not as_2d else None,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        pickable=True
    )
    
    bounds = merged.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=0 if as_2d else 45,
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light", # Simple, reliable Mapbox/Carto light style
        tooltip={"text": "Retail Centre: {clean_id}\nVisits: {Visit_Count}"}
    )
    
    return deck


import streamlit as st

@st.cache_data
def load_postcode_coords():
    try:
        pc_path = r'c:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Postcode\postcode_CA_new.shp'
        gdf = gpd.read_file(pc_path)
        return gdf[['postcode', 'latitude', 'longitude']].rename(columns={'postcode': 'Postcode'})
    except Exception as e:
        print(f"Error loading postcode coords: {e}")
        return pd.DataFrame()

@st.cache_data
def load_centre_coords_v2():
    try:
        gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
        gdf['clean_id'] = gdf['RC_ID'].apply(clean_id)
        
        # Suppress CRS warning: project to 3857 (meters) for centroid then back to 4326 (degrees)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # For small areas, projected centroid is more accurate than geographic centroid
            tmp_gdf = gdf.to_crs(epsg=3857)
            tmp_gdf['center_lon'] = tmp_gdf.geometry.centroid.to_crs(epsg=4326).x
            tmp_gdf['center_lat'] = tmp_gdf.geometry.centroid.to_crs(epsg=4326).y
        
        return tmp_gdf[['clean_id', 'center_lon', 'center_lat']]
    except Exception as e:
        print(f"Error loading centre coords: {e}")
        return pd.DataFrame()

@st.cache_data
def load_agent_postcodes():
    try:
        df = pd.read_parquet(config.UTILITY_SCORES_AVG, columns=['household', 'Postcode'])
        return df.drop_duplicates(subset=['household'])
    except Exception as e:
        print(f"Error loading agent postcodes: {e}")
        return pd.DataFrame()

def plot_market_share_static_map(visits_df):
    """
    Calculates the dominant retail centre for each LSOA and returns a matplotlib Figure.
    """
    if visits_df.empty: return None
    
    agents_df = load_agent_postcodes()
    if agents_df.empty: return None
    
    import geopandas as gpd
    import pandas as pd
    
    # Drop Postcode from agents_df if already in visits_df to avoid suffixing (Postcode_x/_y)
    if 'Postcode' in visits_df.columns:
        agents_df = agents_df.drop(columns=['Postcode'])
        
    merged = visits_df.merge(agents_df, left_on='AgentID', right_on='household', how='left')
    counts = merged.groupby(['Postcode', 'Retail_Centre']).size().reset_index(name='cnt')
    
    try:
        # Load Postcode geometries and fetch centroids to do a clean point-in-polygon join
        pc_path = r'c:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Postcode\postcode_CA_new.shp'
        pc_gdf = gpd.read_file(pc_path)
        pc_pts = pc_gdf[['postcode', 'geometry']].rename(columns={'postcode': 'Postcode'}).copy()
        
        # Use centroids to place the postcode definitively inside one LSOA
        # Must suppress warning regarding geographic crs centroid calculation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc_pts['geometry'] = pc_pts.geometry.centroid
            
        # Load CA boundaries (LSOAs)
        ca_path = r'C:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Boundary\CA boundary.shp'
        lsoa_gdf = gpd.read_file(ca_path)
        
        # Ensure identical projections before spatial joining
        if pc_pts.crs != lsoa_gdf.crs:
            pc_pts = pc_pts.to_crs(lsoa_gdf.crs)
            
        # We index the LSOAs dynamically to avoid depending on a specific column name (e.g. LSOA11CD vs lsoa11cd)
        lsoa_gdf['LSOA_ID'] = lsoa_gdf.index
        
        # Determine which LSOA block each Postcode's centroid falls inside
        pc_to_lsoa = gpd.sjoin(pc_pts, lsoa_gdf, how='inner', predicate='intersects')
        mapping = pc_to_lsoa[['Postcode', 'LSOA_ID']].drop_duplicates()
        
        # Push the postcode trip counts up into their parent LSOA
        counts_lsoa = counts.merge(mapping, on='Postcode', how='inner')
        counts_lsoa = counts_lsoa.groupby(['LSOA_ID', 'Retail_Centre'])['cnt'].sum().reset_index()
        
        # Identify the most prominent retail centre serving that LSOA territory
        idx = counts_lsoa.groupby('LSOA_ID')['cnt'].idxmax()
        dominant = counts_lsoa.loc[idx]
        
        # Re-attach the dominating retail centre data onto the actual LSOA polygons
        merged_gdf = lsoa_gdf.merge(dominant, on='LSOA_ID', how='inner')
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot the LSOA patchwork
        merged_gdf.plot(column='Retail_Centre', 
                        categorical=True, 
                        cmap='tab20', 
                        legend=True,
                        legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1, 0.5), 'title': 'Dominant Retail Centre', 'fontsize': 9, 'title_fontsize': 11, 'ncol': 2},
                        linewidth=0.2, 
                        edgecolor='white',
                        ax=ax)
                        
        # Optionally overlay the dissolved regional border for high contrast
        bnd_path = r'C:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Boundary\CA boundary_disolved.shp'
        try:
            bnd_gdf = gpd.read_file(bnd_path)
            if bnd_gdf.crs != merged_gdf.crs:
                bnd_gdf = bnd_gdf.to_crs(merged_gdf.crs)
            bnd_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        except Exception as e:
            print(f"Warning: Dissolved boundary fail: {e}")
            pass
        
        total_visits = visits_df['Visits'].sum() if 'Visits' in visits_df.columns else len(visits_df)
        ax.set_title(f"Regional Market Dominance by LSOA (N = {total_visits:,} Trips)", fontsize=16, pad=15)
        ax.set_axis_off()
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating LSOA static market share map: {e}")
        return None

def generate_hover_flow_map_html(visits_df):
    """
    Creates a standalone deck.gl HTML visualization.
    Displays all retail centres and reveals incoming flows when hovering over a centre.
    """
    if visits_df.empty:
        return ""
        
    visits_df = visits_df.copy()
    visits_df['clean_id'] = visits_df['Retail_Centre'].astype(str).apply(clean_id)
    
    agents_df = load_agent_postcodes()
    if agents_df.empty: return ""
    
    # Drop Postcode from agents_df if already in visits_df to avoid suffixing (Postcode_x/_y)
    if 'Postcode' in visits_df.columns:
        agents_df = agents_df.drop(columns=['Postcode'])

    merged = visits_df.merge(agents_df, left_on='AgentID', right_on='household', how='left')
    flows = merged.groupby(['clean_id', 'Postcode']).size().reset_index(name='Visits')

    pc_coords = load_postcode_coords()
    if pc_coords.empty: return ""
    
    flows = flows.merge(pc_coords, on='Postcode', how='inner')
    flows.rename(columns={'latitude': 'source_lat', 'longitude': 'source_lon', 'clean_id': 'target_id'}, inplace=True)

    c_coords = load_centre_coords_v2()
    if c_coords.empty: return ""
    
    flows = flows.merge(c_coords, left_on='target_id', right_on='clean_id', how='inner')
    flows.rename(columns={'center_lat': 'dest_lat', 'center_lon': 'dest_lon'}, inplace=True)

    flows = flows.dropna(subset=['source_lat', 'source_lon', 'dest_lat', 'dest_lon'])
    
    import numpy as np
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(flows['source_lat']), np.radians(flows['dest_lat'])
    delta_phi, delta_lambda = np.radians(flows['dest_lat'] - flows['source_lat']), np.radians(flows['dest_lon'] - flows['source_lon'])
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    flows['dist_m'] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    flows['tot_dist'] = flows['dist_m'] * flows['Visits']
    
    # Avoid FutureWarning by explicitly selecting columns before apply
    avg_dists = flows.groupby('target_id')[['tot_dist', 'Visits']].apply(
        lambda x: x['tot_dist'].sum() / x['Visits'].sum() if x['Visits'].sum() > 0 else 0
    ).reset_index(name='avg_radius')
    
    c_coords = c_coords.merge(avg_dists, left_on='clean_id', right_on='target_id', how='left')
    c_coords['avg_radius'] = c_coords['avg_radius'].fillna(0)
    
    max_visits = flows['Visits'].max() if not flows.empty else 1
    center_lon = c_coords['center_lon'].mean() if not c_coords.empty else -2.9
    center_lat = c_coords['center_lat'].mean() if not c_coords.empty else 53.4

    centers_json = c_coords.to_dict(orient='records')
    
    # Cap the maximum flows to the top 75,000 largest trips globally to prevent map crashing
    # Round coordinates tightly to 4 decimals, discard postcodes to drastically cut JSON size
    if len(flows) > 75000:
        flows = flows.sort_values('Visits', ascending=False).head(75000)
    
    flows['source_lon'] = flows['source_lon'].round(4)
    flows['source_lat'] = flows['source_lat'].round(4)
    flows['dest_lon'] = flows['dest_lon'].round(4)
    flows['dest_lat'] = flows['dest_lat'].round(4)
    if 'dist_m' in flows.columns:
        flows['dist_m'] = flows['dist_m'].round(0)
        
    # Slim down flows to reduce HTML payload size
    flows_slim = flows[['target_id', 'source_lon', 'source_lat', 'dest_lon', 'dest_lat', 'Visits', 'dist_m']].to_dict(orient='records')

    import json
    
    try:
        bnd_path = r'C:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Boundary\CA boundary_disolved.shp'
        bnd_gdf = gpd.read_file(bnd_path).to_crs(epsg=4326)
        bnd_geojson = json.dumps(bnd_gdf.__geo_interface__)
    except Exception as e:
        print(f"Error loading boundary: {e}")
        bnd_geojson = "{}"
        
    html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
        <script src="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.js"></script>
        <link href="https://unpkg.com/maplibre-gl@3.0.0/dist/maplibre-gl.css" rel="stylesheet" />
        <style>body {{ margin: 0; padding: 0; }} #map {{ width: 100vw; height: 100vh; border-radius: 8px; overflow: hidden; }} #tooltip {{ position: absolute; z-index: 10; pointer-events: none; display: none; background: rgba(255,255,255,0.9); color: black; padding: 10px; font-family: sans-serif; font-size: 13px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}</style>
      </head>
      <body>
        <div id="map"></div>
        <div id="tooltip"></div>
        <script>
          const boundary = {bnd_geojson};
          const centers = {json.dumps(centers_json)};
          const flows = {json.dumps(flows_slim)};
          const maxVisits = {max_visits};
          
          let hoveredCenterId = null;
          
          function renderLayers() {{
              const activeFlows = flows.filter(f => f.target_id === hoveredCenterId);
              
              const centerLayer = new deck.ScatterplotLayer({{
                  id: 'centers',
                  data: centers,
                  getPosition: d => [d.center_lon, d.center_lat],
                  getFillColor: d => d.clean_id === hoveredCenterId ? [239, 68, 68, 255] : [255, 255, 255, 200],
                  getRadius: d => d.clean_id === hoveredCenterId ? 300 : 150,
                  pickable: true,
                  onHover: info => {{
                      if (info.object) {{
                          hoveredCenterId = info.object.clean_id;
                          document.getElementById('tooltip').style.display = 'block';
                          document.getElementById('tooltip').style.left = (info.x + 10) + 'px';
                          document.getElementById('tooltip').style.top = (info.y + 10) + 'px';
                          document.getElementById('tooltip').innerHTML = '<b>Retail Centre ' + info.object.clean_id + '</b><hr style="margin:4px 0;border-color:#555;"/>Avg Commute: ' + (info.object.avg_radius/1000).toFixed(2) + ' km<br/><span style="color:#d1d5db;">Hovering reveals inbound trips</span>';
                      }} else {{
                          hoveredCenterId = null;
                          document.getElementById('tooltip').style.display = 'none';
                      }}
                      renderLayers();
                  }}
              }});
              
              const activeCenter = centers.filter(d => d.clean_id === hoveredCenterId);
              const bufferLayer = new deck.ScatterplotLayer({{
                  id: 'buffers',
                  data: activeCenter,
                  getPosition: d => [d.center_lon, d.center_lat],
                  getFillColor: [219, 39, 119, 30],
                  getLineColor: [219, 39, 119, 150],
                  lineWidthMinPixels: 2,
                  stroked: true,
                  getRadius: d => d.avg_radius || 0,
                  pickable: false,
              }});
              
              const arcLayer = new deck.ArcLayer({{
                  id: 'arcs',
                  data: activeFlows,
                  getSourcePosition: d => [d.source_lon, d.source_lat],
                  getTargetPosition: d => [d.dest_lon, d.dest_lat],
                  getSourceColor: d => {{
                      const ratio = Math.min((d.dist_m || 0) / 20000.0, 1.0);
                      return [230 + (25 * ratio), 80 + (100 * ratio), 0 + (50 * ratio), 200];
                  }},
                  getTargetColor: d => {{
                      const ratio = Math.min((d.dist_m || 0) / 20000.0, 1.0);
                      return [230 + (25 * ratio), 80 + (100 * ratio), 0 + (50 * ratio), 200];
                  }},
                  getWidth: d => 1 + (d.Visits / maxVisits) * 15,
                  getTilt: 15
              }});
              
              const boundaryLayer = new deck.GeoJsonLayer({{
                  id: 'boundary',
                  data: boundary,
                  stroked: true,
                  filled: false,
                  lineWidthMinPixels: 2,
                  getLineColor: [200, 200, 200, 150],
              }});
              
              deckgl.setProps({{ layers: [boundaryLayer, bufferLayer, centerLayer, arcLayer] }});
          }}
          
          const deckgl = new deck.DeckGL({{
              container: 'map',
              mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
              initialViewState: {{
                  longitude: {center_lon},
                  latitude: {center_lat},
                  zoom: 9.5,
                  pitch: 45
              }},
              controller: true,
              layers: []
          }});
          
          renderLayers();
        </script>
      </body>
    </html>
    """
    return html



def create_catchment_map(visits_df, target_centre_id, top_percent=0.8):
    """
    Creates an interactive catchment map highlighting the region where the top X% of visits originated from.
    """
    if visits_df.empty:
        return None
        
    visits_df = visits_df.copy()
    visits_df['clean_id'] = visits_df['Retail_Centre'].astype(str).apply(clean_id)
    visits_centre = visits_df[visits_df['clean_id'] == str(target_centre_id)]
    
    if visits_centre.empty:
        return None

    agents_df = load_agent_postcodes()
    if agents_df.empty: return None
    
    # Drop Postcode from agents_df if already in visits_df to avoid suffixing (Postcode_x/_y)
    if 'Postcode' in visits_df.columns:
        agents_df = agents_df.drop(columns=['Postcode'])

    merged = visits_centre.merge(agents_df, left_on='AgentID', right_on='household', how='left')
    flows = merged.groupby('Postcode').size().reset_index(name='Visits')
    
    # Sort and compute cumulative percentage
    flows = flows.sort_values(by='Visits', ascending=False)
    total_visits = flows['Visits'].sum()
    flows['cumulative_visits'] = flows['Visits'].cumsum()
    flows['cumulative_pct'] = flows['cumulative_visits'] / total_visits
    
    flows['prev_pct'] = flows['cumulative_pct'].shift(fill_value=0)
    top_flows = flows[flows['prev_pct'] < top_percent]

    try:
        pc_path = r'c:\Users\sgabalog\Documents\P3\Model\Distance\Data\Liverpool Postcode\postcode_CA_new.shp'
        gdf = gpd.read_file(pc_path)
    except Exception as e:
        print(f"Error loading postcode shapefile for catchment: {e}")
        return None
        
    gdf = gdf[['postcode', 'geometry']].rename(columns={'postcode': 'Postcode'})
    top_gdf = gdf.merge(top_flows, on='Postcode', how='inner')
    if top_gdf.empty: return None
    
    # Project to UK metric, buffer by 1500m to form a continuous blob, union, project back
    top_gdf = top_gdf.to_crs(epsg=27700)
    top_gdf['geometry'] = top_gdf['geometry'].buffer(1500)
    catchment_geom = top_gdf.unary_union
    
    catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_geom], crs="EPSG:27700").to_crs(epsg=4326)
    
    c_coords = load_centre_coords()
    dest = c_coords[c_coords['clean_id'] == str(target_centre_id)]
    if dest.empty: return None
    
    dest_lon = dest.iloc[0]['center_lon']
    dest_lat = dest.iloc[0]['center_lat']

    import pydeck as pdk
    
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=catchment_gdf.__geo_interface__,
        opacity=0.4,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color=[0, 204, 255, 100],
        get_line_color=[0, 100, 255, 255],
        get_line_width=30,
        pickable=False
    )
    
    dest_layer = pdk.Layer(
        "ScatterplotLayer",
        data=dest,
        get_position=["center_lon", "center_lat"],
        get_fill_color=[255, 0, 128, 255],
        get_radius=500,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=dest_lat,
        longitude=dest_lon,
        zoom=10,
        pitch=0, 
    )

    deck = pdk.Deck(
        layers=[geojson_layer, dest_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": f"Catchment Area (Top {int(top_percent * 100)}%)"}
    )
    return deck

if __name__ == "__main__":
    pass
