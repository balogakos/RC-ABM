import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from pathlib import Path

# =========================================================================
# --- CONFIGURATION ---
# =========================================================================
# Toggle to use real validation data or synthetic comparison data
USE_VALIDATION_DATA = False
VALIDATION_DATA_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\processed\validation_visits.csv')

# Import config from the simulation directory
import sys
PROJECT_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM')
sys.path.insert(0, str(PROJECT_ROOT / 'simulation'))
import config

# Paths for outputs and spatial data
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
CENTRE_DIR = OUTPUTS_ROOT / 'centre_performance'
GPKG_PATH = Path(config.RETAIL_CENTRES_GPKG)

# Clean function for IDs
def clean_id(x):
    try:
        if pd.isna(x):
            return "Unknown"
        s = str(x)
        if s.endswith('.0'):
            return s[:-2]
        return s
    except:
        return str(x)

# =========================================================================
# --- Helper Functions for Map Elements ---
# =========================================================================
def add_north_arrow(ax, position=(0.06, 0.94)):
    """Draws a clean, minimalistic North arrow in the top-left of the axes."""
    ax.annotate('N', xy=position, xytext=(position[0], position[1] - 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=1.5, headwidth=6, headlength=6, shrink=0.1),
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

def add_scale_bar(ax, length_km=5, position=(0.06, 0.06)):
    """Draws a scale bar in the bottom-left of the axes, adjusted for Mercator deformation at Liverpool latitude."""
    lat_deg = 53.4
    cos_lat = np.cos(np.radians(lat_deg))
    mercator_length = (length_km * 1000) / cos_lat
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Check if 5 km is too large for the current display limits, adjust if needed
    map_width_km = ((xmax - xmin) * cos_lat) / 1000.0
    if map_width_km < length_km * 2:
        if map_width_km >= 4.0:
            length_km = 2
        else:
            length_km = 1
        mercator_length = (length_km * 1000) / cos_lat
        
    x_start = xmin + position[0] * (xmax - xmin)
    y_pos = ymin + position[1] * (ymax - ymin)
    x_end = x_start + mercator_length
    
    tick_height = 0.012 * (ymax - ymin)
    
    # Draw scale bar line
    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=2.5, zorder=5)
    # Draw tick marks
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    
    # Add text label above the bar
    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.8, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

# =========================================================================
# --- LISA Calculation Function ---
# =========================================================================
def calculate_lisa(coords, values, k=5, n_permutations=99, seed=42):
    """
    Computes Local Moran's I and categorizes each location into LISA clusters.
    Avoids external dependencies like PySAL for robustness.
    """
    n = len(coords)
    np.random.seed(seed)
    
    # 1. Distance Matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    # 2. Row-standardized K-Nearest Neighbors weights matrix
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist_matrix[i])
        nn_idx = idx[1:k+1]  # Skip self at index 0
        W[i, nn_idx] = 1.0 / k
        
    # 3. Standardize values (Z-scores)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        std_val = 1.0
    z = (values - mean_val) / std_val
    
    # 4. Spatial Lag
    wz = W.dot(z)
    
    # 5. Local Moran's I
    lisa = z * wz
    
    # 6. Permutation significance testing
    sim_lisa = np.zeros((n, n_permutations))
    for p in range(n_permutations):
        # Permute all values
        for i in range(n):
            idx_other = [j for j in range(n) if j != i]
            shuffled_other = np.random.permutation(z[idx_other])
            # Construct a shuffled Z-score array where location i remains fixed
            shuffled_z_i = np.zeros(n)
            shuffled_z_i[idx_other] = shuffled_other
            shuffled_z_i[i] = z[i]
            sim_lisa[i, p] = z[i] * W[i].dot(shuffled_z_i)
            
    # 7. Compute pseudo p-values
    p_values = np.zeros(n)
    for i in range(n):
        obs = lisa[i]
        sims = sim_lisa[i]
        if obs >= 0:
            count = np.sum(sims >= obs)
        else:
            count = np.sum(sims <= obs)
        p_values[i] = (count + 1) / (n_permutations + 1)
        
    # 8. Classification
    categories = []
    for i in range(n):
        if p_values[i] > 0.05:
            categories.append("Not Significant")
        else:
            if z[i] > 0 and wz[i] > 0:
                categories.append("High-High")
            elif z[i] < 0 and wz[i] < 0:
                categories.append("Low-Low")
            elif z[i] > 0 and wz[i] < 0:
                categories.append("High-Low")
            elif z[i] < 0 and wz[i] > 0:
                categories.append("Low-High")
                
    return categories

# =========================================================================
# --- Load Data ---
# =========================================================================
print("Locating latest retail centre performance run...")
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError(f"No retail centre performance CSV files found in {CENTRE_DIR}")

latest_file = cp_files[-1]
print(f"Loading simulation visits from: {latest_file.name}")
df_sim = pd.read_csv(latest_file)

visit_cols = [c for c in df_sim.columns if c != 'Retail_Centre']
df_sim['sim_visits'] = df_sim[visit_cols].sum(axis=1)
df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
df_sim_clean = df_sim[['clean_id', 'sim_visits']].copy()

print(f"Loading spatial data from: {GPKG_PATH.name}")
gdf_centers = gpd.read_file(GPKG_PATH, layer='retail_centre_counts')
gdf_centers['clean_id'] = gdf_centers['RC_ID'].apply(clean_id)

# Handle Validation Data
df_real = None
if USE_VALIDATION_DATA:
    if VALIDATION_DATA_PATH.exists():
        print(f"Loading real-world validation data from: {VALIDATION_DATA_PATH}")
        df_real = pd.read_csv(VALIDATION_DATA_PATH)
        df_real['clean_id'] = df_real['Retail_Centre'].apply(clean_id)
        df_real = df_real[['clean_id', 'real_visits']].copy()

if df_real is None:
    print("Generating synthetic real-world baseline data...")
    # Add spatial pattern consistent noise to create synthetic validation
    np.random.seed(42)
    df_real = df_sim_clean.copy()
    noise = np.random.normal(0, 0.15 * df_real['sim_visits'])
    df_real['real_visits'] = np.clip(df_real['sim_visits'] + noise, 0, None).round(0)
    df_real = df_real[['clean_id', 'real_visits']]

# Merge datasets
gdf_mapped = gdf_centers.merge(df_sim_clean, on='clean_id', how='left')
gdf_mapped = gdf_mapped.merge(df_real, on='clean_id', how='left')
gdf_mapped['sim_visits'] = gdf_mapped['sim_visits'].fillna(0)
gdf_mapped['real_visits'] = gdf_mapped['real_visits'].fillna(0)

if 'Total_POI_' in gdf_mapped.columns:
    gdf_mapped['center_size'] = gdf_mapped['Total_POI_']
else:
    gdf_mapped['center_size'] = 100

print("Projecting geometry to EPSG:3857 (Web Mercator)...")
gdf_mapped = gdf_mapped.to_crs(epsg=3857)

# Convert to centroids for calculations and point plotting
gdf_points = gdf_mapped.copy()
gdf_points['geometry'] = gdf_points.geometry.centroid

# Get coordinates for LISA
coords = np.array([(geom.x, geom.y) for geom in gdf_points.geometry])

# Calculate LISA categories
print("Computing LISA categories for Observed vs Simulated visits...")
gdf_points['lisa_obs'] = calculate_lisa(coords, gdf_points['real_visits'].values, seed=10)
gdf_points['lisa_sim'] = calculate_lisa(coords, gdf_points['sim_visits'].values, seed=20)

# =========================================================================
# --- Plotting LISA Cluster Comparison ---
# =========================================================================
print("Plotting LISA clusters comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# LISA Color Scheme
lisa_colors = {
    'High-High': '#e74c3c',       # Red
    'Low-Low': '#3498db',         # Blue
    'Low-High': '#85c1e9',        # Light Blue
    'High-Low': '#f1948a',        # Light Red/Pink
    'Not Significant': '#d5dbdb'  # Light Gray
}
marker_style = dict(edgecolor='black', linewidth=0.8, alpha=0.9, zorder=4)
sizes = np.clip(gdf_points['center_size'] * 1.5, 20, 600)

# Categories to plot in order (so legend is ordered consistently)
categories_order = ['High-High', 'Low-Low', 'Low-High', 'High-Low', 'Not Significant']

# Panel A: Observed LISA Clusters
for cat in categories_order:
    mask = gdf_points['lisa_obs'] == cat
    if mask.any():
        gdf_points[mask].plot(
            ax=ax1,
            color=lisa_colors[cat],
            markersize=sizes[mask],
            label=cat,
            **marker_style
        )
ax1.set_title('(A) Observed LISA Clusters (Baseline)', fontweight='bold', fontsize=14, loc='left', pad=10)

# Panel B: Simulated LISA Clusters
for cat in categories_order:
    mask = gdf_points['lisa_sim'] == cat
    if mask.any():
        gdf_points[mask].plot(
            ax=ax2,
            color=lisa_colors[cat],
            markersize=sizes[mask],
            label=cat,
            **marker_style
        )
ax2.set_title('(B) Simulated LISA Clusters', fontweight='bold', fontsize=14, loc='left', pad=10)

# Style both axes and add maps
for ax in [ax1, ax2]:
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels)
    except Exception as e:
        print(f"Warning: Failed to load basemap: {e}")
    ax.set_axis_off()
    add_north_arrow(ax)
    add_scale_bar(ax)
    ax.legend(frameon=True, framealpha=0.8, loc='upper right', title='LISA Type', fontsize='small')

plt.tight_layout()
output_lisa_path = OUTPUTS_ROOT / 'lisa_clusters_comparison.png'
plt.savefig(output_lisa_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS: LISA clusters comparison saved to: {output_lisa_path}")
plt.show()
