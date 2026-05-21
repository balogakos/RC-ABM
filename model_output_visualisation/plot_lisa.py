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
USE_VALIDATION_DATA = True
VALIDATION_DATA_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\model_output_visualisation\synthetic_data\footfall_validation.csv')

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

def add_scale_bar(ax, length_km=2, position=(0.98, 0.03)):
    """Draws a smaller scale bar in the bottom-right of the axes."""
    lat_deg = 53.4
    cos_lat = np.cos(np.radians(lat_deg))
    mercator_length = (length_km * 1000) / cos_lat

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    map_width_km = ((xmax - xmin) * cos_lat) / 1000.0
    if map_width_km < length_km * 2:
        if map_width_km >= 4.0:
            length_km = 1
        else:
            length_km = 0.5
        mercator_length = (length_km * 1000) / cos_lat

    frac_x, frac_y = position
    y_pos = ymin + frac_y * (ymax - ymin)

    if frac_x > 0.5:
        x_end = xmin + frac_x * (xmax - xmin)
        x_start = x_end - mercator_length
    else:
        x_start = xmin + frac_x * (xmax - xmin)
        x_end = x_start + mercator_length

    tick_height = 0.01 * (ymax - ymin)

    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=1.8, zorder=5)
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.2, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.2, zorder=5)

    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.75, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.05'))

def zoom_out(ax, fraction=0.1):
    """Expand axis limits by a fraction to give extra map padding."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax - xmin) * fraction
    dy = (ymax - ymin) * fraction
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

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

# Load Liverpool boundary outline if available and project it for maps
BOUNDARY_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA boundary_disolved.shp')
gdf_boundary = None
if BOUNDARY_PATH.exists():
    try:
        gdf_boundary = gpd.read_file(BOUNDARY_PATH)
        gdf_boundary = gdf_boundary.to_crs(epsg=3857)
    except Exception as e:
        print(f"Warning: failed to read Liverpool boundary: {e}")

# Get coordinates for LISA
coords = np.array([(geom.x, geom.y) for geom in gdf_points.geometry])

# Calculate LISA categories
print("Computing LISA categories for Observed vs Simulated visits...")
gdf_points['lisa_obs'] = calculate_lisa(coords, gdf_points['real_visits'].values, seed=10)
gdf_points['lisa_sim'] = calculate_lisa(coords, gdf_points['sim_visits'].values, seed=20)

# =========================================================================
# --- Load Diffusion Data for comparison ---
# =========================================================================
print("Loading diffusion comparison data...")
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if cp_files:
    latest_file = cp_files[-1]
    df_sim = pd.read_csv(latest_file)
    visit_cols = [c for c in df_sim.columns if c != 'Retail_Centre']
    df_sim['sim_visits_on'] = df_sim[visit_cols].sum(axis=1)
    df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
    df_sim_clean = df_sim[['clean_id', 'sim_visits_on']].copy()
    
    # Merge to get diffusion data, preserving geometry from gdf_points
    gdf_diff = gdf_points.merge(df_sim_clean, on='clean_id', how='left')
    gdf_diff['sim_visits_on'] = gdf_diff['sim_visits_on'].fillna(0)
    
    # Generate synthetic Diffusion OFF data
    np.random.seed(100)
    noise_off = np.random.normal(0, 0.25 * gdf_diff['real_visits'])
    gdf_diff['sim_visits_off'] = np.clip(gdf_diff['real_visits'] + noise_off, 0, None).round(0)
    
    # Compute residuals
    gdf_diff['diff_on'] = gdf_diff['sim_visits_on'] - gdf_diff['real_visits']
    gdf_diff['diff_off'] = gdf_diff['sim_visits_off'] - gdf_diff['real_visits']
    
    # For binning: determine common limits
    max_diff = max(
        abs(gdf_diff['diff_on'].min()), abs(gdf_diff['diff_on'].max()),
        abs(gdf_diff['diff_off'].min()), abs(gdf_diff['diff_off'].max())
    )
    if max_diff == 0:
        max_diff = 1
    bins = [-max_diff * 0.5, -max_diff * 0.1, max_diff * 0.1, max_diff * 0.5]
    
    gdf_diff_points = gdf_diff.copy()
    gdf_diff_points['geometry'] = gdf_diff_points.geometry.centroid
else:
    gdf_diff = None
    gdf_diff_points = None
    print("WARNING: Could not load diffusion data")

# =========================================================================
# --- Plotting LISA Cluster Comparison with Diffusion ---
# =========================================================================
print("Plotting 2x2 comparison: LISA (top) and Diffusion Residuals (bottom)...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

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
legend_handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=lisa_colors[cat],
               markeredgecolor='black', markersize=10, linestyle='', label=cat)
    for cat in categories_order
]
for ax in [ax1, ax2]:
    if gdf_boundary is not None:
        try:
            gdf_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, zorder=5, alpha=0.9)
        except Exception as e:
            print(f"Warning: Failed to draw Liverpool boundary: {e}")
    ax.set_axis_off()
    zoom_out(ax, 0.10)
    add_north_arrow(ax)
    add_scale_bar(ax)
    ax.legend(handles=legend_handles, frameon=True, framealpha=0.8,
              loc='upper right', title='LISA Type', fontsize='small', markerscale=1.0,
              handletextpad=0.5, labelspacing=0.7, borderpad=0.4)

# Panel C & D: Diffusion Residuals (only if diffusion data loaded)
if gdf_diff_points is not None:
    diff_marker_style = dict(edgecolor='black', linewidth=0.8, alpha=0.9)
    diff_sizes = np.clip(gdf_diff_points['center_size'] * 1.5, 15, 600)
    
    # Panel C: Diffusion ON Residuals
    gdf_diff_points.plot(
        ax=ax3,
        column='diff_on',
        cmap='RdBu',
        scheme='UserDefined',
        classification_kwds={'bins': bins},
        markersize=diff_sizes,
        legend=True,
        legend_kwds={'loc': 'upper right', 'title': 'Residual (Sim - Real)', 'fmt': '{:.0f}', 'fontsize': 'small'},
        **diff_marker_style,
        zorder=6
    )
    ax3.set_title('(C) Diffusion ON - Residuals', fontweight='bold', fontsize=14, loc='left', pad=10)
    
    # Panel D: Diffusion OFF Residuals
    gdf_diff_points.plot(
        ax=ax4,
        column='diff_off',
        cmap='RdBu',
        scheme='UserDefined',
        classification_kwds={'bins': bins},
        markersize=diff_sizes,
        legend=True,
        legend_kwds={'loc': 'upper right', 'title': 'Residual (Sim - Real)', 'fmt': '{:.0f}', 'fontsize': 'small'},
        **diff_marker_style,
        zorder=6
    )
    ax4.set_title('(D) Diffusion OFF - Residuals', fontweight='bold', fontsize=14, loc='left', pad=10)
    
    # Style diffusion axes
    for ax in [ax3, ax4]:
        if gdf_boundary is not None:
            try:
                gdf_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, zorder=5, alpha=0.9)
            except Exception as e:
                print(f"Warning: Failed to draw Liverpool boundary: {e}")
        ax.set_axis_off()
        zoom_out(ax, 0.10)
        add_north_arrow(ax)
        add_scale_bar(ax)

fig.subplots_adjust(wspace=0.05, hspace=0.05)
output_lisa_path = OUTPUTS_ROOT / 'lisa_diffusion_comparison.png'
plt.savefig(output_lisa_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS: LISA/Diffusion comparison saved to: {output_lisa_path}")
plt.show()
