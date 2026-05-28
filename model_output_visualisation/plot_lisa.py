import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================================================================
# --- CONFIGURATION ---
# =========================================================================
USE_VALIDATION_DATA = True
PROJECT_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM')
VALIDATION_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'footfall_validation.csv'
VACANCY_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'vacancy_rankings.csv'
SPEND_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'spend_rankings.csv'

# Import config from the simulation directory
import sys
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
        s = str(x).strip()
        if s.endswith('.0'):
            return s[:-2]
        return s
    except:
        return str(x)

# =========================================================================
# --- Helper Functions for Map Elements ---
# =========================================================================
def add_north_arrow(ax, position=(0.06, 0.94)):
    ax.annotate('N', xy=position, xytext=(position[0], position[1] - 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=1.5, headwidth=6, headlength=6, shrink=0.1),
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

def add_scale_bar(ax, length_km=5, position=(0.95, 0.05)):
    lat_deg = 53.4
    cos_lat = np.cos(np.radians(lat_deg))
    mercator_length = (length_km * 1000) / cos_lat
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + position[1] * (ymax - ymin)
    x_end = xmin + position[0] * (xmax - xmin)
    x_start = x_end - mercator_length
    tick_height = 0.012 * (ymax - ymin)
    
    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=1.8, zorder=5)
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.2, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.2, zorder=5)
    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.75, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.05'))

def zoom_out(ax, fraction=0.05):
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
    n = len(coords)
    np.random.seed(seed)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist_matrix[i])
        nn_idx = idx[1:k+1]
        W[i, nn_idx] = 1.0 / k
        
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        std_val = 1.0
    z = (values - mean_val) / std_val
    wz = W.dot(z)
    lisa = z * wz
    
    sim_lisa = np.zeros((n, n_permutations))
    for p in range(n_permutations):
        for i in range(n):
            idx_other = [j for j in range(n) if j != i]
            shuffled_other = np.random.permutation(z[idx_other])
            shuffled_z_i = np.zeros(n)
            shuffled_z_i[idx_other] = shuffled_other
            shuffled_z_i[i] = z[i]
            sim_lisa[i, p] = z[i] * W[i].dot(shuffled_z_i)
            
    p_values = np.zeros(n)
    for i in range(n):
        obs = lisa[i]
        sims = sim_lisa[i]
        if obs >= 0:
            count = np.sum(sims >= obs)
        else:
            count = np.sum(sims <= obs)
        p_values[i] = (count + 1) / (n_permutations + 1)
        
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
df_sim = pd.read_csv(latest_file)
visit_cols = [c for c in df_sim.columns if c != 'Retail_Centre']
df_sim['sim_visits'] = df_sim[visit_cols].sum(axis=1)
df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
df_sim_clean = df_sim[['clean_id', 'sim_visits']].copy()

gdf_centers = gpd.read_file(GPKG_PATH, layer='retail_centre_counts')
gdf_centers['clean_id'] = gdf_centers['RC_ID'].apply(clean_id)

df_real = pd.read_csv(VALIDATION_DATA_PATH)
df_real['clean_id'] = df_real['Retail_Centre'].apply(clean_id)
df_real = df_real[['clean_id', 'real_visits']].copy()

df_spend = pd.read_csv(SPEND_DATA_PATH)
df_spend['clean_id'] = df_spend['Retail_Centre'].apply(clean_id)
df_spend = df_spend[['clean_id', 'Spend Rank']].copy()

# Merge
gdf_mapped = gdf_centers.merge(df_sim_clean, on='clean_id', how='left')
gdf_mapped = gdf_mapped.merge(df_real, on='clean_id', how='left')
gdf_mapped = gdf_mapped.merge(df_spend, on='clean_id', how='left')
gdf_mapped['sim_visits'] = gdf_mapped['sim_visits'].fillna(0)
gdf_mapped['real_visits'] = gdf_mapped['real_visits'].fillna(0)

# Ranks
gdf_mapped['sim_rank'] = gdf_mapped['sim_visits'].rank(ascending=False, method='min')
gdf_mapped['Spend Rank'] = gdf_mapped['Spend Rank'].fillna(gdf_mapped['Spend Rank'].max() + 1)
gdf_mapped['sim_rank'] = gdf_mapped['sim_rank'].fillna(gdf_mapped['sim_rank'].max() + 1)

if 'Total_POI_' in gdf_mapped.columns:
    gdf_mapped['center_size'] = gdf_mapped['Total_POI_']
else:
    gdf_mapped['center_size'] = 100

gdf_mapped = gdf_mapped.to_crs(epsg=3857)

# Convert to centroids for calculations
gdf_points = gdf_mapped.copy()
gdf_points['geometry'] = gdf_points.geometry.centroid

BOUNDARY_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA boundary_disolved.shp')
gdf_boundary = None
if BOUNDARY_PATH.exists():
    try:
        gdf_boundary = gpd.read_file(BOUNDARY_PATH).to_crs(epsg=3857)
    except Exception as e:
        print(f"Warning: failed to read Liverpool boundary: {e}")

# Get coordinates for LISA
coords = np.array([(geom.x, geom.y) for geom in gdf_points.geometry])

# Calculate LISA categories for Footfall and Spend
print("Computing LISA categories for Footfall and Spend...")
gdf_points['lisa_foot_obs'] = calculate_lisa(coords, gdf_points['real_visits'].values, seed=10)
gdf_points['lisa_foot_sim'] = calculate_lisa(coords, gdf_points['sim_visits'].values, seed=20)
gdf_points['lisa_spend_obs'] = calculate_lisa(coords, (1.0 / gdf_points['Spend Rank']).values, seed=30)
gdf_points['lisa_spend_sim'] = calculate_lisa(coords, (1.0 / gdf_points['sim_rank']).values, seed=40)

# =========================================================================
# --- Plotting LISA Cluster Comparison (2x2 Grid) ---
# =========================================================================
print("Plotting 2x2 comparison: Observed and Simulated LISA clusters for Footfall & Spend...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 15))

# LISA Color Scheme
lisa_colors = {
    'High-High': '#d8527a',       # Crimson/Rose
    'Low-Low': '#2b7bba',         # Muted Ocean Blue
    'Low-High': '#a5c8e1',        # Light Blue
    'High-Low': '#f4a4b4',        # Light Crimson/Pink
    'Not Significant': '#e2e8f0'  # Off-white / light gray
}
marker_style = dict(edgecolor='black', linewidth=0.6, alpha=0.9, zorder=6)
sizes = np.clip(gdf_points['center_size'] * 1.5, 15, 600)
categories_order = ['High-High', 'Low-Low', 'Low-High', 'High-Low', 'Not Significant']

# Setup custom legend
legend_handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=lisa_colors[cat],
               markeredgecolor='black', markersize=10, linestyle='', label=cat)
    for cat in categories_order
]

# --- Row 1: Footfall LISA ---
for cat in categories_order:
    mask = gdf_points['lisa_foot_obs'] == cat
    if mask.any():
        gdf_points[mask].plot(ax=ax1, color=lisa_colors[cat], markersize=sizes[mask], label=cat, **marker_style)
ax1.set_title('9.1 Observed Footfall LISA Clusters', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)

for cat in categories_order:
    mask = gdf_points['lisa_foot_sim'] == cat
    if mask.any():
        gdf_points[mask].plot(ax=ax2, color=lisa_colors[cat], markersize=sizes[mask], label=cat, **marker_style)
ax2.set_title('9.2 Simulated Footfall LISA Clusters', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)

# --- Row 2: Spend LISA ---
for cat in categories_order:
    mask = gdf_points['lisa_spend_obs'] == cat
    if mask.any():
        gdf_points[mask].plot(ax=ax3, color=lisa_colors[cat], markersize=sizes[mask], label=cat, **marker_style)
ax3.set_title('9.3 Observed Spend LISA Clusters', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)

for cat in categories_order:
    mask = gdf_points['lisa_spend_sim'] == cat
    if mask.any():
        gdf_points[mask].plot(ax=ax4, color=lisa_colors[cat], markersize=sizes[mask], label=cat, **marker_style)
ax4.set_title('9.4 Simulated Spend LISA Clusters', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)

# Style all axes and add boundary
for ax in (ax1, ax2, ax3, ax4):
    if gdf_boundary is not None:
        try:
            gdf_boundary.boundary.plot(ax=ax, edgecolor='#475569', linewidth=0.8, zorder=5, alpha=0.6)
        except:
            pass
    ax.set_axis_off()
    zoom_out(ax, 0.05)
    add_north_arrow(ax)
    add_scale_bar(ax)
    ax.legend(handles=legend_handles, frameon=True, framealpha=0.9, facecolor='#fcfcfc', edgecolor='#e2e8f0',
              loc='upper right', title='LISA Type', fontsize='small', markerscale=1.0)

fig.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()

# Save LISA comparison
output_lisa_path = OUTPUTS_ROOT / 'lisa_diffusion_comparison.png'
plt.savefig(output_lisa_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS: LISA comparison saved to: {output_lisa_path}")
