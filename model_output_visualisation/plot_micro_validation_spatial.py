import os
import sys
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from pathlib import Path

# =========================================================================
# --- USER CONFIGURATION (COMPETITOR AREA ZOOM) ---
# =========================================================================
COMPETITOR_A = '4669'  # Smaller town centre (24 POIs)
COMPETITOR_B = '2446'  # Medium town centre (55 POIs)

# Paths
VISUALISATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALISATION_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
SYNTHETIC_DIR = VISUALISATION_DIR / 'synthetic_data'

# Import config for paths
sys.path.insert(0, str(PROJECT_ROOT / 'simulation'))
import config
GPKG_PATH = Path(config.RETAIL_CENTRES_GPKG)

# Clean IDs
def clean_id(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s

# =========================================================================
# --- 1. Load Data ---
# =========================================================================
print("Loading data for spatial micro-validation...")

# Find latest centre performance
CENTRE_DIR = OUTPUTS_ROOT / 'centre_performance'
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError("No performance runs found.")
df_sim = pd.read_csv(cp_files[-1])
df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
visit_cols = [c for c in df_sim.columns if c not in ['Retail_Centre', 'clean_id']]
df_sim['sim_visits'] = df_sim[visit_cols].sum(axis=1)

# Load geometry
gdf_centers = gpd.read_file(GPKG_PATH, layer='retail_centre_counts')
gdf_centers['clean_id'] = gdf_centers['RC_ID'].apply(clean_id)

# Load validation metrics
df_footfall = pd.read_csv(SYNTHETIC_DIR / 'footfall_validation.csv')
df_footfall['clean_id'] = df_footfall['Retail_Centre'].apply(clean_id)

df_vacancy = pd.read_csv(SYNTHETIC_DIR / 'vacancy_rankings.csv')
df_vacancy['clean_id'] = df_vacancy['Retail_Centre'].apply(clean_id)

df_spend = pd.read_csv(SYNTHETIC_DIR / 'spend_rankings.csv')
df_spend['clean_id'] = df_spend['Retail_Centre'].apply(clean_id)

# Merge visits to geometry
gdf_centers = gdf_centers.merge(df_sim[['clean_id', 'sim_visits']], on='clean_id', how='left')
gdf_centers = gdf_centers.merge(df_footfall[['clean_id', 'real_visits']], on='clean_id', how='left')
gdf_centers = gdf_centers.merge(df_vacancy[['clean_id', 'Vacancy Rank']], on='clean_id', how='left')
gdf_centers = gdf_centers.merge(df_spend[['clean_id', 'Spend Rank']], on='clean_id', how='left')

# Convert to Web Mercator for basemap compatibility
gdf_centers = gdf_centers.to_crs(epsg=3857)

# Filter to ONLY target competitors
comp_gdf = gdf_centers[gdf_centers['clean_id'].isin([COMPETITOR_A, COMPETITOR_B])].copy()
if len(comp_gdf) < 2:
    raise ValueError(f"Centres {COMPETITOR_A} or {COMPETITOR_B} not found in spatial data.")

# Create buffer around competitor centroids for the map viewport limits
bbox = comp_gdf.total_bounds  # [xmin, ymin, xmax, ymax]
x_mid = (bbox[0] + bbox[2]) / 2
y_mid = (bbox[1] + bbox[3]) / 2
# Expand viewport to make sure 5km scale bar fits comfortably
half_width = max((bbox[2] - bbox[0]) / 2, 3000) + 500
xlim = [x_mid - half_width, x_mid + half_width]
ylim = [y_mid - half_width, y_mid + half_width]

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
    
    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=2.5, zorder=5)
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.8, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

# =========================================================================
# --- 2. Plotting (Side-by-Side Map & Ratio) ---
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# --- LEFT MAP: Spatial Choice Zoom-In (Polygons Only) ---
# Define colors for the two centres
colors_map = {COMPETITOR_A: '#c2780e', COMPETITOR_B: '#1f4e79'}

for idx, row in comp_gdf.iterrows():
    c_id = row['clean_id']
    color = colors_map.get(c_id, '#64748b')
    
    # Plot actual boundary polygon of the centre
    gpd.GeoSeries(row.geometry).plot(ax=ax1, color=color, alpha=0.55, edgecolor=color, linewidth=2.5, zorder=3)
    
    # Plot centroid marker for visual anchor
    centroid = row.geometry.centroid
    ax1.plot(centroid.x, centroid.y, marker='o', color='#1e293b', markersize=6, zorder=4)
    
    # Metric callout text box
    label_text = (
        f"Centre ID: {c_id}\n"
        f"Sim Visits: {int(row['sim_visits'])}\n"
        f"Real Visits: {int(row['real_visits'])}\n"
        f"Spend Rank: {int(row['Spend Rank'])}\n"
        f"Vacancy Rank: {int(row['Vacancy Rank'])}"
    )
    # Adjust callout position based on which competitor it is
    xy_offset = (20, 20) if c_id == COMPETITOR_A else (-110, -90)
    ax1.annotate(
        label_text,
        xy=(centroid.x, centroid.y),
        xytext=xy_offset, textcoords="offset points",
        fontsize=9, fontweight='bold', color='#334155',
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafbfc", ec=color, linewidth=2.0, alpha=0.9, zorder=5),
        arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, zorder=4)
    )

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_title('6.1 Spatial Competitor Zoom-In (Actual Boundaries)', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)
ax1.axis('off')
ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron, zoom=13)
add_scale_bar(ax1)

# --- RIGHT PANEL: Choice & Spend Ratios Comparison ---
row_a = comp_gdf[comp_gdf['clean_id'] == COMPETITOR_A].iloc[0]
row_b = comp_gdf[comp_gdf['clean_id'] == COMPETITOR_B].iloc[0]

# Calculate ratios (A / B)
sim_ratio = row_a['sim_visits'] / row_b['sim_visits']
real_ratio = row_a['real_visits'] / row_b['real_visits']
# Spend value proxy: Inverse of spend ranks (lower rank = higher spend value)
spend_ratio = (1.0 / row_a['Spend Rank']) / (1.0 / row_b['Spend Rank'])

ratio_labels = ['Simulated Footfall\n(Model Choice)', 'Real Footfall\n(Empirical Choice)', 'Spend Value Proxy\n(Economic Output)']
ratios = [sim_ratio, real_ratio, spend_ratio]
bar_colors = ['#1f4e79', '#c2780e', '#5b3a7d']

rects = ax2.bar(ratio_labels, ratios, color=bar_colors, width=0.45, edgecolor='#475569', linewidth=0.8)

# Styling
ax2.set_facecolor('#fafbfc')
ax2.grid(True, axis='y', linestyle=':', alpha=0.7, color='#cbd5e1')
ax2.grid(False, axis='x')
ax2.spines['left'].set_color('#94a3b8')
ax2.spines['bottom'].set_color('#94a3b8')
ax2.spines['left'].set_linewidth(1.0)
ax2.spines['bottom'].set_linewidth(1.0)
ax2.tick_params(colors='#475569', width=1.0, labelsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel(f'Competitive Ratio Value ({COMPETITOR_A} / {COMPETITOR_B})', fontweight='bold', color='#334155', fontsize=10, labelpad=8)
ax2.set_title('6.2 Choice & Spend Ratio Calibration Check', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)

# Annotate ratio bars
for rect in rects:
    height = rect.get_height()
    ax2.annotate(f'{height:.2f}x',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color='#1e293b', fontweight='bold')

plt.tight_layout()

# Save image
output_path = OUTPUTS_ROOT / 'micro_validation_spatial.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Spatial micro-validation map saved to: {output_path}")
