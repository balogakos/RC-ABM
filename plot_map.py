import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from matplotlib.lines import Line2D

# =========================================================================
# --- CONFIGURATION ---
# =========================================================================
# Toggle to use real validation data or synthetic comparison data
USE_VALIDATION_DATA = False
VALIDATION_DATA_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\processed\validation_visits.csv')

# Plot options: True to plot rank-transformed visits, False for raw visits
PLOT_RANKED = False


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
# --- 1. Load Simulation Data ---
# =========================================================================
print("Locating latest retail centre performance run...")
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError(f"No retail centre performance CSV files found in {CENTRE_DIR}")

latest_file = cp_files[-1]
print(f"Loading simulation visits from: {latest_file.name}")
df_sim = pd.read_csv(latest_file)

# Sum all visit types to get total simulated visits per centre
visit_cols = [c for c in df_sim.columns if c != 'Retail_Centre']
df_sim['sim_visits'] = df_sim[visit_cols].sum(axis=1)
df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
df_sim_clean = df_sim[['clean_id', 'sim_visits']].copy()

# =========================================================================
# --- 2. Load Geometry Data ---
# =========================================================================
print(f"Loading spatial data from: {GPKG_PATH.name}")
if not GPKG_PATH.exists():
    raise FileNotFoundError(f"Retail centre GPKG not found at {GPKG_PATH}")

gdf_centers = gpd.read_file(GPKG_PATH, layer='retail_centre_counts')
gdf_centers['clean_id'] = gdf_centers['RC_ID'].apply(clean_id)

# =========================================================================
# --- 3. Handle Validation Data (Real-World Baseline) ---
# =========================================================================
df_real = None
if USE_VALIDATION_DATA:
    if VALIDATION_DATA_PATH.exists():
        print(f"Loading real-world validation data from: {VALIDATION_DATA_PATH}")
        df_real = pd.read_csv(VALIDATION_DATA_PATH)
        df_real['clean_id'] = df_real['Retail_Centre'].apply(clean_id)
        df_real = df_real[['clean_id', 'real_visits']].copy()
    else:
        print(f"WARNING: Validation file not found at {VALIDATION_DATA_PATH}.")
        print("Falling back to generating synthetic validation data.")

if df_real is None:
    # Generate synthetic validation data with noise for demo/placeholder purposes
    print("Generating synthetic real-world baseline data...")
    # Seed for reproducibility of placeholder noise
    np.random.seed(42)
    df_real = df_sim_clean.copy()
    # Add normal noise with std = 15% of the simulated visits (min 0)
    noise = np.random.normal(0, 0.15 * df_real['sim_visits'])
    df_real['real_visits'] = np.clip(df_real['sim_visits'] + noise, 0, None).round(0)
    df_real = df_real[['clean_id', 'real_visits']]

# Join datasets
gdf_mapped = gdf_centers.merge(df_sim_clean, on='clean_id', how='left')
gdf_mapped = gdf_mapped.merge(df_real, on='clean_id', how='left')

# Fill NaNs with 0
gdf_mapped['sim_visits'] = gdf_mapped['sim_visits'].fillna(0)
gdf_mapped['real_visits'] = gdf_mapped['real_visits'].fillna(0)
gdf_mapped['difference'] = gdf_mapped['sim_visits'] - gdf_mapped['real_visits']

# Define a size column for markers based on Total POIs (fallback to default if missing)
if 'Total_POI_' in gdf_mapped.columns:
    gdf_mapped['center_size'] = gdf_mapped['Total_POI_']
else:
    gdf_mapped['center_size'] = 100

# Project to Web Mercator (EPSG:3857) for compatibility with web-map tile providers
print("Projecting geometry to EPSG:3857 (Web Mercator)...")
gdf_mapped = gdf_mapped.to_crs(epsg=3857)

# Convert polygons to centroids for clean point marker plotting
gdf_points = gdf_mapped.copy()
gdf_points['geometry'] = gdf_points.geometry.centroid

# Load Liverpool boundary (optional) and project to Web Mercator
BOUNDARY_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA boundary_disolved.shp')
gdf_boundary = None
if BOUNDARY_PATH.exists():
    try:
        gdf_boundary = gpd.read_file(BOUNDARY_PATH)
        gdf_boundary = gdf_boundary.to_crs(epsg=3857)
    except Exception as e:
        print(f"Warning: failed to read Liverpool boundary: {e}")

# =========================================================================
# --- 4. Helper Functions for Map Elements ---
# =========================================================================
def add_north_arrow(ax, position=(0.06, 0.94)):
    """Draws a clean, minimalistic North arrow in the top-left of the axes."""
    ax.annotate('N', xy=position, xytext=(position[0], position[1] - 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=1.5, headwidth=6, headlength=6, shrink=0.1),
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

def add_scale_bar(ax, length_km=5, position=(0.06, 0.06)):
    """Draws a scale bar on the axes (position given in axes fraction).

    If the horizontal position is > 0.5 the bar will be drawn to the left
    so it remains visible when placed near the right-hand edge.
    """
    lat_deg = 53.4
    cos_lat = np.cos(np.radians(lat_deg))
    mercator_length = (length_km * 1000) / cos_lat

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Adjust the chosen length if it would be too large for the current view
    map_width_km = ((xmax - xmin) * cos_lat) / 1000.0
    if map_width_km < length_km * 2:
        if map_width_km >= 4.0:
            length_km = 2
        else:
            length_km = 1
        mercator_length = (length_km * 1000) / cos_lat

    # Calculate proposed start position in data coords
    frac_x, frac_y = position
    y_pos = ymin + frac_y * (ymax - ymin)

    # If positioned on the right half, draw the bar leftwards to keep it inside frame
    if frac_x > 0.5:
        x_end = xmin + frac_x * (xmax - xmin)
        x_start = x_end - mercator_length
    else:
        x_start = xmin + frac_x * (xmax - xmin)
        x_end = x_start + mercator_length

    tick_height = 0.012 * (ymax - ymin)

    # Draw scale bar line and ticks
    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=2.5, zorder=5)
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)

    # Add text label above the bar
    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.8, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

def zoom_out(ax, fraction=0.1):
    """Expand axis limits by a fraction to give extra map padding."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax - xmin) * fraction
    dy = (ymax - ymin) * fraction
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

# =========================================================================
# --- 5. Plotting 2x2 Layout ---
# =========================================================================
print("Generating maps and separate correlation plot outputs...")

# Prepare consistent binned categories for the three maps to ensure stable legend ranks
def safe_qcut(series, q=5):
    try:
        cuts = pd.qcut(series.replace([np.inf, -np.inf], np.nan).fillna(0), q=q, labels=False, duplicates='drop')
        # If qcut returned float dtype or all NaN, fallback to rank-based cut
        if cuts.isnull().all():
            return pd.Series(np.zeros(len(series)), index=series.index)
        return cuts.astype(float) + 1.0
    except Exception:
        # fallback: equal-frequency via rank
        ranks = series.rank(method='first').fillna(0)
        return pd.qcut(ranks, q=min(q, int(len(series.dropna()))), labels=False, duplicates='drop').astype(float) + 1.0

# Create binned numeric categories 1..5 for consistent legends
gdf_points['sim_bin'] = safe_qcut(gdf_points['sim_visits'], q=5)
gdf_points['real_bin'] = safe_qcut(gdf_points['real_visits'], q=5)

# For difference, create symmetric bins around zero
max_diff = max(abs(gdf_points['difference'].min()), abs(gdf_points['difference'].max()))
if max_diff == 0:
    max_diff = 1.0
diff_bins = np.linspace(-max_diff, max_diff, 6)
gdf_points['diff_bin'] = pd.cut(gdf_points['difference'].fillna(0), bins=diff_bins, labels=False, include_lowest=True).astype(float) + 1.0

# Common styling and sizes
marker_style = dict(edgecolor='black', linewidth=0.6, alpha=0.9)
sizes = np.clip(gdf_points['center_size'] * 1.5, 15, 600)

# FIGURE 1: Three maps side-by-side
fig_maps, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sc1 = gdf_points.plot(
    ax=ax1,
    column='sim_bin',
    cmap='Blues',
    categorical=True,
    markersize=sizes,
    legend=True,
    legend_kwds={'title': 'Simulated (quantiles)', 'fontsize': 'small'},
    **marker_style,
    zorder=6
)
ax1.set_title('(A) Simulated Activity', fontweight='bold', fontsize=12, loc='left')

sc2 = gdf_points.plot(
    ax=ax2,
    column='real_bin',
    cmap='Blues',
    categorical=True,
    markersize=sizes,
    legend=True,
    legend_kwds={'title': 'Baseline (quantiles)', 'fontsize': 'small'},
    **marker_style,
    zorder=6
)
ax2.set_title('(B) Real-World Baseline', fontweight='bold', fontsize=12, loc='left')

sc3 = gdf_points.plot(
    ax=ax3,
    column='diff_bin',
    cmap='RdBu',
    categorical=True,
    markersize=sizes,
    legend=True,
    legend_kwds={'title': 'Residual (binned)', 'fontsize': 'small'},
    **marker_style,
    zorder=6
)
ax3.set_title('(C) Simulation Residuals', fontweight='bold', fontsize=12, loc='left')

# Add boundary, north arrow and scale bar for each map axis
for ax in (ax1, ax2, ax3):
    if gdf_boundary is not None:
        try:
            gdf_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, zorder=5, alpha=0.9)
        except Exception as e:
            print(f"Warning: Failed to draw Liverpool boundary: {e}")
    ax.set_axis_off()
    zoom_out(ax, 0.10)
    add_north_arrow(ax)
    # move further into corner
    add_scale_bar(ax, position=(0.99, 0.02))

# Save the maps figure
maps_output = OUTPUTS_ROOT / 'retail_activity_maps.png'
fig_maps.subplots_adjust(wspace=0.05)
plt.savefig(maps_output, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Maps saved to: {maps_output}")
plt.close(fig_maps)

# =========================================================================
# --- 6. Correlation Performance Plot (Panel D) ---
# =========================================================================
print("Calculating validation performance statistics...")
x_data = gdf_mapped['real_visits']
y_data = gdf_mapped['sim_visits']
n_centres = len(gdf_mapped)

# Calculate statistical metrics on raw values
r_val, r_pval = stats.pearsonr(x_data, y_data)
rho_val, rho_pval = stats.spearmanr(x_data, y_data)
rmse = np.sqrt(np.mean((y_data - x_data)**2))
mae = np.mean(np.abs(y_data - x_data))

# Determine what data to plot on the axes based on PLOT_RANKED config
if PLOT_RANKED:
    x_plot = gdf_mapped['real_visits'].rank()
    y_plot = gdf_mapped['sim_visits'].rank()
    x_label = 'Validation Data (Rank)'
    y_label = 'Simulated Data (Rank)'
    title_suffix = ' (Ranked)'
else:
    x_plot = gdf_mapped['real_visits']
    y_plot = gdf_mapped['sim_visits']
    x_label = 'Validation Data (Visits)'
    y_label = 'Simulated Data (Visits)'
    title_suffix = ' (Raw)'

df_plot = gdf_mapped.copy()
df_plot['x_plot'] = x_plot
df_plot['y_plot'] = y_plot

# Clean labels for the legend
df_plot['centre_typ'] = df_plot['centre_typ'].map({
    'centre': 'Centre', 
    'retail_park': 'Retail Park'
}).fillna('Unknown')

# Styling options matching plot_results.py style rules
custom_palette = {'Centre': '#3498db', 'Retail Park': '#e67e22', 'Unknown': '#7f8c8d'}

# =========================================================================
# --- 6. Correlation Performance Plot (separate figure) ---
# =========================================================================
print("Generating separate correlation plot...")

# Prepare plotting data
x_data = gdf_mapped['real_visits']
y_data = gdf_mapped['sim_visits']
n_centres = len(gdf_mapped)

# Calculate statistics
r_val, r_pval = stats.pearsonr(x_data, y_data)
rho_val, rho_pval = stats.spearmanr(x_data, y_data)
rmse = np.sqrt(np.mean((y_data - x_data)**2))
mae = np.mean(np.abs(y_data - x_data))

if PLOT_RANKED:
    x_plot = gdf_mapped['real_visits'].rank()
    y_plot = gdf_mapped['sim_visits'].rank()
    x_label = 'Validation Data (Rank)'
    y_label = 'Simulated Data (Rank)'
    title_suffix = ' (Ranked)'
else:
    x_plot = gdf_mapped['real_visits']
    y_plot = gdf_mapped['sim_visits']
    x_label = 'Validation Data (Visits)'
    y_label = 'Simulated Data (Visits)'
    title_suffix = ' (Raw)'

df_plot = gdf_mapped.copy()
df_plot['x_plot'] = x_plot
df_plot['y_plot'] = y_plot

# Normalise centre type labels
df_plot['centre_typ'] = df_plot['centre_typ'].map({
    'centre': 'Centre',
    'retail_park': 'Retail Park'
}).fillna('Unknown')

custom_palette = {'Centre': '#3498db', 'Retail Park': '#e67e22', 'Unknown': '#7f8c8d'}

# FIGURE 2: Correlation plot
fig_corr, axc = plt.subplots(1, 1, figsize=(8, 6))

sns.scatterplot(
    data=df_plot,
    x='x_plot',
    y='y_plot',
    hue='centre_typ',
    size='center_size',
    sizes=(40, 400),
    palette=custom_palette,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8,
    ax=axc,
    legend=False  # we'll add custom legends
)

# Regression line
sns.regplot(
    data=df_plot,
    x='x_plot',
    y='y_plot',
    scatter=False,
    color='#e74c3c',
    ax=axc,
    line_kws={'linestyle': '--', 'linewidth': 2.0}
)

axc.set_title(f'Correlation Performance{title_suffix}', fontweight='bold', fontsize=13, loc='left')
axc.set_xlabel(x_label, fontweight='bold', fontsize=11)
axc.set_ylabel(y_label, fontweight='bold', fontsize=11)
axc.spines['top'].set_visible(False)
axc.spines['right'].set_visible(False)
axc.spines['left'].set_linewidth(1.2)
axc.spines['bottom'].set_linewidth(1.2)
axc.tick_params(width=1.2, labelsize=10)
axc.grid(False)

# Combined size and regression legend (bottom-right)
size_vals = df_plot['center_size'].dropna()
if len(size_vals) == 0:
    size_vals = pd.Series([100])
minv, maxv = size_vals.min(), size_vals.max()
rep_values = [10, 50, 100]
smin, smax = 40, 400
def map_area(raw):
    r = max(minv, min(maxv, raw)) if maxv > minv else raw
    if maxv == minv:
        return (smin + smax) / 2.0
    return smin + (r - minv) / (maxv - minv) * (smax - smin)
handles_size = [axc.scatter([], [], s=map_area(v), color='gray', edgecolor='black') for v in rep_values]
labels_size = [f'{v}' for v in rep_values]

# Regression line handle
reg_line = Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2.0)

# Combine both into a single legend in bottom-right corner
combined_handles = handles_size + [reg_line]
combined_labels = labels_size + ['Fitted Regression']
leg_combined = axc.legend(combined_handles, combined_labels, title='Centre size (POIs)', 
                          frameon=True, framealpha=0.95, loc='lower right', fontsize='medium')

# Performance textbox
stats_text = (
    f"Pearson r: {r_val:.3f} (p: {r_pval:.2e})\n"
    f"Spearman rho: {rho_val:.3f} (p: {rho_pval:.2e})\n"
    f"RMSE: {rmse:.1f}\n"
    f"MAE: {mae:.1f}\n"
    f"Centres (n): {n_centres}"
)
axc.text(0.05, 0.95, stats_text, transform=axc.transAxes, ha='left', va='top', fontsize=9.0, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.4'))

# Save correlation figure
corr_output = OUTPUTS_ROOT / 'retail_activity_correlation.png'
plt.tight_layout()
plt.savefig(corr_output, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Correlation plot saved to: {corr_output}")
plt.close(fig_corr)
# Draw correlation scatter plot using Seaborn
sns.scatterplot(
    data=df_plot,
    x='x_plot',
    y='y_plot',
    hue='centre_typ',
    size='center_size',
    sizes=(40, 400),
    palette=custom_palette,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8,
    ax=ax4
)

# Draw fitted regression line with 95% confidence interval
sns.regplot(
    data=df_plot,
    x='x_plot',
    y='y_plot',
    scatter=False,
    color='#e74c3c',
    ax=ax4,
    line_kws={'linestyle': '--', 'linewidth': 2.0, 'label': 'Fitted Regression'}
)

# Match styling to plot_results.py layout details
ax4.set_title(f'(D) Correlation Performance{title_suffix}', fontweight='bold', fontsize=14, loc='left', pad=10)
ax4.set_xlabel(x_label, fontweight='bold', fontsize=11)
ax4.set_ylabel(y_label, fontweight='bold', fontsize=11)

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(1.2)
ax4.spines['bottom'].set_linewidth(1.2)
ax4.tick_params(width=1.2, labelsize=10)
ax4.grid(False)

# Combined size and regression legend (bottom-right)
size_vals = df_plot['center_size'].dropna()
if len(size_vals) == 0:
    size_vals = pd.Series([100])
minv, maxv = size_vals.min(), size_vals.max()
# Use simple category bounds as requested
rep_values = [10, 50, 100]

# Map raw values to marker areas used by seaborn (sizes=(40,400))
smin, smax = 40, 400
def map_area_from_bounds(raw):
    # clamp to min/max observed
    r = max(minv, min(maxv, raw)) if maxv > minv else raw
    if maxv == minv:
        return (smin + smax) / 2.0
    return smin + (r - minv) / (maxv - minv) * (smax - smin)

handles_size = [ax4.scatter([], [], s=map_area_from_bounds(v), color='gray', edgecolor='black') for v in rep_values]
labels_size = [f'{v}' for v in rep_values]

# Get regression line handle
reg_lines = [ln for ln in ax4.get_lines() if ln.get_label() == 'Fitted Regression']
reg_handle = reg_lines[0] if len(reg_lines) > 0 else Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2.0)

# Combine both into a single legend in bottom-right corner
combined_handles = handles_size + [reg_handle]
combined_labels = labels_size + ['Fitted Regression']
leg = ax4.legend(combined_handles, combined_labels, title='Centre size (POIs)', frameon=True, framealpha=0.95,
                 loc='lower right', fontsize='medium')

# Add textbox with performance metrics in the top left corner (no overlaps)
stats_text = (
    f"Pearson $r$: {r_val:.3f} (p: {r_pval:.2e})\n"
    f"Spearman $\\rho$: {rho_val:.3f} (p: {rho_pval:.2e})\n"
    f"RMSE: {rmse:.1f}\n"
    f"MAE: {mae:.1f}\n"
    f"Centres ($n$): {n_centres}"
)
ax4.text(
    0.05, 0.95, stats_text,
    transform=ax4.transAxes,
    ha='left', va='top', fontsize=9.5, fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.4')
)

# =========================================================================
# --- 7. Save Output Figure ---
# =========================================================================
plt.tight_layout()

# Save default mapping output file
default_output = OUTPUTS_ROOT / 'retail_activity_mapping.png'
plt.savefig(default_output, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Default map saved to: {default_output}")

# Save mode-specific file
if PLOT_RANKED:
    mode_output = OUTPUTS_ROOT / 'retail_activity_mapping_ranked.png'
else:
    mode_output = OUTPUTS_ROOT / 'retail_activity_mapping_raw.png'

plt.savefig(mode_output, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Mode-specific map saved to: {mode_output}")

plt.show()

