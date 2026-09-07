import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# House-style palette
# ---------------------------------------------------------------------------
ORANGE  = '#E8482C'
NAVY    = '#194091'
BG      = '#F5F7F8'
TEXT    = '#37474F'
SUBTEXT = '#546E7A'
RULE    = '#CFD8DC'
TEAL    = '#4DB6AC'

# =========================================================================
# --- CONFIGURATION ---
# =========================================================================
# Toggle to use real validation data or synthetic comparison data
USE_VALIDATION_DATA  = True
VALIDATION_DATA_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\model_output_visualisation\synthetic_data\footfall_validation.csv')
SPEND_DATA_PATH      = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\model_output_visualisation\synthetic_data\spend_rankings.csv')

# Import config from the simulation directory
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'simulation'))
import config

# Paths for outputs and spatial data
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
CENTRE_DIR   = OUTPUTS_ROOT / 'centre_performance'
GPKG_PATH    = Path(config.RETAIL_CENTRES_GPKG)

# Ablation (Diffusion OFF) ensemble mean file
# Produced by: multi_runs/run_ablation.py + aggregate_ensemble_ablation.py
ABLATION_PERF_FILE = (
    PROJECT_ROOT / 'multi_runs' / 'results_ablation'
    / 'ensemble_centre_performance_ablation_mean.csv'
)

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

def add_scale_bar(ax, length_km=5, position=(0.98, 0.03)):
    """Draws a smaller scale bar in the bottom-right of the axes."""
    lat_deg = 53.4
    cos_lat = np.cos(np.radians(lat_deg))
    mercator_length = (length_km * 1000) / cos_lat

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

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
# --- Load Data ---
# =========================================================================
print("Locating latest retail centre performance run...")
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError(f"No retail centre performance CSV files found in {CENTRE_DIR}")

latest_file = cp_files[-1]
print(f"Loading simulation visits (Diffusion ON) from: {latest_file.name}")
df_sim = pd.read_csv(latest_file)

# Sum visits across columns except Retail_Centre
visit_cols = [c for c in df_sim.columns if c != 'Retail_Centre']
df_sim['sim_visits_on'] = df_sim[visit_cols].sum(axis=1)
df_sim['clean_id'] = df_sim['Retail_Centre'].apply(clean_id)
df_sim_clean = df_sim[['clean_id', 'sim_visits_on']].copy()

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

df_spend = None
if SPEND_DATA_PATH.exists():
    print(f"Loading real-world spend data from: {SPEND_DATA_PATH}")
    df_spend = pd.read_csv(SPEND_DATA_PATH)
    df_spend['clean_id'] = df_spend['Retail_Centre'].apply(clean_id)
    df_spend = df_spend[['clean_id', 'Spend Rank']].copy()

# Join datasets
gdf_mapped = gdf_centers.merge(df_sim_clean, on='clean_id', how='left')
gdf_mapped['sim_visits_on'] = gdf_mapped['sim_visits_on'].fillna(0)

if df_real is not None:
    gdf_mapped = gdf_mapped.merge(df_real, on='clean_id', how='left')
    gdf_mapped['real_visits'] = gdf_mapped['real_visits'].fillna(0)
else:
    # Generate synthetic validation data
    np.random.seed(42)
    noise_real = np.random.normal(0, 0.10 * gdf_mapped['sim_visits_on'])
    gdf_mapped['real_visits'] = np.clip(gdf_mapped['sim_visits_on'] + noise_real, 0, None).round(0)

if df_spend is not None:
    gdf_mapped = gdf_mapped.merge(df_spend, on='clean_id', how='left')
    gdf_mapped['Spend Rank'] = gdf_mapped['Spend Rank'].fillna(gdf_mapped['Spend Rank'].max() + 1)

# Load Diffusion OFF ablation data from the real ensemble run
df_sim_off = None
if ABLATION_PERF_FILE.exists():
    print(f"Loading real Diffusion OFF ablation data from: {ABLATION_PERF_FILE.name}")
    df_abl = pd.read_csv(ABLATION_PERF_FILE)
    # Sum across trip-type x mode columns to get total visits
    abl_visit_cols = [c for c in df_abl.columns if c not in ('Retail_Centre', 'Total_Revenue', 'Total_Visits')]
    if 'Total_Visits' in df_abl.columns:
        df_abl['sim_visits_off'] = df_abl['Total_Visits']
    else:
        df_abl['sim_visits_off'] = df_abl[abl_visit_cols].sum(axis=1)
    df_abl['clean_id'] = df_abl['Retail_Centre'].apply(clean_id) if 'Retail_Centre' in df_abl.columns \
        else df_abl.index.map(clean_id)
    df_sim_off = df_abl[['clean_id', 'sim_visits_off']].copy()
else:
    print("WARNING: Ablation data not found at:")
    print(f"  {ABLATION_PERF_FILE}")
    print("Run multi_runs/run_ablation.py then aggregate_ensemble_ablation.py to generate it.")
    print("Falling back to synthetic noise placeholder (NOT suitable for publication).")

# Join Diffusion OFF visits — real ablation data or synthetic fallback
if df_sim_off is not None:
    gdf_mapped = gdf_mapped.merge(df_sim_off, on='clean_id', how='left')
    gdf_mapped['sim_visits_off'] = gdf_mapped['sim_visits_off'].fillna(0)
else:
    # Synthetic fallback — clear warning already printed above
    np.random.seed(100)
    noise_off = np.random.normal(0, 0.25 * gdf_mapped['real_visits'])
    gdf_mapped['sim_visits_off'] = np.clip(gdf_mapped['real_visits'] + noise_off, 0, None).round(0)

# Calculate simulated ranks
gdf_mapped['sim_rank_on'] = gdf_mapped['sim_visits_on'].rank(ascending=False, method='min')
gdf_mapped['sim_rank_off'] = gdf_mapped['sim_visits_off'].rank(ascending=False, method='min')
gdf_mapped['real_rank'] = gdf_mapped['real_visits'].rank(ascending=False, method='min')

# Compute residuals (Observed Rank - Simulated Rank)
gdf_mapped['footfall_diff_on'] = gdf_mapped['real_rank'] - gdf_mapped['sim_rank_on']
gdf_mapped['footfall_diff_off'] = gdf_mapped['real_rank'] - gdf_mapped['sim_rank_off']
gdf_mapped['spend_diff_on'] = gdf_mapped['Spend Rank'] - gdf_mapped['sim_rank_on']
gdf_mapped['spend_diff_off'] = gdf_mapped['Spend Rank'] - gdf_mapped['sim_rank_off']

if 'Total_POI_' in gdf_mapped.columns:
    gdf_mapped['center_size'] = gdf_mapped['Total_POI_']
else:
    gdf_mapped['center_size'] = 100

print("Projecting geometry to EPSG:3857 (Web Mercator)...")
gdf_mapped = gdf_mapped.to_crs(epsg=3857)

# Centroids for point markers
gdf_points = gdf_mapped.copy()
gdf_points['geometry'] = gdf_points.geometry.centroid

# Load Liverpool boundary outline if available
BOUNDARY_PATHS = [
    Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA_boundary.shp'),
    Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA boundary_disolved.shp')
]
gdf_boundary = None
for path in BOUNDARY_PATHS:
    if path.exists():
        try:
            gdf_boundary = gpd.read_file(path)
            gdf_boundary = gdf_boundary.to_crs(epsg=3857)
            break
        except Exception as e:
            print(f"Warning: failed to read Liverpool boundary from {path}: {e}")

# =========================================================================
# --- Plotting side-by-side diffusion maps ---
# =========================================================================
print("Setting up 2x2 behavioral diffusion comparison maps...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 15))

# --- House-style: figure and axes backgrounds ---
fig.patch.set_facecolor(BG)
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(BG)

# --- House-style: header ---
fig.text(0.04, 0.99, 'Social Diffusion — Ablation Comparison',
         fontsize=22, fontweight='bold', color=TEXT, ha='left', va='top')
fig.text(0.04, 0.965,
         'Rank residuals (Observed − Simulated) for footfall and spend: Diffusion ON vs OFF',
         fontsize=12, color=SUBTEXT, ha='left', va='top')
rule_y = 0.955
fig.add_artist(lines.Line2D([0.04, 0.96], [rule_y, rule_y],
                             color=RULE, linewidth=1.5, transform=fig.transFigure,
                             clip_on=False))
marker_style = dict(edgecolor=TEXT, linewidth=0.6, alpha=0.65, zorder=6)
sizes = np.clip(gdf_points['center_size'] * 0.7, 10, 300)

def residual_qcut(series, q=5):
    """Bin residuals into quantiles."""
    try:
        cuts = pd.qcut(series, q=q, labels=False, duplicates='drop')
        return cuts.astype(float) + 1.0
    except Exception:
        ranks = series.rank(method='first').fillna(0)
        return pd.qcut(ranks, q=min(q, len(series.dropna())), labels=False, duplicates='drop').astype(float) + 1.0

# Quantile-bin all residuals
gdf_points['footfall_on_bin'] = residual_qcut(gdf_points['footfall_diff_on'], q=5)
gdf_points['footfall_off_bin'] = residual_qcut(gdf_points['footfall_diff_off'], q=5)
gdf_points['spend_on_bin'] = residual_qcut(gdf_points['spend_diff_on'], q=5)
gdf_points['spend_off_bin'] = residual_qcut(gdf_points['spend_diff_off'], q=5)

# Row 1: Footfall residuals
# Diffusion ON → Blues; Diffusion OFF → YlOrRd; Residual maps → RdBu
gdf_points.plot(ax=ax1, column='footfall_on_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax1.set_title('10.1 Diffusion ON — Footfall Residuals',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)

gdf_points.plot(ax=ax2, column='footfall_off_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax2.set_title('10.2 Diffusion OFF — Footfall Residuals',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)

# Row 2: Spend residuals
gdf_points.plot(ax=ax3, column='spend_on_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax3.set_title('10.3 Diffusion ON — Spend Residuals',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)

gdf_points.plot(ax=ax4, column='spend_off_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax4.set_title('10.4 Diffusion OFF — Spend Residuals',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)

# Apply boundaries and styling
for ax in [ax1, ax2, ax3, ax4]:
    if gdf_boundary is not None:
        try:
            gdf_boundary.boundary.plot(ax=ax, edgecolor=SUBTEXT, linewidth=0.8, zorder=5, alpha=0.6)
        except Exception as e:
            print(f"Warning: Failed to draw boundary: {e}")
    ax.set_axis_off()
    zoom_out(ax, 0.05)
    add_north_arrow(ax)
    add_scale_bar(ax)

# =========================================================================
# --- Save Figure ---
# =========================================================================
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_file = OUTPUTS_ROOT / 'diffusion_accuracy_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"SUCCESS: Diffusion comparison map saved to: {output_file}")
plt.close()
