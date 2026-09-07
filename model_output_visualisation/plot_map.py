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
# --- HOUSE STYLE PALETTE ---
# =========================================================================
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
USE_VALIDATION_DATA = True
PROJECT_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM')
VALIDATION_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'footfall_validation.csv'
VACANCY_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'vacancy_rankings.csv'
SPEND_DATA_PATH = PROJECT_ROOT / 'model_output_visualisation' / 'synthetic_data' / 'spend_rankings.csv'

# Plot options: True to plot rank-transformed visits
PLOT_RANKED = True

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
# --- 1. Load Data ---
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

# Load geometry data
print(f"Loading spatial data from: {GPKG_PATH.name}")
gdf_centers = gpd.read_file(GPKG_PATH, layer='retail_centre_counts')
gdf_centers['clean_id'] = gdf_centers['RC_ID'].apply(clean_id)

# Load real-world validation data
df_real = pd.read_csv(VALIDATION_DATA_PATH)
df_real['clean_id'] = df_real['Retail_Centre'].apply(clean_id)
df_real = df_real[['clean_id', 'real_visits']].copy()

df_vacancy = pd.read_csv(VACANCY_DATA_PATH)
df_vacancy['clean_id'] = df_vacancy['Retail_Centre'].apply(clean_id)
df_vacancy = df_vacancy[['clean_id', 'Vacancy Rank']].copy()

df_spend = pd.read_csv(SPEND_DATA_PATH)
df_spend['clean_id'] = df_spend['Retail_Centre'].apply(clean_id)
df_spend = df_spend[['clean_id', 'Spend Rank']].copy()

# Merge all datasets
gdf_mapped = gdf_centers.merge(df_sim_clean, on='clean_id', how='inner')
gdf_mapped = gdf_mapped.merge(df_real, on='clean_id', how='inner')
gdf_mapped = gdf_mapped.merge(df_vacancy, on='clean_id', how='inner')
gdf_mapped = gdf_mapped.merge(df_spend, on='clean_id', how='inner')

# Ranks
gdf_mapped['sim_rank'] = gdf_mapped['sim_visits'].rank(ascending=False, method='min')
gdf_mapped['real_rank'] = gdf_mapped['real_visits'].rank(ascending=False, method='min')

if 'Total_POI_' in gdf_mapped.columns:
    gdf_mapped['center_size'] = gdf_mapped['Total_POI_']
else:
    gdf_mapped['center_size'] = 100

# Project to Web Mercator (EPSG:3857)
print("Projecting geometry to EPSG:3857 (Web Mercator)...")
gdf_mapped = gdf_mapped.to_crs(epsg=3857)

# Convert to centroids for plotting
gdf_points = gdf_mapped.copy()
gdf_points['geometry'] = gdf_points.geometry.centroid

# Load Liverpool boundary
BOUNDARY_PATH = Path(r'C:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Boundary\CA boundary_disolved.shp')
gdf_boundary = None
if BOUNDARY_PATH.exists():
    try:
        gdf_boundary = gpd.read_file(BOUNDARY_PATH).to_crs(epsg=3857)
    except Exception as e:
        print(f"Warning: failed to read Liverpool boundary: {e}")

# =========================================================================
# --- 2. Helper Functions for Map Elements ---
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

    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=2.5, zorder=5)
    ax.plot([x_start, x_start], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.plot([x_end, x_end], [y_pos - tick_height, y_pos + tick_height], color='black', linewidth=1.5, zorder=5)
    ax.text((x_start + x_end) / 2, y_pos + tick_height * 0.8, f"{length_km} km",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='black', zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

def zoom_out(ax, fraction=0.1):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax - xmin) * fraction
    dy = (ymax - ymin) * fraction
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

def safe_qcut(series, q=5):
    try:
        # Invert rank values if needed to make larger values represent higher spend/visits
        cuts = pd.qcut(series, q=q, labels=False, duplicates='drop')
        return cuts.astype(float) + 1.0
    except Exception:
        ranks = series.rank(method='first').fillna(0)
        return pd.qcut(ranks, q=min(q, len(series.dropna())), labels=False, duplicates='drop').astype(float) + 1.0

# Create binned numeric categories 1..5 for consistent legends
gdf_points['sim_visits_bin'] = safe_qcut(gdf_points['sim_visits'], q=5)
gdf_points['real_visits_bin'] = safe_qcut(gdf_points['real_visits'], q=5)

# For Spend Ranks: Invert because rank 1 is highest spend value
gdf_points['sim_spend_bin'] = safe_qcut(1.0 / gdf_points['sim_rank'], q=5)
gdf_points['real_spend_bin'] = safe_qcut(1.0 / gdf_points['Spend Rank'], q=5)
# Styling and sizes
marker_style = dict(edgecolor=TEXT, linewidth=0.6, alpha=0.65, zorder=6)
sizes = np.clip(gdf_points['center_size'] * 0.7, 10, 300)

# Ranks residuals (Observed Rank - Simulated Rank)
# positive = model overpredicted (simulated rank is smaller/better number than observed)
# negative = model underpredicted
gdf_points['footfall_residual'] = gdf_points['real_rank'] - gdf_points['sim_rank']
gdf_points['spend_residual'] = gdf_points['Spend Rank'] - gdf_points['sim_rank']

# =========================================================================
# --- 3. Plotting 3x2 Spatial Reproduction Maps ---
# =========================================================================
print("Generating 3x2 spatial reproduction maps with residuals...")
fig_maps, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 15))
fig_maps.patch.set_facecolor(BG)

# Header
fig_maps.text(0.04, 0.99, 'Spatial Reproduction of Retail Activity',
              fontsize=22, fontweight='bold', color=TEXT, ha='left')
fig_maps.text(0.04, 0.96, 'Observed vs. simulated footfall and spend \u2014 Liverpool City Region retail centres.',
              fontsize=13, color=SUBTEXT, ha='left')

# Top Row: Footfall (Visits)
gdf_points.plot(ax=ax1, column='real_visits_bin', cmap='YlOrRd', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Observed Footfall (quantiles)', 'fontsize': 'small'}, **marker_style)
ax1.set_title('7.1 Observed Retail Footfall', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

gdf_points.plot(ax=ax2, column='sim_visits_bin', cmap='Blues', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Simulated Footfall (quantiles)', 'fontsize': 'small'}, **marker_style)
ax2.set_title('7.2 Simulated Retail Footfall', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

# Footfall Residuals (Quantile Bins)
def residual_qcut(series, q=5):
    """Bin residuals into quantiles with descriptive labels."""
    try:
        cuts = pd.qcut(series, q=q, labels=False, duplicates='drop')
        return cuts.astype(float) + 1.0
    except Exception:
        ranks = series.rank(method='first').fillna(0)
        return pd.qcut(ranks, q=min(q, len(series.dropna())), labels=False, duplicates='drop').astype(float) + 1.0

gdf_points['footfall_resid_bin'] = residual_qcut(gdf_points['footfall_residual'], q=5)
gdf_points.plot(ax=ax3, column='footfall_resid_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax3.set_title('7.3 Footfall Rank Residuals', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

# Bottom Row: Spend (Inverse Ranks)
gdf_points.plot(ax=ax4, column='real_spend_bin', cmap='YlOrRd', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Observed Spend (quantiles)', 'fontsize': 'small'}, **marker_style)
ax4.set_title('7.4 Observed Retail Spend', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

gdf_points.plot(ax=ax5, column='sim_spend_bin', cmap='Blues', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Simulated Spend (quantiles)', 'fontsize': 'small'}, **marker_style)
ax5.set_title('7.5 Simulated Retail Spend', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

# Spend Residuals (Quantile Bins)
gdf_points['spend_resid_bin'] = residual_qcut(gdf_points['spend_residual'], q=5)
gdf_points.plot(ax=ax6, column='spend_resid_bin', cmap='RdBu', categorical=True, markersize=sizes,
                legend=True, legend_kwds={'title': 'Residual Quantile', 'fontsize': 'small'}, **marker_style)
ax6.set_title('7.6 Spend Rank Residuals', fontsize=13, fontweight='bold', color=TEXT, loc='left', pad=8)

# Add boundary, north arrow and scale bar
for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
    if gdf_boundary is not None:
        try:
            gdf_boundary.boundary.plot(ax=ax, edgecolor=TEXT, linewidth=0.8, zorder=5, alpha=0.5)
        except:
            pass
    ax.set_axis_off()
    zoom_out(ax, 0.05)
    add_north_arrow(ax)
    add_scale_bar(ax)

# Save the maps figure
maps_output = OUTPUTS_ROOT / 'retail_activity_maps.png'
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_maps.savefig(maps_output, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"SUCCESS: Maps saved to: {maps_output}")
plt.close(fig_maps)

# =========================================================================
# --- 4. Plotting 1x3 Correlation Plots ---
# =========================================================================
print("Generating 1x3 correlation plots...")
fig_corr, (axc1, axc2, axc3) = plt.subplots(1, 3, figsize=(22, 6))
fig_corr.patch.set_facecolor(BG)

# Header
fig_corr.text(0.04, 1.04, 'Simulated vs. Observed Rank Correlations',
              fontsize=22, fontweight='bold', color=TEXT, ha='left')
fig_corr.text(0.04, 0.99, 'Pearson r and Spearman rho between simulated and observed footfall, spend, and vacancy ranks.',
              fontsize=13, color=SUBTEXT, ha='left')

df_plot = gdf_mapped.copy()
df_plot['centre_typ'] = df_plot['centre_typ'].map({'centre': 'Centre', 'retail_park': 'Retail Park'}).fillna('Unknown')
custom_palette = {'Centre': NAVY, 'Retail Park': ORANGE, 'Unknown': SUBTEXT}

# --- PANEL 1: Footfall Correlation ---
sns.scatterplot(data=df_plot, x='real_visits', y='sim_visits', hue='centre_typ', size='center_size',
                sizes=(40, 400), palette=custom_palette, alpha=0.85, edgecolor='black', linewidth=0.8, ax=axc1, legend=False)
sns.regplot(data=df_plot, x='real_visits', y='sim_visits', scatter=False, color=ORANGE, ax=axc1,
            line_kws={'linestyle': '--', 'linewidth': 2.0})

r_f, p_f = stats.pearsonr(df_plot['real_visits'], df_plot['sim_visits'])
rho_f, rp_f = stats.spearmanr(df_plot['real_visits'], df_plot['sim_visits'])
rmse_f = np.sqrt(np.mean((df_plot['sim_visits'] - df_plot['real_visits'])**2))

stats_f = f"Pearson r: {r_f:.3f} (p: {p_f:.2e})\nSpearman rho: {rho_f:.3f}\nRMSE: {rmse_f:.1f}"
axc1.text(0.05, 0.95, stats_f, transform=axc1.transAxes, ha='left', va='top', fontsize=9.0, fontweight='bold',
         color=TEXT, bbox=dict(facecolor=BG, alpha=0.95, edgecolor=RULE, boxstyle='round,pad=0.4'))
axc1.set_title('8.1 Footfall Correlation (Visits)', fontweight='bold', fontsize=13, color=TEXT, loc='left')
axc1.set_xlabel('Observed Visits', fontweight='bold', color=TEXT, fontsize=11)
axc1.set_ylabel('Simulated Visits', fontweight='bold', color=TEXT, fontsize=11)

# --- PANEL 2: Spend Rank Correlation ---
# We compare Simulated Rank vs. Real Spend Rank
sns.scatterplot(data=df_plot, x='Spend Rank', y='sim_rank', hue='centre_typ', size='center_size',
                sizes=(40, 400), palette=custom_palette, alpha=0.85, edgecolor='black', linewidth=0.8, ax=axc2, legend=False)
sns.regplot(data=df_plot, x='Spend Rank', y='sim_rank', scatter=False, color=ORANGE, ax=axc2,
            line_kws={'linestyle': '--', 'linewidth': 2.0})

r_s, p_s = stats.pearsonr(df_plot['Spend Rank'], df_plot['sim_rank'])
rho_s, rp_s = stats.spearmanr(df_plot['Spend Rank'], df_plot['sim_rank'])

stats_s = f"Pearson r: {r_s:.3f} (p: {p_s:.2e})\nSpearman rho: {rho_s:.3f}"
axc2.text(0.05, 0.95, stats_s, transform=axc2.transAxes, ha='left', va='top', fontsize=9.0, fontweight='bold',
         color=TEXT, bbox=dict(facecolor=BG, alpha=0.95, edgecolor=RULE, boxstyle='round,pad=0.4'))
axc2.set_title('8.2 Spend Rank Correlation', fontweight='bold', fontsize=13, color=TEXT, loc='left')
axc2.set_xlabel('Observed Spend Rank', fontweight='bold', color=TEXT, fontsize=11)
axc2.set_ylabel('Simulated Visits Rank', fontweight='bold', color=TEXT, fontsize=11)

# --- PANEL 3: Vacancy Rank Correlation ---
# We compare Simulated Rank vs. Real Vacancy Rank
sns.scatterplot(data=df_plot, x='Vacancy Rank', y='sim_rank', hue='centre_typ', size='center_size',
                sizes=(40, 400), palette=custom_palette, alpha=0.85, edgecolor='black', linewidth=0.8, ax=axc3, legend=False)
sns.regplot(data=df_plot, x='Vacancy Rank', y='sim_rank', scatter=False, color=ORANGE, ax=axc3,
            line_kws={'linestyle': '--', 'linewidth': 2.0})

r_v, p_v = stats.pearsonr(df_plot['Vacancy Rank'], df_plot['sim_rank'])
rho_v, rp_v = stats.spearmanr(df_plot['Vacancy Rank'], df_plot['sim_rank'])

stats_v = f"Pearson r: {r_v:.3f} (p: {p_v:.2e})\nSpearman rho: {rho_v:.3f}"
axc3.text(0.05, 0.95, stats_v, transform=axc3.transAxes, ha='left', va='top', fontsize=9.0, fontweight='bold',
         color=TEXT, bbox=dict(facecolor=BG, alpha=0.95, edgecolor=RULE, boxstyle='round,pad=0.4'))
axc3.set_title('8.3 Vacancy Rank Correlation', fontweight='bold', fontsize=13, color=TEXT, loc='left')
axc3.set_xlabel('Observed Vacancy Rank', fontweight='bold', color=TEXT, fontsize=11)
axc3.set_ylabel('Simulated Visits Rank', fontweight='bold', color=TEXT, fontsize=11)

# Global styling for correlation axes
for ax in [axc1, axc2, axc3]:
    ax.set_facecolor(BG)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, color=RULE)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6, color=RULE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(RULE)
    ax.spines['bottom'].set_color(RULE)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(colors=TEXT, width=1.0, labelsize=10)

# Legend setup (shared custom legend)
reg_line = Line2D([0], [0], color=ORANGE, linestyle='--', linewidth=2.0)
type_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
                for label, color in custom_palette.items()]
combined_handles = type_handles + [reg_line]
combined_labels = list(custom_palette.keys()) + ['Fitted Regression']
axc3.legend(combined_handles, combined_labels, frameon=True, framealpha=0.95,
            facecolor=BG, edgecolor=RULE, loc='lower right', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.90])

# Save correlation figure
corr_output = OUTPUTS_ROOT / 'retail_activity_correlation.png'
fig_corr.savefig(corr_output, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"SUCCESS: Correlation plot saved to: {corr_output}")
plt.close(fig_corr)
