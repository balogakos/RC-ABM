import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================================
# --- USER CONFIGURATION ---
# =========================================================================
# Options:
# 'latest'  : Plots only the single most recent simulation run (no SD shading, clean lines).
# 'average' : Plots the average of all runs found in the subfolders.
#             If > 1 run is found, standard deviations are shown as shaded alpha backgrounds.
PLOT_MODE = 'latest'  

# Root paths for outputs (updated subdirectory structure)
OUTPUTS_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs')
DAILY_DIR = OUTPUTS_ROOT / 'daily_summaries'
CONV_DIR = OUTPUTS_ROOT / 'utility_convergence'

use_rows = 180
retail_types = ['comparison', 'entertainment', 'food_drink', 'grocery', 'service']

print(f"Plotting mode: {PLOT_MODE.upper()}")

# =========================================================================
# --- 1. Load Data based on PLOT_MODE ---
# =========================================================================
daily_files = []
conv_files = []

if PLOT_MODE == 'latest':
    # Find the single latest daily summary file
    d_files = sorted(DAILY_DIR.glob('daily_summary_*.csv'), key=os.path.getmtime)
    if d_files:
        daily_files = [d_files[-1]]
    
    # Find the single latest utility convergence file
    c_files = sorted(CONV_DIR.glob('utility_convergence_*.csv'), key=os.path.getmtime)
    if c_files:
        conv_files = [c_files[-1]]
else:
    # 'average' mode: load all files found in the subdirectories
    daily_files = sorted(DAILY_DIR.glob('daily_summary_*.csv'))
    conv_files = sorted(CONV_DIR.glob('utility_convergence_*.csv'))

# Process daily summaries (Panels 1 & 2)
daily_dfs = []
for p in daily_files:
    tmp = pd.read_csv(p)
    if 'Day' not in tmp.columns:
        continue
    cols = ['Day'] + [c for c in retail_types + ['Total_Visits'] if c in tmp.columns]
    tmp = tmp.loc[:, cols].head(use_rows).copy()
    daily_dfs.append(tmp)

# Process convergence metrics (Panels 3 & 4)
conv_dfs = []
for p in conv_files:
    tmp = pd.read_csv(p)
    if 'Day' not in tmp.columns:
        continue
    cols = ['Day', 'JSD', 'KS_Distance', 
            'Spearman_Cum_Day_to_Day', 'Spearman_Cum_Final_Anchor',
            'Spearman_Daily_Day_to_Day', 'Spearman_Daily_Final_Anchor']
    cols = [c for c in cols if c in tmp.columns]
    tmp = tmp.loc[:, cols].head(use_rows).copy()
    conv_dfs.append(tmp)

num_daily_runs = len(daily_dfs)
num_conv_runs = len(conv_dfs)

print(f"Loaded {num_daily_runs} daily summary run(s).")
print(f"Loaded {num_conv_runs} utility convergence run(s).")

if num_daily_runs == 0:
    raise ValueError(f"No daily summary CSV files found in {DAILY_DIR}")
if num_conv_runs == 0:
    raise ValueError(f"No utility convergence CSV files found in {CONV_DIR}")

# --- Aggregate daily summaries ---
df_daily_all = pd.concat(daily_dfs, keys=[str(i) for i in range(num_daily_runs)], names=['source']).reset_index(level='source')
agg_daily = df_daily_all.groupby('Day').agg({c: ['mean', 'std'] for c in retail_types + ['Total_Visits']})
agg_daily.columns = [f'{col}_{stat}' for col, stat in agg_daily.columns]
df_daily = agg_daily.reset_index()

# --- Aggregate utility convergence ---
df_conv_all = pd.concat(conv_dfs, keys=[str(i) for i in range(num_conv_runs)], names=['source']).reset_index(level='source')
metric_cols = [c for c in ['JSD', 'KS_Distance', 
                          'Spearman_Cum_Day_to_Day', 'Spearman_Cum_Final_Anchor',
                          'Spearman_Daily_Day_to_Day', 'Spearman_Daily_Final_Anchor'] 
               if c in df_conv_all.columns]
agg_conv = df_conv_all.groupby('Day').agg({c: ['mean', 'std'] for c in metric_cols})
agg_conv.columns = [f'{col}_{stat}' for col, stat in agg_conv.columns]
df_conv = agg_conv.reset_index()

# =========================================================================
# --- 2. Plotting (1x2 Grid, Premium Aesthetics) ---
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

# Premium color palette for retail types
category_colors = {
    'comparison': '#1f4e79',      # Muted steel blue
    'entertainment': '#c2780e',   # Warm bronze
    'food_drink': '#a23b5f',      # Muted burgundy/rose
    'grocery': '#2d6a4f',         # Forest green
    'service': '#5b3a7d'          # Muted plum/purple
}

# --- PANEL 1: Daily Visits by Retail Type ---
for col in retail_types:
    legend_name = col.replace('_', ' ').title()
    color = category_colors.get(col, '#7f8c8d')
    line, = ax1.plot(df_daily['Day'], df_daily[f'{col}_mean'], linewidth=2.5, color=color, label=legend_name)
    
    # Shade SD only if we have more than 1 run
    if num_daily_runs > 1:
        lower = df_daily[f'{col}_mean'] - df_daily[f'{col}_std'].fillna(0.0)
        upper = df_daily[f'{col}_mean'] + df_daily[f'{col}_std'].fillna(0.0)
        ax1.fill_between(df_daily['Day'], lower, upper, color=color, alpha=0.15)

ax1.set_title('3.1 Daily Visits by Retail Type', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)
ax1.set_xlabel('Day')
ax1.set_ylabel('Number of Visits')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, framealpha=0.9, facecolor='#fcfcfc', edgecolor='#e2e8f0', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

# --- PANEL 2: Total Daily Visits ---
total_color = '#2c3e50'  # Deep slate gray/blue
line, = ax2.plot(df_daily['Day'], df_daily['Total_Visits_mean'], linewidth=3, color=total_color, label='Total Visits')

# Shade SD only if we have more than 1 run
if num_daily_runs > 1:
    lower_total = df_daily['Total_Visits_mean'] - df_daily['Total_Visits_std'].fillna(0.0)
    upper_total = df_daily['Total_Visits_mean'] + df_daily['Total_Visits_std'].fillna(0.0)
    ax2.fill_between(df_daily['Day'], lower_total, upper_total, color=total_color, alpha=0.15)

ax2.set_title('3.2 Total Daily Visits', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)
ax2.set_xlabel('Day')
ax2.set_ylabel('Total Visits')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, framealpha=0.9, facecolor='#fcfcfc', edgecolor='#e2e8f0', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# Global styling enhancements for both axes
for ax in [ax1, ax2]:
    ax.set_facecolor('#fafbfc')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7, color='#cbd5e1')
    ax.grid(False, axis='x')
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(colors='#475569', width=1.0, labelsize=9)
    ax.set_xlabel('Day', fontweight='bold', color='#334155', fontsize=10, labelpad=8)
    ax.yaxis.label.set_color('#334155')
    ax.yaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(10)

    
    # Force integer ticks on X-axis (days)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()

# Save the visualization to outputs folder
output_img = OUTPUTS_ROOT / f"stability_metrics_visualization_{PLOT_MODE}.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_img}")
# plt.show()
