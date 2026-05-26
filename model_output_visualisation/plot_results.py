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
PLOT_MODE = 'average'  

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
# --- 2. Plotting (2x2 Grid, 4 Panels) ---
# =========================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

# --- PANEL 1: Daily Visits by Retail Type ---
for col in retail_types:
    legend_name = col.replace('_', ' ').title()
    line, = ax1.plot(df_daily['Day'], df_daily[f'{col}_mean'], linewidth=2, label=legend_name)
    
    # Shade SD only if we have more than 1 run
    if num_daily_runs > 1:
        color = line.get_color()
        lower = df_daily[f'{col}_mean'] - df_daily[f'{col}_std'].fillna(0.0)
        upper = df_daily[f'{col}_mean'] + df_daily[f'{col}_std'].fillna(0.0)
        ax1.fill_between(df_daily['Day'], lower, upper, color=color, alpha=0.25)

ax1.set_title('Daily Visits by Retail Type', fontweight='bold', loc='left', pad=10)
ax1.set_xlabel('Day')
ax1.set_ylabel('Number of Visits')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.tick_params(width=1.2)
ax1.legend(frameon=True, framealpha=0.8, loc='best', fontsize=8, ncol=2)
ax1.grid(False)

# --- PANEL 2: Total Daily Visits ---
line, = ax2.plot(df_daily['Day'], df_daily['Total_Visits_mean'], linewidth=2, color='#111111', label='Total Visits')

# Shade SD only if we have more than 1 run
if num_daily_runs > 1:
    color = line.get_color()
    lower_total = df_daily['Total_Visits_mean'] - df_daily['Total_Visits_std'].fillna(0.0)
    upper_total = df_daily['Total_Visits_mean'] + df_daily['Total_Visits_std'].fillna(0.0)
    ax2.fill_between(df_daily['Day'], lower_total, upper_total, color=color, alpha=0.25)

ax2.set_title('Total Daily Visits', fontweight='bold', loc='left', pad=10)
ax2.set_xlabel('Day')
ax2.set_ylabel('Total Visits')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.tick_params(width=1.2)
ax2.legend(frameon=True, framealpha=0.8, loc='best', fontsize=9)
ax2.grid(False)

# --- PANEL 3: Utility Convergence (with twin y-axis) ---
if 'JSD_mean' in df_conv.columns:
    # JSD (Left y-axis)
    color_jsd = '#1f77b4'
    line1, = ax3.plot(df_conv['Day'], df_conv['JSD_mean'], linewidth=2, color=color_jsd, label='JSD')
    
    # Shade SD only if we have more than 1 run
    if num_conv_runs > 1:
        lower_jsd = df_conv['JSD_mean'] - df_conv['JSD_std'].fillna(0.0)
        upper_jsd = df_conv['JSD_mean'] + df_conv['JSD_std'].fillna(0.0)
        ax3.fill_between(df_conv['Day'], lower_jsd, upper_jsd, color=color_jsd, alpha=0.25)
    
    ax3.set_ylabel('Jensen-Shannon Divergence (JSD)', color='#111111', fontweight='bold', fontsize=9)
    ax3.tick_params(axis='y', labelcolor='#111111', width=1.2)
    ax3.spines['left'].set_color('#111111')
    ax3.spines['left'].set_linewidth(1.2)

    # KS Distance (Right y-axis sharing same x-axis)
    ax3_twin = ax3.twinx()
    color_ks = '#ff7f0e'
    line2, = ax3_twin.plot(df_conv['Day'], df_conv['KS_Distance_mean'], linewidth=2, color=color_ks, linestyle='--', label='KS Dist')
    
    # Shade SD only if we have more than 1 run
    if num_conv_runs > 1:
        lower_ks = df_conv['KS_Distance_mean'] - df_conv['KS_Distance_std'].fillna(0.0)
        upper_ks = df_conv['KS_Distance_mean'] + df_conv['KS_Distance_std'].fillna(0.0)
        ax3_twin.fill_between(df_conv['Day'], lower_ks, upper_ks, color=color_ks, alpha=0.25)
    
    ax3_twin.set_ylabel('Kolmogorov-Smirnov (KS) Distance', color='#111111', fontweight='bold', fontsize=9)
    ax3_twin.tick_params(axis='y', labelcolor='#111111', width=1.2)
    ax3_twin.spines['right'].set_color('#111111')
    ax3_twin.spines['right'].set_linewidth(1.2)
    ax3_twin.spines['right'].set_visible(True)
    ax3_twin.spines['top'].set_visible(False)
    ax3_twin.spines['left'].set_visible(False)
    ax3_twin.grid(False)

    # Clean up primary axes spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(1.2)
    ax3.tick_params(axis='x', width=1.2)
    ax3.grid(False)
    
    # Combined legend inside
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, frameon=True, framealpha=0.8, loc='best', fontsize=9)
    
    # Force integer labels on X-axis (days)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax3.set_title('Utility Convergence', fontweight='bold', loc='left', pad=10)
ax3.set_xlabel('Day')

# --- PANEL 4: Retail Hierarchy Rank Stability ---
if 'Spearman_Cum_Day_to_Day_mean' in df_conv.columns:
    # Cumulative Day-to-Day
    line_c1, = ax4.plot(df_conv['Day'], df_conv['Spearman_Cum_Day_to_Day_mean'], linewidth=2.5, color='#2ca02c', label='Cumulative Size (D2D)')
    if num_conv_runs > 1:
        color_c1 = line_c1.get_color()
        l_c1 = df_conv['Spearman_Cum_Day_to_Day_mean'] - df_conv['Spearman_Cum_Day_to_Day_std'].fillna(0.0)
        u_c1 = df_conv['Spearman_Cum_Day_to_Day_mean'] + df_conv['Spearman_Cum_Day_to_Day_std'].fillna(0.0)
        ax4.fill_between(df_conv['Day'], l_c1, u_c1, color=color_c1, alpha=0.2)
    
    # Cumulative Anchor to Final
    line_c2, = ax4.plot(df_conv['Day'], df_conv['Spearman_Cum_Final_Anchor_mean'], linewidth=2.5, color='#d62728', linestyle=':', label='Cumulative Size (Final)')
    if num_conv_runs > 1:
        color_c2 = line_c2.get_color()
        l_c2 = df_conv['Spearman_Cum_Final_Anchor_mean'] - df_conv['Spearman_Cum_Final_Anchor_std'].fillna(0.0)
        u_c2 = df_conv['Spearman_Cum_Final_Anchor_mean'] + df_conv['Spearman_Cum_Final_Anchor_std'].fillna(0.0)
        ax4.fill_between(df_conv['Day'], l_c2, u_c2, color=color_c2, alpha=0.2)

    # Daily Day-to-Day
    line_d1, = ax4.plot(df_conv['Day'], df_conv['Spearman_Daily_Day_to_Day_mean'], linewidth=1.5, color='#9467bd', alpha=0.6, label='Daily (D2D)')
    if num_conv_runs > 1:
        color_d1 = line_d1.get_color()
        l_d1 = df_conv['Spearman_Daily_Day_to_Day_mean'] - df_conv['Spearman_Daily_Day_to_Day_std'].fillna(0.0)
        u_d1 = df_conv['Spearman_Daily_Day_to_Day_mean'] + df_conv['Spearman_Daily_Day_to_Day_std'].fillna(0.0)
        ax4.fill_between(df_conv['Day'], l_d1, u_d1, color=color_d1, alpha=0.1)

    # Daily Anchor to Final
    line_d2, = ax4.plot(df_conv['Day'], df_conv['Spearman_Daily_Final_Anchor_mean'], linewidth=1.5, color='#8c564b', alpha=0.6, linestyle=':', label='Daily (Final)')
    if num_conv_runs > 1:
        color_d2 = line_d2.get_color()
        l_d2 = df_conv['Spearman_Daily_Final_Anchor_mean'] - df_conv['Spearman_Daily_Final_Anchor_std'].fillna(0.0)
        u_d2 = df_conv['Spearman_Daily_Final_Anchor_mean'] + df_conv['Spearman_Daily_Final_Anchor_std'].fillna(0.0)
        ax4.fill_between(df_conv['Day'], l_d2, u_d2, color=color_d2, alpha=0.1)

    # Force integer labels on X-axis (days)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax4.set_title('Retail Hierarchy Rank Stability', fontweight='bold', loc='left', pad=10)
ax4.set_xlabel('Day')
ax4.set_ylabel('Spearman Rank Correlation (ρ)')
ax4.set_ylim(-0.1, 1.05)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(1.2)
ax4.spines['bottom'].set_linewidth(1.2)
ax4.tick_params(width=1.2)
ax4.legend(frameon=True, framealpha=0.8, loc='best', fontsize=8, ncol=2)
ax4.grid(False)

plt.tight_layout()

# Save the visualization to outputs folder
output_img = OUTPUTS_ROOT / f"stability_metrics_visualization_{PLOT_MODE}.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_img}")
plt.show()
