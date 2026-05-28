import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Root paths for outputs
OUTPUTS_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs')
CENTRE_DIR = OUTPUTS_ROOT / 'centre_performance'

# Find the latest retail centre performance file
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError(f"No retail centre performance CSV files found in {CENTRE_DIR}")

latest_file = cp_files[-1]
print(f"Loading latest retail centre performance: {latest_file.name}")

# Load the performance data
df = pd.read_csv(latest_file)

# The first column is 'Retail_Centre' (ID), the rest are visit metrics
visit_cols = [c for c in df.columns if c != 'Retail_Centre']

# Calculate total visits per retail centre
df['Total_Visits'] = df[visit_cols].sum(axis=1)

# Sort centres by total visits
df_sorted_desc = df.sort_values(by='Total_Visits', ascending=False).reset_index(drop=True)
df_sorted_asc = df.sort_values(by='Total_Visits', ascending=True).reset_index(drop=True)

# Total number of centres and total visits
num_centres = len(df)
total_visits = df['Total_Visits'].sum()

# --- Calculations for Panel A ---
ranks = np.arange(1, num_centres + 1)
visits_by_rank = df_sorted_desc['Total_Visits'].values

# --- Calculations for Panel B (Lorenz Curve) ---
# Cumulative proportion of retail centres
cum_prop_centres = np.arange(1, num_centres + 1) / num_centres
# Cumulative proportion of total visits
cum_prop_visits = df_sorted_asc['Total_Visits'].cumsum().values / total_visits

# --- Plotting (1x2 Grid, 2 Panels) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# --- PANEL A: Retail centre rank by simulated visits ---
line1, = ax1.plot(ranks, visits_by_rank, linewidth=2.5, color='#1f4e79', label='Simulated Visits')
ax1.set_title('4.1 Retail Centre Rank by Simulated Visits', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)
ax1.set_xlabel('Rank')
ax1.set_ylabel('Simulated Visits')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, framealpha=0.9, facecolor='#fcfcfc', edgecolor='#e2e8f0', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

# --- PANEL B: Cumulative proportion of retail centres ---
line2, = ax2.plot(cum_prop_centres, cum_prop_visits, linewidth=2.5, color='#2d6a4f', label='Model Lorenz Curve')
line3, = ax2.plot([0, 1], [0, 1], linewidth=1.5, color='#64748b', linestyle='--', label='Line of Equality')
ax2.set_title('4.2 Cumulative Proportion of Retail Centres', fontweight='bold', fontsize=11, color='#1e293b', loc='left', pad=10)
ax2.set_xlabel('Cumulative Proportion of Retail Centres')
ax2.set_ylabel('Cumulative Proportion of Total Visits')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, framealpha=0.9, facecolor='#fcfcfc', edgecolor='#e2e8f0', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

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
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold', color='#334155', fontsize=10, labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold', color='#334155', fontsize=10, labelpad=8)




plt.tight_layout()

# Save the visualization to outputs folder
output_img = OUTPUTS_ROOT / "emergent_retail_hierarchy.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Emergent Retail Hierarchy visualization saved to: {output_img}")
# plt.show()
