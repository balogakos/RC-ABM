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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- PANEL A: Retail centre rank by simulated visits ---
line1, = ax1.plot(ranks, visits_by_rank, linewidth=2, color='#1f77b4', label='Simulated Visits')
ax1.set_title('Retail centre rank by simulated visits', fontweight='bold', loc='left', pad=15)
ax1.set_xlabel('Rank')
ax1.set_ylabel('Simulated Visits')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.tick_params(width=1.2)
ax1.grid(False)
ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))

# --- PANEL B: Cumulative proportion of retail centres ---
line2, = ax2.plot(cum_prop_centres, cum_prop_visits, linewidth=2, color='#2ca02c', label='Model Lorenz Curve')
line3, = ax2.plot([0, 1], [0, 1], linewidth=1.5, color='#7f7f7f', linestyle='--', label='Line of Equality')
ax2.set_title('Cumulative proportion of retail centres', fontweight='bold', loc='left', pad=15)
ax2.set_xlabel('Cumulative proportion of retail centres')
ax2.set_ylabel('Cumulative proportion of total visits')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.tick_params(width=1.2)
ax2.grid(False)
ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))

# Set overall title on figure matching requested title exactly
fig.suptitle('Emergent Retail Heirarchy', fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the visualization to outputs folder
output_img = OUTPUTS_ROOT / "emergent_retail_hierarchy.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Emergent Retail Hierarchy visualization saved to: {output_img}")
plt.show()
