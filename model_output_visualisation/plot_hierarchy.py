import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
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
df_sorted_asc  = df.sort_values(by='Total_Visits', ascending=True).reset_index(drop=True)

# Total number of centres and total visits
num_centres  = len(df)
total_visits = df['Total_Visits'].sum()

# --- Calculations for Panel A ---
ranks          = np.arange(1, num_centres + 1)
visits_by_rank = df_sorted_desc['Total_Visits'].values

# --- Calculations for Panel B (Lorenz Curve) ---
# Cumulative proportion of retail centres
cum_prop_centres = np.arange(1, num_centres + 1) / num_centres
# Cumulative proportion of total visits
cum_prop_visits  = df_sorted_asc['Total_Visits'].cumsum().values / total_visits

# --- Gini coefficient ---
vals = np.sort(df['Total_Visits'].values)
n    = len(vals)
gini = (2 * np.sum((np.arange(1, n + 1) * vals))) / (n * vals.sum()) - (n + 1) / n

# ---------------------------------------------------------------------------
# Figure & axes setup
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.patch.set_facecolor(BG)

# --- Header pattern ---
MAIN_TITLE = 'Emergent Retail Hierarchy'
SUBTITLE   = 'Rank-size distribution and Lorenz concentration curve (ensemble mean, 120 days).'
fig.text(0.04, 0.98, MAIN_TITLE, fontsize=22, fontweight='bold', color=TEXT,    ha='left')
fig.text(0.04, 0.93, SUBTITLE,   fontsize=13,                    color=SUBTEXT, ha='left')
fig.add_artist(lines.Line2D([0.04, 0.96], [0.905, 0.905], color=RULE, linewidth=1.5))

# ---------------------------------------------------------------------------
# PANEL A: Retail centre rank by simulated visits
# ---------------------------------------------------------------------------
ax1.plot(ranks, visits_by_rank, linewidth=2.5, color=NAVY, label='Simulated Visits')
ax1.fill_between(ranks, visits_by_rank, alpha=0.08, color=NAVY)

ax1.set_title('4.1 Retail Centre Rank by Simulated Visits',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)
ax1.set_xlabel('Rank',             fontweight='bold', color=TEXT, fontsize=11, labelpad=8)
ax1.set_ylabel('Simulated Visits', fontweight='bold', color=TEXT, fontsize=11, labelpad=8)
ax1.legend(frameon=True, facecolor=BG, edgecolor=RULE,
           loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

# ---------------------------------------------------------------------------
# PANEL B: Cumulative proportion of retail centres (Lorenz curve)
# ---------------------------------------------------------------------------
ax2.plot(cum_prop_centres, cum_prop_visits,
         linewidth=2.5, color=NAVY, label='Model Lorenz Curve')
ax2.plot([0, 1], [0, 1],
         linewidth=1.5, color=SUBTEXT, linestyle='--', label='Line of Equality')
ax2.fill_between(cum_prop_centres, cum_prop_visits, cum_prop_centres,
                 alpha=0.10, color=ORANGE)
ax2.text(0.05, 0.92, f'Gini = {gini:.3f}',
         transform=ax2.transAxes, fontsize=11, fontweight='bold', color=ORANGE)

ax2.set_title('4.2 Cumulative Proportion of Retail Centres',
              fontweight='bold', fontsize=13, color=TEXT, loc='left', pad=10)
ax2.set_xlabel('Cumulative Proportion of Retail Centres',
               fontweight='bold', color=TEXT, fontsize=11, labelpad=8)
ax2.set_ylabel('Cumulative Proportion of Total Visits',
               fontweight='bold', color=TEXT, fontsize=11, labelpad=8)
ax2.legend(frameon=True, facecolor=BG, edgecolor=RULE,
           loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------
for ax in [ax1, ax2]:
    ax.set_facecolor(BG)

    # Spines: hide top, right, bottom; style left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color(RULE)
    ax.spines['left'].set_linewidth(1)

    # Grid
    ax.grid(True,  axis='y', linestyle=':', alpha=0.6, color=RULE)
    ax.grid(False, axis='x')

    # Ticks
    ax.tick_params(colors=TEXT, labelsize=10)

# ---------------------------------------------------------------------------
# Layout & save
# ---------------------------------------------------------------------------
plt.tight_layout(rect=[0, 0.02, 1, 0.88])

# Save the visualization to outputs folder
output_img = OUTPUTS_ROOT / "emergent_retail_hierarchy.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"Emergent Retail Hierarchy visualization saved to: {output_img}")
# plt.show()
