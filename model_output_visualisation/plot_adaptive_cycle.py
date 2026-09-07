import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.lines import Line2D
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

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

HIGHLIGHT_PALETTE = [
    NAVY, ORANGE, TEAL,
    '#9C3D54', '#7E57C2', '#558B2F',
    '#FF8F00', '#0097A7', '#6D4C41',
    '#455A64', '#AD1457', '#33691E',
]

# =========================================================================
# --- CONFIGURATION & PATHS ---
# =========================================================================
# Root paths for outputs (updated subdirectory structure)
OUTPUTS_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs')
CONV_DIR = OUTPUTS_ROOT / 'utility_convergence'

# Load evaluation frequency from simulation config or default to 10
try:
    sys.path.append(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM')
    from simulation.config import EVAL_FREQ
except ImportError:
    EVAL_FREQ = 30

print(f"Adaptive performance cycle evaluation frequency: {EVAL_FREQ} days")

# =========================================================================
# --- 1. Load Data ---
# =========================================================================
# Find the single latest utility convergence file
c_files = sorted(CONV_DIR.glob('utility_convergence_*.csv'), key=os.path.getmtime)
if not c_files:
    print(f"Error: No utility convergence CSV files found in {CONV_DIR}")
    sys.exit(1)

latest_csv = c_files[-1]
print(f"Loading data from: {latest_csv}")
df = pd.read_csv(latest_csv)

if 'Day' not in df.columns:
    print("Error: CSV does not contain 'Day' column.")
    sys.exit(1)

# Identify centre columns (exclude standard convergence metrics)
non_centre_cols = [
    'Day', 'JSD', 'KS_Distance', 
    'Spearman_Cum_Day_to_Day', 'Spearman_Cum_Final_Anchor',
    'Spearman_Daily_Day_to_Day', 'Spearman_Daily_Final_Anchor'
]
centre_cols = [c for c in df.columns if c not in non_centre_cols]

if not centre_cols:
    print("Error: No retail centre ID columns found in CSV.")
    sys.exit(1)

print(f"Total retail centres tracked in CSV: {len(centre_cols)}")

# =========================================================================
# --- 2. Calculate Ranks and Select Subset ---
# =========================================================================
# Calculate rank position day-by-day (ascending=False: highest utility is Rank 1)
df_ranks = df[centre_cols].rank(axis=1, ascending=False, method='min')

# Select subset of retail centres: Top 12 by final day attractiveness utility
final_day_utilities = df.iloc[-1][centre_cols]
num_to_select = min(12, len(centre_cols))
top_centres = final_day_utilities.sort_values(ascending=False).head(num_to_select).index.tolist()
print(f"Selected top {num_to_select} retail centres on final day: {top_centres}")

# =========================================================================
# --- 3. Plotting ---
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# House-style figure background
fig.patch.set_facecolor(BG)

# --- Figure-level header ---
fig.text(0.04, 0.99, 'Adaptive Performance Cycle',
         fontsize=22, fontweight='bold', color=TEXT, ha='left')
fig.text(0.04, 0.94,
         'Attractiveness utility evolution and relative rank trajectory for top-12 centres over 120 days.',
         fontsize=13, color=SUBTEXT, ha='left')
fig.add_artist(lines.Line2D([0.04, 0.96], [0.91, 0.91], color=RULE, linewidth=1.5))

# Build colour list for top centres (capped to available palette length)
colors = [HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)] for i in range(num_to_select)]

# Plot Panel A: Attractiveness Utility Evolution
# 1. Background paths for all other centres (non-highlighted)
for col in centre_cols:
    if col not in top_centres:
        ax1.plot(df['Day'], df[col], color=RULE, linewidth=0.5, alpha=0.3, zorder=1)

# 2. Highlighted foreground paths for top centres
for idx, centre in enumerate(top_centres):
    ax1.plot(df['Day'], df[centre], label=f"Centre {centre}", color=colors[idx], linewidth=2.0, alpha=0.9, zorder=3)

# Plot Panel B: Relative Hierarchy Rank Position
# 1. Background paths for all other centres (non-highlighted)
for col in centre_cols:
    if col not in top_centres:
        ax2.plot(df['Day'], df_ranks[col], color=RULE, linewidth=0.5, alpha=0.3, zorder=1)

# 2. Highlighted foreground paths for top centres
for idx, centre in enumerate(top_centres):
    ax2.plot(df['Day'], df_ranks[centre], label=f"Centre {centre}", color=colors[idx], linewidth=2.0, alpha=0.9, zorder=3)

# Invert Y-axis for ranks (Rank 1 at the top)
ax2.invert_yaxis()
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Draw Timing Strikes for Adaptive Cycle Evaluation (every EVAL_FREQ days)
max_day = df['Day'].max()
eval_days = [d for d in range(1, int(max_day) + 1) if d % EVAL_FREQ == 0]

for day in eval_days:
    ax1.axvline(x=day, color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.axvline(x=day, color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.6)

# Create custom dummy legend entry for the evaluation strikes
strike_handle = Line2D([0], [0], color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.6,
                       label='Evaluation Strike')

# --- Panel titles ---
ax1.set_title("Attractiveness Utility Evolution ($U_t$)",
              fontweight='bold', color=TEXT, fontsize=13, loc='left', pad=12)
ax1.set_xlabel("Day")
ax1.set_ylabel("Attractiveness Utility")

ax2.set_title("Relative Hierarchy Rank Position",
              fontweight='bold', color=TEXT, fontsize=13, loc='left', pad=12)
ax2.set_xlabel("Day")
ax2.set_ylabel("Relative Rank Position")

# --- Global axis styling ---
for ax in [ax1, ax2]:
    ax.set_facecolor(BG)
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(RULE)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_color(RULE)
    ax.spines['bottom'].set_linewidth(1.0)
    # Grid
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, color=RULE)
    ax.grid(False, axis='x')
    # Ticks
    ax.tick_params(colors=TEXT, width=1.0, labelsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Axis labels
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold', color=TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold', color=TEXT, fontsize=11, labelpad=8)

# --- Legends (include evaluation strike handle) ---
handles1, labels1 = ax1.get_legend_handles_labels()
# Avoid duplicate handles by filtering
if strike_handle.get_label() not in labels1:
    handles1.append(strike_handle)
ax1.legend(handles=handles1, frameon=True, framealpha=0.9,
           facecolor=BG, edgecolor=RULE,
           loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)

handles2, labels2 = ax2.get_legend_handles_labels()
if strike_handle.get_label() not in labels2:
    handles2.append(strike_handle)
ax2.legend(handles=handles2, frameon=True, framealpha=0.9,
           facecolor=BG, edgecolor=RULE,
           loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)

plt.tight_layout(rect=[0, 0.02, 1, 0.89], pad=3.0)

# Save final visualization
output_path = OUTPUTS_ROOT / 'adaptive_performance_cycle.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"\nSUCCESS: Adaptive cycle tracking plot saved to: {output_path}")
