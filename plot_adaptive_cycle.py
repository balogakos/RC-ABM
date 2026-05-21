import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

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
    EVAL_FREQ = 10

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

# Generate high-contrast, distinct color palette for the tracked paths
colormap = plt.cm.tab20
colors = [colormap(i) for i in np.linspace(0, 0.95, num_to_select)]

# Plot Panel A: Attractiveness Utility Evolution
for idx, centre in enumerate(top_centres):
    ax1.plot(df['Day'], df[centre], label=f"Centre {centre}", color=colors[idx], linewidth=2.0, alpha=0.85)

# Plot Panel B: Relative Hierarchy Rank Position
for idx, centre in enumerate(top_centres):
    ax2.plot(df['Day'], df_ranks[centre], label=f"Centre {centre}", color=colors[idx], linewidth=2.0, alpha=0.85)

# Invert Y-axis for ranks (Rank 1 at the top)
ax2.invert_yaxis()
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Draw Timing Strikes for Adaptive Cycle Evaluation (every EVAL_FREQ days)
max_day = df['Day'].max()
eval_days = [d for d in range(1, int(max_day) + 1) if d % EVAL_FREQ == 0]

for day in eval_days:
    ax1.axvline(x=day, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.5)
    ax2.axvline(x=day, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.5)

# Create custom dummy legend entry for the evaluation strikes
strike_handle = Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7, label='Evaluation Strike (10-Day Cycle)')

# Setup Panel A styling
ax1.set_title("Panel A: Attractiveness Utility Evolution ($U_t$)", fontweight='bold', fontsize=12, pad=12, loc='left')
ax1.set_xlabel("Simulation Day", fontweight='bold', fontsize=11)
ax1.set_ylabel("Attractiveness Utility", fontweight='bold', fontsize=11)
ax1.set_xlim(df['Day'].min() - 2, max_day + 2)

# Setup Panel B styling
ax2.set_title("Panel B: Relative Hierarchy Rank Position", fontweight='bold', fontsize=12, pad=12, loc='left')
ax2.set_xlabel("Simulation Day", fontweight='bold', fontsize=11)
ax2.set_ylabel("Relative Rank Position (Rank 1 is Top)", fontweight='bold', fontsize=11)
ax2.set_xlim(df['Day'].min() - 2, max_day + 2)

# Global plot adjustments (Spines, Grids)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    ax.grid(False)

# Add legends with the evaluation strike included
handles1, labels1 = ax1.get_legend_handles_labels()
handles1.append(strike_handle)
ax1.legend(handles=handles1, loc='upper right', frameon=True, framealpha=0.9, edgecolor='#e2e8f0', fontsize=9, ncol=2)

handles2, labels2 = ax2.get_legend_handles_labels()
handles2.append(strike_handle)
ax2.legend(handles=handles2, loc='upper right', frameon=True, framealpha=0.9, edgecolor='#e2e8f0', fontsize=9, ncol=2)

# Add informational text box on Panel A explaining the adaptive cycle mechanism
textbox_style = dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95, linewidth=0.8)
info_text = (
    "Adaptive Performance Cycle:\n"
    f"• Evaluation Strikes: Occur every {EVAL_FREQ} days (dashed red lines).\n"
    "• Attractiveness Boosts: Underperforming centres receive a utility\n"
    "  boost (up to +30%) to prevent lock-in and encourage recovery.\n"
    "• Social Diffusion: Social influence is updated at each cycle\n"
    "  to spread consumer preference updates through the social network."
)
ax1.text(0.04, 0.05, info_text, transform=ax1.transAxes, fontsize=9.5, verticalalignment='bottom', bbox=textbox_style, color='#334155')

plt.suptitle("Ecosystem Dynamics: Tracking Retail Centre Evolution & Adaptive Evaluation Cycles", fontweight='bold', fontsize=14, y=0.98)
plt.tight_layout()

# Save final visualization
output_path = OUTPUTS_ROOT / 'adaptive_performance_cycle.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS: Adaptive cycle tracking plot saved to: {output_path}")
