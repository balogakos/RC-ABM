import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
OUTPUTS_ROOT = Path(r'C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs')
CONV_DIR = OUTPUTS_ROOT / 'utility_convergence'
REPORTS_DIR = OUTPUTS_ROOT / 'reports'  # Fallback reports folder

# Find the active reports folder (latest reports_* directory)
dirs = sorted(OUTPUTS_ROOT.glob('reports_*'), key=os.path.getmtime)
if dirs:
    REPORTS_DIR = dirs[-1]

print(f"Targeting reports folder for saving plot: {REPORTS_DIR}")

# 1. Load utility convergence data
c_files = sorted(CONV_DIR.glob('utility_convergence_*.csv'), key=os.path.getmtime)
if not c_files:
    print("Error: No utility convergence data found.")
    sys.exit(1)

latest_csv = c_files[-1]
df = pd.read_csv(latest_csv)

# Exclude non-centre columns
non_centre_cols = [
    'Day', 'JSD', 'KS_Distance', 
    'Spearman_Cum_Day_to_Day', 'Spearman_Cum_Final_Anchor',
    'Spearman_Daily_Day_to_Day', 'Spearman_Daily_Final_Anchor'
]
centre_cols = [c for c in df.columns if c not in non_centre_cols]

# 2. Identify top socially boosted vs. decaying centres
first_day_u = df.iloc[0][centre_cols]
last_day_u = df.iloc[-1][centre_cols]
utility_change = last_day_u - first_day_u

top_boosted = utility_change.sort_values(ascending=False).head(5).index.tolist()
top_decayed = utility_change.sort_values(ascending=True).head(5).index.tolist()

# 3. Detect policy-intervened centres (jumps of >= 4% on evaluation boundaries)
eval_boundaries = [31, 61, 91, 121, 151]
intervened_centres = set()
for day in eval_boundaries:
    if day in df['Day'].values:
        row_prev = df[df['Day'] == day - 1].iloc[0]
        row_curr = df[df['Day'] == day].iloc[0]
        for c in centre_cols:
            u_prev = row_prev[c]
            u_curr = row_curr[c]
            if u_prev > 0 and (u_curr / u_prev) >= 1.04:
                intervened_centres.add(c)

intervened_centres = list(intervened_centres)
print(f"Detected policy intervened centres: {intervened_centres}")

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Social Influence (S-Curve & Decay dynamics)
colormap_boost = plt.cm.Greens
colormap_decay = plt.cm.Reds

# Global average path for reference
global_avg_path = df[centre_cols].mean(axis=1)
ax1.plot(df['Day'], global_avg_path, color='#64748b', linewidth=2.0, linestyle='--', label='Global Average Utility', zorder=2)

# Plot top boosted (Greens)
for idx, c in enumerate(top_boosted):
    color = colormap_boost(0.5 + 0.1 * idx)
    ax1.plot(df['Day'], df[c], color=color, linewidth=2.0, alpha=0.9, label=f'Boosted Centre {c}', zorder=3)

# Plot top decayed (Reds)
for idx, c in enumerate(top_decayed):
    color = colormap_decay(0.5 + 0.1 * idx)
    ax1.plot(df['Day'], df[c], color=color, linewidth=2.0, alpha=0.8, label=f'Decaying Centre {c}', zorder=3)

ax1.set_title("A. Social Influence: Growth vs. Decay Trajectories", fontweight='bold', loc='left', pad=12)
ax1.set_xlabel("Day")
ax1.set_ylabel("Attractiveness Utility ($U_t$)")
ax1.legend(frameon=True, framealpha=0.9, fontsize=8, ncol=2, loc='upper left')

# Panel B: Policy Interventions (Welfare Boosts)
ax2.plot(df['Day'], global_avg_path, color='#64748b', linewidth=2.0, linestyle='--', label='Global Average Utility', zorder=2)

if intervened_centres:
    colormap_interv = plt.cm.Purples
    for idx, c in enumerate(intervened_centres[:6]):  # Plot top 6 max
        color = colormap_interv(0.5 + 0.08 * idx)
        ax2.plot(df['Day'], df[c], color=color, linewidth=2.2, alpha=0.9, label=f'Intervened Centre {c}', zorder=3)
    ax2.set_title("B. Policy Interventions: Attractiveness Boosts", fontweight='bold', loc='left', pad=12)
else:
    # Fallback if no interventions were triggered (or if it's a short test run)
    # We display the next set of highly active centres to illustrate stability
    active_centres = last_day_u.sort_values(ascending=False).iloc[5:10].index.tolist()
    colormap_active = plt.cm.Purples
    for idx, c in enumerate(active_centres):
        color = colormap_active(0.5 + 0.1 * idx)
        ax2.plot(df['Day'], df[c], color=color, linewidth=1.8, alpha=0.8, label=f'Active Centre {c}', zorder=3)
    ax2.set_title("B. Active Centre Attractiveness Stability (No Jumps)", fontweight='bold', loc='left', pad=12)

# Draw evaluation strike vertical lines
for day in [30, 60, 90, 120, 150]:
    ax2.axvline(x=day, color='#e74c3c', linestyle=':', linewidth=1.0, alpha=0.4, label='Policy Eval strike' if day == 30 else "")

ax2.set_xlabel("Day")
ax2.set_ylabel("Attractiveness Utility ($U_t$)")
ax2.legend(frameon=True, framealpha=0.9, fontsize=8, ncol=2, loc='upper left')

# Style cleanup
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2)
    ax.grid(False)

plt.tight_layout()

# Save figure
output_img = REPORTS_DIR / "09_boosted_centres_trajectories.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Focus plot on affected boosted centres saved to: {output_img}")
