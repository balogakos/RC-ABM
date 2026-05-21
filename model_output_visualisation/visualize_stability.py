import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Find the latest convergence CSV file in the outputs directory
outputs_dir = r"C:\Users\sgabalog\Documents\P3\Model\Retail_ABM\outputs"
csv_files = glob.glob(os.path.join(outputs_dir, "utility_convergence_*.csv"))

if not csv_files:
    print("No utility_convergence_*.csv files found in outputs directory.")
    exit(1)

latest_file = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {latest_file}")

df = pd.read_csv(latest_file)

# Create figure with 2 subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Panel B: Utility Distribution Convergence ---
ax1 = axes[0]
color = '#1f77b4'
line1 = ax1.plot(df['Day'], df['JSD'], marker='o', color=color, linewidth=2, label='Jensen-Shannon Divergence (JSD)')
ax1.set_xlabel('Simulation Day', fontweight='bold')
ax1.set_ylabel('JSD (Velocity of Drift)', color=color, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis for KS distance
ax2 = ax1.twinx()
color = '#ff7f0e'
line2 = ax2.plot(df['Day'], df['KS_Distance'], marker='s', color=color, linewidth=2, linestyle='--', label='KS Distance')
ax2.set_ylabel('Kolmogorov-Smirnov Distance', color=color, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True)
ax1.set_title('Panel B: Utility Distribution Convergence', fontweight='bold', pad=15)

# --- Panel C: Retail Hierarchy Rank Stability ---
ax3 = axes[1]

# Plot Cumulative visit ranks stability
ax3.plot(df['Day'], df['Spearman_Cum_Day_to_Day'], marker='o', color='#2ca02c', linewidth=2.5, label='Cumulative Size (Day-to-Day)')
ax3.plot(df['Day'], df['Spearman_Cum_Final_Anchor'], marker='^', color='#d62728', linewidth=2.5, linestyle=':', label='Cumulative Size (Anchor to Final)')

# Plot Daily visit ranks stability
ax3.plot(df['Day'], df['Spearman_Daily_Day_to_Day'], marker='o', color='#9467bd', linewidth=1.5, alpha=0.6, label='Daily Visits (Day-to-Day)')
ax3.plot(df['Day'], df['Spearman_Daily_Final_Anchor'], marker='v', color='#8c564b', linewidth=1.5, alpha=0.6, linestyle=':', label='Daily Visits (Anchor to Final)')

ax3.set_xlabel('Simulation Day', fontweight='bold')
ax3.set_ylabel('Spearman Rank Correlation (ρ)', fontweight='bold')
ax3.set_ylim(-0.1, 1.05)
ax3.legend(loc='lower right', frameon=True)
ax3.set_title('Panel C: Retail Hierarchy Rank Stability', fontweight='bold', pad=15)

# Adjust layout and show title
plt.suptitle('Ecosystem Structural Dynamics & Hierarchy Stability', fontweight='bold', y=0.98)
plt.tight_layout()

# Save the visualization
output_img = os.path.join(outputs_dir, "stability_metrics_visualization.png")
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved successfully to: {output_img}")
plt.show()
