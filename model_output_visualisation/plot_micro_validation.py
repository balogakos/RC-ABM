import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pathlib import Path

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
# --- USER CONFIGURATION (ZOOM-IN RC_IDs) ---
# =========================================================================
# You can customize these RC_IDs at the top to target specific case-studies
REGIONAL_CENTRE = '1728'   # Large dominant centre
TOWN_CENTRE     = '477'    # Medium town centre
LOCAL_CENTRE    = '67'     # Small convenience/local centre

# Geographically close competing centres to evaluate zero-sum dynamics
COMPETITOR_A    = '90'
COMPETITOR_B    = '1728'

# Paths
VISUALISATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALISATION_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
SYNTHETIC_DIR = VISUALISATION_DIR / 'synthetic_data'

# =========================================================================
# --- 1. Load Data ---
# =========================================================================
print("Loading datasets for Micro-Level Validation...")

# Find latest centre performance file
CENTRE_DIR = OUTPUTS_ROOT / 'centre_performance'
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)
if not cp_files:
    raise FileNotFoundError("No centre performance files found. Run the simulation first.")
latest_perf_file = cp_files[-1]

df_perf = pd.read_csv(latest_perf_file)
visit_cols = [c for c in df_perf.columns if c != 'Retail_Centre']
df_perf['sim_visits'] = df_perf[visit_cols].sum(axis=1)

# Clean RC_ID
def clean_id(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s
df_perf['Retail_Centre'] = df_perf['Retail_Centre'].apply(clean_id)

# Load synthetic validation files
df_footfall = pd.read_csv(SYNTHETIC_DIR / 'footfall_validation.csv')
df_footfall['Retail_Centre'] = df_footfall['Retail_Centre'].apply(clean_id)

df_vacancy = pd.read_csv(SYNTHETIC_DIR / 'vacancy_rankings.csv')
df_vacancy['Retail_Centre'] = df_vacancy['Retail_Centre'].apply(clean_id)

df_spend = pd.read_csv(SYNTHETIC_DIR / 'spend_rankings.csv')
df_spend['Retail_Centre'] = df_spend['Retail_Centre'].apply(clean_id)

# Ranks (1 = best/highest)
df_perf['Simulated Rank'] = df_perf['sim_visits'].rank(ascending=False, method='min').astype(int)
df_footfall['Footfall Rank'] = df_footfall['real_visits'].rank(ascending=False, method='min').astype(int)

# Merge everything
df_val = df_perf[['Retail_Centre', 'sim_visits', 'Simulated Rank']].copy()
df_val = df_val.merge(df_footfall[['Retail_Centre', 'real_visits', 'Footfall Rank']], on='Retail_Centre')
df_val = df_val.merge(df_vacancy, on='Retail_Centre')
df_val = df_val.merge(df_spend, on='Retail_Centre')

# Rename for cleanliness
df_val = df_val.rename(columns={
    'Vacancy Rank': 'Real Vacancy Rank',
    'Spend Rank': 'Real Spend Rank',
    'Footfall Rank': 'Real Footfall Rank'
})

print("Merged validation dataset sample:")
print(df_val.head(5))

# =========================================================================
# --- 2. Plotting ---
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Apply house-style background to figure and both axes
fig.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

# --- Figure-level header ---
fig.text(0.04, 0.98,
         'Centre-Level Validation — Rank Alignment',
         fontsize=22, fontweight='bold', color=TEXT, ha='left')
fig.text(0.04, 0.93,
         'Simulated footfall rank vs. observed footfall, spend, and vacancy ranks across case-study centres.',
         fontsize=13, color=SUBTEXT, ha='left')
fig.add_artist(lines.Line2D([0.04, 0.96], [0.90, 0.90], color=RULE, linewidth=1.5))

# --- PLOT 1: Local Alignment (Grouped Rank Bar Chart) ---
case_studies = [
    ('Regional Centre (ID: ' + REGIONAL_CENTRE + ')', REGIONAL_CENTRE),
    ('Town Centre (ID: ' + TOWN_CENTRE + ')', TOWN_CENTRE),
    ('Local Centre (ID: ' + LOCAL_CENTRE + ')', LOCAL_CENTRE)
]

labels = [item[0] for item in case_studies]
ids = [item[1] for item in case_studies]

df_cases = df_val[df_val['Retail_Centre'].isin(ids)].set_index('Retail_Centre').reindex(ids)

x = np.arange(len(labels))
width = 0.2

# Get ranks for the cases
sim_ranks = df_cases['Simulated Rank'].values
real_ranks = df_cases['Real Footfall Rank'].values
spend_ranks = df_cases['Real Spend Rank'].values
vac_ranks = df_cases['Real Vacancy Rank'].values

# Ranks are plotted: lower value is better, so we will invert Y-axis
rects1 = ax1.bar(x - 1.5*width, sim_ranks,   width, label='Simulated Footfall Rank', color=NAVY)
rects2 = ax1.bar(x - 0.5*width, real_ranks,  width, label='Real Footfall Rank',      color=ORANGE)
rects3 = ax1.bar(x + 0.5*width, spend_ranks, width, label='Real Spend Rank',         color=TEAL)
rects4 = ax1.bar(x + 1.5*width, vac_ranks,   width, label='Real Vacancy Rank',       color='#9C3D54')

ax1.set_title('5.1 Local Performance Rank Alignment (Lower = Better)',
              fontweight='bold', color=TEXT, fontsize=13, loc='left', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Rank (out of ' + str(len(df_val)) + ' centres)',
               fontweight='bold', color=TEXT, fontsize=11)

# Spines: hide top/right/bottom; left spine = RULE, lw=1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_color(RULE)
ax1.spines['left'].set_linewidth(1)

ax1.tick_params(color=TEXT, labelsize=10)
ax1.grid(True, axis='y', linestyle=':', alpha=0.6, color=RULE)
ax1.grid(False, axis='x')

# Invert Y-axis so rank 1 is at the top!
ax1.invert_yaxis()
# Add a margin to Y limits to accommodate the labels at the top (which is rank 1)
ymin, ymax = ax1.get_ylim()
ax1.set_ylim(ymin + 15, -5)

ax1.legend(frameon=True, framealpha=0.9, facecolor=BG, edgecolor=RULE, loc='lower left', fontsize=9)

# Add values above bars
def autolabel(rects, axis):
    for rect in rects:
        height = rect.get_height()
        # Since y-axis is inverted, placement is adjusted
        axis.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if axis.yaxis_inverted() else -12),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,
                    color=TEXT, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)
autolabel(rects4, ax1)


# --- PLOT 2: Competitor Pair-Wise Dynamics (Ratios) ---
# Retrieve visits for competitor pair
row_a = df_val[df_val['Retail_Centre'] == COMPETITOR_A].iloc[0]
row_b = df_val[df_val['Retail_Centre'] == COMPETITOR_B].iloc[0]

sim_ratio = row_a['sim_visits'] / row_b['sim_visits']
real_ratio = row_a['real_visits'] / row_b['real_visits']

# For spend rank ratio, we can use the inverse rank as a proxy for value ratio
# Spend Rank: lower is better, so inverse of rank represents relative value
spend_ratio = (1.0 / row_a['Real Spend Rank']) / (1.0 / row_b['Real Spend Rank'])

ratio_labels = ['Simulated Footfall Ratio', 'Real Footfall Ratio', 'Spend Value Proxy Ratio']
ratios = [sim_ratio, real_ratio, spend_ratio]
bar_colors = [NAVY, ORANGE, TEAL]

rects = ax2.bar(ratio_labels, ratios, color=bar_colors, width=0.5)

ax2.set_title(f'5.2 Competitor Choice Ratio ({COMPETITOR_A} vs {COMPETITOR_B})',
              fontweight='bold', color=TEXT, fontsize=13, loc='left', pad=10)
ax2.set_ylabel('Ratio Value (A / B)', fontweight='bold', color=TEXT, fontsize=11)

# Spines: hide top/right/bottom; left spine = RULE, lw=1
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_color(RULE)
ax2.spines['left'].set_linewidth(1)

ax2.tick_params(color=TEXT, labelsize=10)
ax2.grid(True, axis='y', linestyle=':', alpha=0.6, color=RULE)
ax2.grid(False, axis='x')

# Annotate ratio bars
for rect in rects:
    height = rect.get_height()
    ax2.annotate(f'{height:.2f}x',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10,
                color=TEXT, fontweight='bold')

plt.tight_layout(rect=[0, 0.02, 1, 0.88], pad=3.0)

# Save image
output_path = OUTPUTS_ROOT / 'micro_validation_metrics.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG)
print(f"Micro-validation plot saved to: {output_path}")
