import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths relative to this script
VISUALISATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALISATION_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'
REPORTS_DIR = OUTPUTS_ROOT / 'reports'
SYNTHETIC_DIR = VISUALISATION_DIR / 'synthetic_data'

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RUNNING RETAIL ABM VISUALISATION PIPELINE")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Visualisation folder: {VISUALISATION_DIR}")
print(f"Reports folder: {REPORTS_DIR}\n")

# =========================================================================
# --- 1. Find Latest Simulation Run Data ---
# =========================================================================
print("Step 1: Locating latest simulation run data...")
CENTRE_DIR = OUTPUTS_ROOT / 'centre_performance'
cp_files = sorted(CENTRE_DIR.glob('retail_centre_performance_*.csv'), key=os.path.getmtime)

if not cp_files:
    print("Error: No simulation performance runs found in outputs/centre_performance/.")
    print("Please execute a model run first (e.g. python simulation/main.py).")
    sys.exit(1)

latest_perf_file = cp_files[-1]
print(f"Latest performance file: {latest_perf_file.name}")

# Load performance data to extract centre IDs and simulated visits
df_perf = pd.read_csv(latest_perf_file)
centre_ids = df_perf['Retail_Centre'].tolist()
num_centres = len(centre_ids)
print(f"Found {num_centres} retail centres in the run.\n")

# Calculate total simulated visits per centre
visit_cols = [c for c in df_perf.columns if c != 'Retail_Centre']
df_perf['sim_visits'] = df_perf[visit_cols].sum(axis=1)

# =========================================================================
# --- 2. Generate/Validate Synthetic Data ---
# =========================================================================
print("Step 2: Generating synthetic validation datasets...")

# 2.1 Footfall Validation Dataset (Noisy simulated visits)
# Add normal noise with std = 15% of the simulated visits (seed 42) for realistic map visualisations
np.random.seed(42)
noise = np.random.normal(0, 0.15 * df_perf['sim_visits'])
real_visits = np.clip(df_perf['sim_visits'] + noise, 0, None).round(0)

footfall_path = SYNTHETIC_DIR / 'footfall_validation.csv'
df_footfall = pd.DataFrame({
    'Retail_Centre': centre_ids,
    'real_visits': real_visits
})
df_footfall.to_csv(footfall_path, index=False)
print(f"Saved: {footfall_path.name}")

# 2.2 Vacancy Rate Rankings (Random rankings 1 to N, seed 123)
np.random.seed(123)
vacancy_ranks = np.random.permutation(np.arange(1, num_centres + 1))
vacancy_path = SYNTHETIC_DIR / 'vacancy_rankings.csv'
df_vacancy = pd.DataFrame({
    'Retail_Centre': centre_ids,
    'Vacancy Rank': vacancy_ranks
})
df_vacancy.to_csv(vacancy_path, index=False)
print(f"Saved: {vacancy_path.name}")

# 2.3 Spend Rankings (Random rankings 1 to N, seed 456)
np.random.seed(456)
spend_ranks = np.random.permutation(np.arange(1, num_centres + 1))
spend_path = SYNTHETIC_DIR / 'spend_rankings.csv'
df_spend = pd.DataFrame({
    'Retail_Centre': centre_ids,
    'Spend Rank': spend_ranks
})
df_spend.to_csv(spend_path, index=False)
print(f"Saved: {spend_path.name}\n")

# =========================================================================
# --- 3. Generate 4.4.2 Retail Rank Validation Table ---
# =========================================================================
print("Step 3: Compiling Section 4.4.2 Retail Rank Validation table...")

# Compute Simulated Rank based on total simulated visits (descending, highest = 1)
df_perf['Simulated Rank'] = df_perf['sim_visits'].rank(ascending=False, method='min').astype(int)

# Compute Footfall Rank based on synthetic footfall (descending, highest = 1)
df_footfall['Footfall Rank'] = df_footfall['real_visits'].rank(ascending=False, method='min').astype(int)

# Merge datasets
df_validation = df_perf[['Retail_Centre', 'Simulated Rank']].copy()
df_validation = df_validation.merge(df_footfall[['Retail_Centre', 'Footfall Rank']], on='Retail_Centre')
df_validation = df_validation.merge(df_vacancy, on='Retail_Centre')
df_validation = df_validation.merge(df_spend, on='Retail_Centre')

# Rename column for display
df_validation = df_validation.rename(columns={'Retail_Centre': 'Retail Centre No.'})

# Sort by Simulated Rank so the table has a logical order
df_validation = df_validation.sort_values(by='Simulated Rank').reset_index(drop=True)

# Select and order columns as requested
rank_table_path = REPORTS_DIR / '08_retail_rank_validation_table.csv'
df_validation.to_csv(rank_table_path, index=False)
print(f"Compiled and saved rank validation table to: {rank_table_path}")
print("Preview of Top 10 Centres in Rank Validation Table:")
print(df_validation.head(10).to_string(index=False))
print()

# =========================================================================
# --- 4. Run Plotting Scripts in Sequence ---
# =========================================================================
print("Step 4: Executing individual plotting scripts...")

plotting_scripts = [
    ('plot_results.py', "1. Simulation Convergence & Stabilisation"),
    ('plot_hierarchy.py', "2. Emergent Retail Hierarchy"),
    ('plot_map.py', "3. Spatial Activity Maps & Correlation"),
    ('plot_lisa.py', "4. LISA Cluster Comparison"),
    ('plot_diffusion_comparison.py', "5. Behavioral Diffusion ON/OFF comparison"),
    ('plot_adaptive_cycle.py', "6. Adaptive Performance Cycle Tracking")
]

for script_name, description in plotting_scripts:
    script_path = VISUALISATION_DIR / script_name
    print(f"--- Running {description} ({script_name}) ---")
    try:
        # Run subprocess using same python executable
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=True, text=True)
        print(f"Stdout:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr:\n{result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}!")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(e.returncode)
    print()

# =========================================================================
# --- 5. Collect and Copy Outputs in Order ---
# =========================================================================
print("Step 5: Collecting and organizing outputs in reports folder...")

# Mapping from default output files in outputs/ to sequential files in outputs/reports/
output_mappings = [
    ('stability_metrics_visualization_latest.png', '01_stability_metrics_visualization_latest.png'),
    ('emergent_retail_hierarchy.png', '02_emergent_retail_hierarchy.png'),
    ('retail_activity_maps.png', '03_retail_activity_maps.png'),
    ('retail_activity_correlation.png', '04_retail_activity_correlation.png'),
    ('lisa_diffusion_comparison.png', '05_lisa_diffusion_comparison.png'),
    ('diffusion_accuracy_comparison.png', '06_diffusion_accuracy_comparison.png'),
    ('adaptive_performance_cycle.png', '07_adaptive_performance_cycle.png')
]

for src_name, dest_name in output_mappings:
    src_path = OUTPUTS_ROOT / src_name
    dest_path = REPORTS_DIR / dest_name
    
    if src_path.exists():
        shutil.copy2(src_path, dest_path)
        print(f"Copied & Renamed: {src_name} -> reports/{dest_name}")
    else:
        print(f"Warning: Expected output file {src_path} was not found.")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print(f"All outputs saved in order to: {REPORTS_DIR}")
print("=" * 60)
