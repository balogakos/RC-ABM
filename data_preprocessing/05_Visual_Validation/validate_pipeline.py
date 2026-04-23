import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def validate_pipeline():
    # Detect directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    PLOT_DIR = os.path.join(DATA_DIR, 'plots')
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print(f"Generating validation plots in: {PLOT_DIR}")
    
    # 1. Compare Merged vs Original centres if GPKGs exist
    counts_path = os.path.join(PROCESSED_DIR, 'retail_centre_type_counts.gpkg')
    merged_path = os.path.join(PROCESSED_DIR, 'retail_centre_type_counts_merged.gpkg')
    
    if os.path.exists(counts_path):
        try:
            gdf_orig = gpd.read_file(counts_path)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_orig.plot(ax=ax, color='blue', alpha=0.5, label='Original Centres')
            
            if os.path.exists(merged_path):
                gdf_merged = gpd.read_file(merged_path)
                gdf_merged.boundary.plot(ax=ax, color='red', linewidth=2, label='Merged Boundary')
            
            plt.title("Retail Centre Spatial Validation (Original vs Merged)")
            plt.savefig(os.path.join(PLOT_DIR, "retail_centre_spatial_validation.png"), dpi=300)
            plt.close()
            print("  - Saved retail_centre_spatial_validation.png")
        except Exception as e:
            print(f"  - Skipped spatial plot: {e}")

    # 2. Utility Score Distribution
    utility_path = os.path.join(PROCESSED_DIR, 'utility_scores_average.parquet')
    if os.path.exists(utility_path):
        try:
            df = pd.read_parquet(utility_path)
            # Find utility columns (numeric)
            non_meta = [c for c in df.columns if c not in ['household', 'id', 'Postcode', 'age_years', 'sex']]
            # Sample some RC columns
            rc_cols = [c for c in non_meta if '_' not in c][:10] # first 10 centre IDs
            
            if rc_cols:
                plt.figure(figsize=(10, 6))
                for col in rc_cols:
                    # Filter out 0s for visibility
                    vals = df[df[col] > 0][col]
                    if not vals.empty:
                        sns.kdeplot(vals, label=f"Centre {col}", alpha=0.6)
                
                plt.title("Utility Score Distribution (Normalized Scores > 0)")
                plt.xlabel("Utility Value (0 to 1)")
                plt.ylabel("Density")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, "utility_distribution_validation.png"), dpi=300)
                plt.close()
                print("  - Saved utility_distribution_validation.png")
        except Exception as e:
            print(f"  - Skipped utility plot: {e}")

    # 3. Transport Time Correlation (Drive vs Walk)
    transport_path = os.path.join(PROCESSED_DIR, 'final_transport_times.parquet')
    if os.path.exists(transport_path):
        try:
            # This is complex because columns contain dicts. 
            # We will sample one postcode and its RC distances.
            df_trans = pd.read_parquet(transport_path).head(1)
            if not df_trans.empty:
                import ast
                def safe_eval(x): return x if isinstance(x, dict) else ast.literal_eval(x) if isinstance(x, str) else {}
                
                walk = safe_eval(df_trans['Walk'].iloc[0])
                drive = safe_eval(df_trans['Drive'].iloc[0])
                
                common_rcs = set(walk.keys()) & set(drive.keys())
                if common_rcs:
                    w_vals = [walk[rc] for rc in common_rcs]
                    d_vals = [drive[rc] for rc in common_rcs]
                    
                    plt.figure(figsize=(8, 8))
                    plt.scatter(d_vals, w_vals, alpha=0.5)
                    plt.plot([0, max(d_vals)], [0, max(d_vals)], 'r--', alpha=0.3)
                    plt.title("Transport Time Sanity Check (Drive vs Walk Mins)")
                    plt.xlabel("Drive Time (mins)")
                    plt.ylabel("Walk Time (mins)")
                    plt.savefig(os.path.join(PLOT_DIR, "transport_time_correlation.png"), dpi=300)
                    plt.close()
                    print("  - Saved transport_time_correlation.png")
        except Exception as e:
            print(f"  - Skipped transport plot: {e}")

    print("\nValidation complete. Check the 'data_local/liverpool/plots/' folder.")

if __name__ == "__main__":
    validate_pipeline()
