from pathlib import Path
import os

# Base Directories
# This file is in simulation/core/paths.py
CORE_DIR = Path(__file__).resolve().parent
SIM_DIR = CORE_DIR.parent
PROJECT_ROOT = SIM_DIR.parent
MODEL_DIR = PROJECT_ROOT.parent

# Data Directories (Following the structure in config.py)
DATA_LOCAL = MODEL_DIR / "data_local" / "liverpool"
PROCESSED_DIR = DATA_LOCAL / "processed"

# Inputs
UTILITY_DIR = PROCESSED_DIR
RETAIL_CENTRES_GPKG = PROCESSED_DIR / "retail_centre_type_counts.gpkg"
TRANSPORT_TIMES_PATH = PROCESSED_DIR / "final_transport_times.parquet"

# Output Directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def ensure_dirs():
    """Ensures required directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Model Dir: {MODEL_DIR}")
    print(f"Utility Dir: {UTILITY_DIR}")
    ensure_dirs()
