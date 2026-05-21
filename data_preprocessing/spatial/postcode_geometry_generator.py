"""
Postcode Geometry Generator
===========================
Generates a unique list of postcodes from the consumer population file,
looks up their geographic coordinates using the Liverpool Postcode directory,
and saves the result as a Shapefile (.shp).

Created for: Retail ABM Spatial Analysis
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import os

# --- Configuration ---
INPUT_CONSUMER_PATH = Path(r"c:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Consumer_Agents\consumer_agents.parquet")
LOOKUP_POSTCODE_SHP = Path(r"c:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\inputs\Liverpool Postcode\postcode_CA_new.shp")
OUTPUT_SHP_PATH     = Path(r"c:\Users\sgabalog\Documents\P3\Model\data_local\liverpool\processed\agent_postcode_geometries.shp")

def main():
    print(f"Reading unique postcodes from: {INPUT_CONSUMER_PATH.name}...")
    # Read only the Postcode column to save memory
    consumer_df = pd.read_parquet(INPUT_CONSUMER_PATH, columns=['Postcode'])
    unique_postcodes = consumer_df['Postcode'].drop_duplicates().dropna().values
    print(f"Found {len(unique_postcodes)} unique postcodes in agent population.")

    print(f"Loading reference geometries from: {LOOKUP_POSTCODE_SHP.name}...")
    lookup_gdf = gpd.read_file(LOOKUP_POSTCODE_SHP)
    
    # Ensure postcode formats match (strip whitespace and uppercase)
    lookup_gdf['postcode'] = lookup_gdf['postcode'].str.strip().str.upper()
    agent_postcodes_clean = [str(pc).strip().upper() for pc in unique_postcodes]
    
    # Create a DataFrame for the unique agent postcodes
    agent_pc_df = pd.DataFrame({'Postcode': unique_postcodes, 'pc_clean': agent_postcodes_clean})
    
    print("Matching postcodes to coordinates...")
    # Join on the clean postcode column
    matched_gdf = lookup_gdf.merge(
        agent_pc_df, 
        left_on='postcode', 
        right_on='pc_clean', 
        how='inner'
    )
    
    # Drop the temporary clean column and the lookup 'postcode' column
    matched_gdf = matched_gdf.drop(columns=['pc_clean', 'postcode'])
    
    print(f"Match successful: {len(matched_gdf)} out of {len(unique_postcodes)} postcodes matched.")
    
    # Ensure output directory exists
    OUTPUT_SHP_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Shapefile to: {OUTPUT_SHP_PATH}...")
    matched_gdf.to_file(OUTPUT_SHP_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
