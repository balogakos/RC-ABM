import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
import pyarrow as pa
import pyarrow.parquet as pq

def generate_sample_data():
    # Detect directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    SAMPLE_DIR = os.path.join(ROOT_DIR, 'sample_data')
    
    cities = ['liverpool', 'manchester', 'birmingham']
    
    for city in cities:
        city_dir = os.path.join(SAMPLE_DIR, city)
        inputs_dir = os.path.join(city_dir, 'inputs')
        os.makedirs(inputs_dir, exist_ok=True)
        
        print(f"Generating toy sample data for {city} in: {inputs_dir}")
        
        # 1. Boundary Shapefile
        boundary_geom = box(0, 0, 1000, 1000)
        gdf_boundary = gpd.GeoDataFrame({'geometry': [boundary_geom]}, crs='EPSG:27700')
        gdf_boundary.to_file(os.path.join(inputs_dir, "CA boundary.shp"))
        
        # 2. Postcodes
        pcd_points = [Point(100, 100), Point(500, 500), Point(900, 900)]
        gdf_pcd = gpd.GeoDataFrame({
            'postcode': ['A1', 'B2', 'C3'],
            'geometry': pcd_points
        }, crs='EPSG:27700')
        gdf_pcd.to_file(os.path.join(inputs_dir, "postcode_CA_new.shp"))
        
        # 3. Retail Centres
        rc_polys = [box(150, 150, 200, 200), box(800, 800, 850, 850)]
        gdf_rc = gpd.GeoDataFrame({
            'RC_ID': ['101', '102'],
            'Total_POI_': [5, 50],
            'geometry': rc_polys
        }, crs='EPSG:27700')
        gdf_rc.to_file(os.path.join(inputs_dir, "Liverpool_new_retail_centres_01-26(version).shp"))
        
        # 4. Foursquare POIs
        poi_points = [Point(160, 160), Point(170, 170), Point(810, 810)]
        gdf_poi = gpd.GeoDataFrame({
            'category_n': ['Pharmacy', 'Supermarket', 'Cafe'],
            'geometry': poi_points
        }, crs='EPSG:27700')
        gdf_poi.to_file(os.path.join(inputs_dir, "foursquare_data.shp"))
        
        # 5. NTS (Daily trip probabilities sample)
        nts_df = pd.DataFrame({
            'PurposeCount_Comparison': [0, 1, 0, 2],
            'PurposeCount_Entertainment': [1, 0, 0, 0],
            'PurposeCount_Service': [0, 0, 1, 0],
            'PurposeCount_Food/Drink': [0, 1, 0, 1]
        })
        os.makedirs(os.path.join(inputs_dir, "NTS"), exist_ok=True)
        nts_df.to_csv(os.path.join(inputs_dir, "NTS", "Cleaned_NTS_Data.csv"), index=False)
        
        # 6. Consumer Agent Parameters
        os.makedirs(os.path.join(inputs_dir, "Consumer_Agents"), exist_ok=True)
        hh_params = pd.DataFrame({
            'household': ['HH1', 'HH2', 'HH3'],
            'Postcode': ['A1', 'B2', 'C3'],
            'W_Retail': [0.5, 0.8, 0.2],
            'W_Cafe': [0.4, 0.1, 0.9]
        })
        hh_params.to_parquet(os.path.join(inputs_dir, "Consumer_Agents", "bulk_parameterised_consumer_agents_full.parquet"))
        hh_params.to_parquet(os.path.join(inputs_dir, "Consumer_Agents", "con_parameterised_consumer_agents_full.parquet"))
        
        print(f"  - Done with {city}")

    print("\nSample data generation complete. You can point your local_data to this folder to test the pipeline.")

if __name__ == "__main__":
    generate_sample_data()
