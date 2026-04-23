"""
Script to:
1. Read in Foursquare POI data, CA boundary, and Retail Centre polygons.
2. Clip Foursquare data to within the CA boundary.
3. Spatially join clipped Foursquare points to Retail Centre polygons.
4. Classify each store into a broad retail type.
5. Count the number of each retail type per retail centre.
6. Save the result as a GeoPackage (.gpkg).
"""

import geopandas as gpd
import pandas as pd

import os

# ──────────────────────────────────────────────
# 0. Configuration & Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(MODEL_ROOT, 'data_local', 'liverpool')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Read in the three shapefiles
# ──────────────────────────────────────────────
print("Loading shapefiles...")

foursquare = gpd.read_file(
    os.path.join(INPUT_DIR, "foursquare_data.shp")
)
boundary = gpd.read_file(
    os.path.join(INPUT_DIR, "CA boundary.shp")
)
retail_centres = gpd.read_file(
    os.path.join(INPUT_DIR, "Liverpool_new_retail_centres_01-26(version).shp")
)

print(f"Foursquare records:     {len(foursquare)}")
print(f"Boundary polygons:      {len(boundary)}")
print(f"Retail centre polygons: {len(retail_centres)}")

print(f"\nFoursquare columns: {list(foursquare.columns)}")
print(f"Retail centre columns: {list(retail_centres.columns)}")

# ──────────────────────────────────────────────
# CONFIGURATION — update these if column names
# differ from what is auto-detected
# ──────────────────────────────────────────────
# Column in the Foursquare data containing the detailed category name
FOURSQUARE_CATEGORY_COL = "category_n"  # <-- UPDATE if different
# NOTE: If this column is missing, the script will print available columns below.

# Column in the retail centres data containing the centre name/ID
RETAIL_CENTRE_NAME_COL = "RC_ID"  # <-- UPDATE if different

# ──────────────────────────────────────────────
# 2. Ensure all layers share the same CRS
# ──────────────────────────────────────────────
if foursquare.crs != boundary.crs:
    print(f"Reprojecting Foursquare data from {foursquare.crs} to {boundary.crs}")
    foursquare = foursquare.to_crs(boundary.crs)

if retail_centres.crs != boundary.crs:
    print(f"Reprojecting Retail Centres from {retail_centres.crs} to {boundary.crs}")
    retail_centres = retail_centres.to_crs(boundary.crs)

# ──────────────────────────────────────────────
# Check for required columns early
# ──────────────────────────────────────────────
if FOURSQUARE_CATEGORY_COL not in foursquare.columns:
    print(f"\n❌ ERROR: Column '{FOURSQUARE_CATEGORY_COL}' not found in Foursquare data.")
    print(f"Available columns: {list(foursquare.columns)}")
    print("Please update 'FOURSQUARE_CATEGORY_COL' in the script to match one of these.")
    exit(1)

if RETAIL_CENTRE_NAME_COL not in retail_centres.columns:
    print(f"\n❌ ERROR: Column '{RETAIL_CENTRE_NAME_COL}' not found in Retail Centres data.")
    print(f"Available columns: {list(retail_centres.columns)}")
    print("Please update 'RETAIL_CENTRE_NAME_COL' in the script to match one of these.")
    exit(1)

# ──────────────────────────────────────────────
# 3. Clip Foursquare data to within the boundary
# ──────────────────────────────────────────────
print("\nClipping Foursquare data to within the CA boundary...")
foursquare_clipped = gpd.clip(foursquare, boundary)
print(f"Foursquare records after clipping: {len(foursquare_clipped)}")

# ──────────────────────────────────────────────
# 4. Spatial join — assign a retail centre to
#    each Foursquare point that falls within one
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 4. Spatial join — assign a retail centre to
#    each Foursquare point that falls within one
# ──────────────────────────────────────────────
print("\nPerforming spatial join with Retail Centres...")
# NOTE: GeoPandas sjoin suffixes are appended as is.
# If lsuffix='_fsq' and rsuffix='_rc', a column 'foo' becomes 'foo_fsq' and 'foo_rc'.
# However, user output showed 'category_n__fsq' which suggests double underscores.
# This often happens if the original column name ends with an underscore? No.
# Or if lsuffix was set to '__fsq'? 
# Let's inspect potential columns more flexibly.
foursquare_with_rc = gpd.sjoin(
    foursquare_clipped,
    retail_centres,
    how="left",
    predicate="within",
    lsuffix="_fsq",
    rsuffix="_rc"
)

# Debug column names after join
print("Columns after join:", foursquare_with_rc.columns.tolist())

# Identify rows that joined successfully
# We need to find the column that indicates a successful join (from the right df).
# Usually 'index_right' (gpd default) unless overridden or suffixed.
# If rsuffix is provided, 'index_right' -> 'index_right_rc' (or similar).
join_col = None

# Candidates for the join indicator column:
candidates = [
    "index_right",
    "index_right_rc",
    "index_rc",   # if index_right was renamed
    RETAIL_CENTRE_NAME_COL,
    f"{RETAIL_CENTRE_NAME_COL}_rc",
    f"{RETAIL_CENTRE_NAME_COL}__rc" # checking double underscore just in case
]

for cand in candidates:
    if cand in foursquare_with_rc.columns:
        join_col = cand
        print(f"Using '{join_col}' to check join status.")
        break

if join_col is None:
    # Fallback: look for ANY column ending in '_rc' or '__rc'
    rc_cols = [c for c in foursquare_with_rc.columns if c.endswith("_rc")]
    if rc_cols:
        join_col = rc_cols[0]
        print(f"Using '{join_col}' (fallback) to check join status.")
    else:
        print("❌ ERROR: spatial join seems to have failed to attach columns.")
        print('Columns:', foursquare_with_rc.columns.tolist())
        exit(1)

assigned = foursquare_with_rc[join_col].notna().sum()
unassigned = foursquare_with_rc[join_col].isna().sum() 
print(f"Stores assigned to a retail centre: {assigned}")
print(f"Stores NOT within any retail centre: {unassigned}")

# ──────────────────────────────────────────────
# 5. Classification mapping
# ──────────────────────────────────────────────

# ... (mapping dict remains same) ...

# ──────────────────────────────────────────────
# 6. Apply classification
# ──────────────────────────────────────────────
print("\nClassifying Foursquare data into broad retail types...")

mapping = {
    'Pharmacy': 'Convenience Store',
    'Drugstore': 'Convenience Store',
    'Cosmetics Store': 'Retail',
    'Fish Market': 'Convenience Store',
    'Frozen Yogurt Shop': 'Cafe',
    'Medical Supply Store': 'Convenience Store',
    'Food and Beverage Retail': 'Foodstore',
    'Martial Arts Dojo': 'Active',
    'Gym and Studio': 'Active',
    'Gym': 'Active',
    'Water Sports': 'Active',
    'Café': 'Cafe',
    'Cupcake Shop': 'Cafe',
    'Coffee Shop': 'Cafe',
    'Donut Shop': 'Cafe',
    'Breakfast Spot': 'Cafe',
    'Juice Bar': 'Cafe',
    'Tea Room': 'Cafe',
    'Gelato Shop': 'Cafe',
    'Ice Cream Parlor': 'Cafe',
    'Bubble Tea Shop': 'Cafe',
    'Liquor Store': 'Convenience Store',
    'Convenience Store': 'Convenience Store',
    'Beer Store': 'Convenience Store',
    'Butcher': 'Convenience Store',
    'Cheese Store': 'Convenience Store',
    'Fuel Station': 'Convenience Store',
    'Fruit and Vegetable Store': 'Convenience Store',
    'Imported Food Store': 'Convenience Store',
    'Newsstand': 'Convenience Store',
    'Market': 'Convenience Store',
    'Bar': 'Entertainment',
    'Pub': 'Entertainment',
    'Beer Bar': 'Entertainment',
    'Wine Bar': 'Entertainment',
    'Arts and Entertainment': 'Entertainment',
    'Indoor Play Area': 'Entertainment',
    'Speakeasy': 'Entertainment',
    'Cocktail Bar': 'Entertainment',
    'Social Club': 'Entertainment',
    'Mini Golf Course': 'Entertainment',
    'Casino': 'Entertainment',
    'Sports Bar': 'Entertainment',
    'Theater': 'Entertainment',
    'Movie Theater': 'Entertainment',
    'Rock Club': 'Entertainment',
    'Dive Bar': 'Entertainment',
    'Arcade': 'Entertainment',
    'Night Club': 'Entertainment',
    'Discount Store': 'Foodstore',
    'Health Food Store': 'Foodstore',
    'Supermarket': 'Foodstore',
    'Grocery Store': 'Foodstore',
    'Gourmet Store': 'Foodstore',
    'Warehouse or Wholesale Store': 'Foodstore',
    'Betting Shop': 'Personal Service',
    'Hair Salon': 'Personal Service',
    'Dentist': 'Personal Service',
    'Pet Service': 'Personal Service',
    'Tattoo Parlor': 'Personal Service',
    'Health and Beauty Service': 'Personal Service',
    'Pet Grooming Service': 'Personal Service',
    'Locksmith': 'Personal Service',
    'Computer Repair Service': 'Professional Services',
    'Body Piercing Shop': 'Personal Service',
    'Barbershop': 'Personal Service',
    'Laundromat': 'Personal Service',
    'Nail Salon': 'Personal Service',
    'Massage Clinic': 'Personal Service',
    'Dry Cleaner': 'Personal Service',
    'Psychic and Astrologer': 'Personal Service',
    'Skin Care Clinic': 'Personal Service',
    'Pilates Studio': 'Personal Service',
    'Photography Studio': 'Professional Services',
    'Real Estate Agency': 'Professional Services',
    'Law Office': 'Professional Services',
    'Bank': 'Professional Services',
    'Library': 'Professional Services',
    'Tourist Information and Service': 'Professional Services',
    'Travel Agency': 'Professional Services',
    'Physical Therapy Clinic': 'Personal Service',
    'Photographer': 'Professional Services',
    'Community Center': 'Professional Services',
    'Loans Agency': 'Professional Services',
    'Veterinarian': 'Professional Services',
    'Funeral Home': 'Professional Services',
    'Pet Sitting and Boarding Service': 'Professional Services',
    'Tour Provider': 'Professional Services',
    'Photography Service': 'Professional Services',
    'Photography Lab': 'Professional Services',
    'Retail': 'Retail',
    'Department Store': 'Retail',
    'Clothing Store': 'Retail',
    'Electronics Store': 'Retail',
    'Record Store': 'Retail',
    'Vintage and Thrift Store': 'Retail',
    'Toy Store': 'Retail',
    'Arts and Crafts Store': 'Retail',
    'Furniture and Home Store': 'Retail',
    'Music Store': 'Retail',
    'Video Games Store': 'Retail',
    "Children's Clothing Store": 'Retail',
    'Bookstore': 'Retail',
    'Shoe Store': 'Retail',
    "Men's Store": 'Retail',
    'Print Store': 'Retail',
    'Antique Store': 'Retail',
    'Hobby Store': 'Retail',
    'Vape Store': 'Retail',
    'Pawn Shop': 'Retail',
    'Bridal Store': 'Retail',
    'Gift Store': 'Retail',
    'Jewelry Store': 'Retail',
    'Shopping Mall': 'Retail',
    'Home Appliance Store': 'Retail',
    'Hardware Store': 'Retail',
    'Flower Store': 'Retail',
    'Newsagent': 'Convenience Store',
    'Miscellaneous Store': 'Retail',
    'Pet Supplies Store': 'Retail',
    'Housewares Store': 'Retail',
    'Shopping Plaza': 'Retail',
    'Car Dealership': 'Retail',
    'Motorcycle Dealership': 'Retail',
    'Used Car Dealership': 'Retail',
    'Computers and Electronics Retail': 'Retail',
    'Packaging Supply Store': 'Retail',
    'Camera Store': 'Retail',
    'Herbs and Spices Store': 'Retail',
    'Running Store': 'Retail',
    'Comic Book Store': 'Retail',
    'Kitchen Supply Store': 'Retail',
    'Textiles Store': 'Retail',
    'Baby Store': 'Retail',
    'Candy Store': 'Retail',
    'Bicycle Store': 'Retail',
    'Dessert Shop': 'Cafe',
    'Mobility Store': 'Retail',
    'Costume Store': 'Retail',
    'Watch Store': 'Retail',
    'Mattress Store': 'Retail',
    'RV and Motorhome Dealership': 'Retail',
    'Vegan and Vegetarian Restaurant': 'Restaurant',
    'Halal Restaurant': 'Restaurant',
    'Chinese Restaurant': 'Restaurant',
    'Fast Food Restaurant': 'Restaurant',
    'Italian Restaurant': 'Restaurant',
    'Pizzeria': 'Restaurant',
    'American Restaurant': 'Restaurant',
    'Indian Restaurant': 'Restaurant',
    'Bistro': 'Restaurant',
    'Turkish Restaurant': 'Restaurant',
    'Noodle Restaurant': 'Restaurant',
    'French Restaurant': 'Restaurant',
    'Diner': 'Restaurant',
    'Restaurant': 'Restaurant',
    'Tapas Restaurant': 'Restaurant',
    'English Restaurant': 'Restaurant',
    'Korean Restaurant': 'Restaurant',
    'Greek Restaurant': 'Restaurant',
    'Wings Joint': 'Restaurant',
    'Indonesian Restaurant': 'Restaurant',
    'Taco Restaurant': 'Restaurant',
    'Fish and Chips Shop': 'Restaurant',
    'Burger Joint': 'Restaurant',
    'Dining and Drinking': 'Restaurant',
    'Steakhouse': 'Restaurant',
    'Asian Restaurant': 'Restaurant',
    'Cantonese Restaurant': 'Restaurant',
    'Fried Chicken Joint': 'Restaurant',
    'Moroccan Restaurant': 'Restaurant',
    'Thai Restaurant': 'Restaurant',
    'Portuguese Restaurant': 'Restaurant',
    'Gastropub': 'Restaurant',
    'Caribbean Restaurant': 'Restaurant',
    'Seafood Restaurant': 'Restaurant',
    'Burrito Restaurant': 'Restaurant',
    'Dim Sum Restaurant': 'Restaurant',
    'Japanese Restaurant': 'Restaurant',
    'Mediterranean Restaurant': 'Restaurant',
    'North Indian Restaurant': 'Restaurant',
    'Lebanese Restaurant': 'Restaurant',
    'Modern European Restaurant': 'Restaurant',
    'New American Restaurant': 'Restaurant',
    'Spanish Restaurant': 'Restaurant',
    'Indie Movie Theater': 'Entertainment',
    'Pastry Shop': 'Cafe',
    'Food Truck': 'Restaurant',
    'Internet Cafe': 'Restaurant',
    'Kebab Restaurant': 'Restaurant',
    'Sushi Restaurant': 'Restaurant',
    'Deli': 'Restaurant',
    'Hotel Bar': 'Restaurant',
    'Vietnamese Restaurant': 'Restaurant',
    'Middle Eastern Restaurant': 'Restaurant',
    'Escape Room': 'Entertainment',
    'Dumpling Restaurant': 'Restaurant',
    'Buffet': 'Restaurant',
    'Sandwich Spot': 'Restaurant',
    'Spa': 'Personal Service',
    'Fishing Store': 'Retail',
    'Mobile Phone Store': 'Retail',
    'Cafe, Coffee, and Tea House': 'Cafe',
    'Car Parts and Accessories': 'Retail',
    'Bakery': 'Cafe',
    'Tanning Salon': 'Personal Service',
    'Stationery Store': 'Retail',
    'Outlet Store': 'Retail',
    'Dance Store': 'Retail',
    'Lingerie Store': 'Retail',
    'Sporting Goods Retail': 'Retail',
    "Women's Store": 'Retail',
    'Smoke Shop': 'Retail',
    'Sports and Recreation': 'Active',
    'Yoga Studio': 'Active',
    'Office Supply Store': 'Retail',
    'Strip Club': 'Entertainment',
    'Carpet Store': 'Retail',
    'Eyecare Store': 'Personal Service',
    'Fashion Accessories Store': 'Retail',
    'Snack Place': 'Cafe',
    'Pakistani Restaurant': 'Restaurant',
    'Lounge': 'Entertainment',
    'Music Venue': 'Entertainment',
    'Eastern European Restaurant': 'Restaurant',
    'Perfume Store': 'Retail',
    'Outdoor Sculpture': 'Entertainment',
    'Framing Store': 'Retail',
    'Art Gallery': 'Entertainment',
    'Brewery': 'Restaurant',
    'Garden Center': 'Retail',
    'Tobacco Store': 'Convenience Store',
    'Beer Garden': 'Entertainment',
    'Comedy Club': 'Entertainment',
    'Boutique': 'Retail',
    'Bingo Center': 'Entertainment',
    'Irish Pub': 'Entertainment',
    'African Restaurant': 'Restaurant',
    'Party Supply Store': 'Retail',
    'Laundry Service': 'Personal Service',
    'Mexican Restaurant': 'Restaurant',
    'Museum': 'Entertainment',
    'Pool Hall': 'Entertainment',
    'Colombian Restaurant': 'Restaurant',
    'Golf Store': 'Retail',
    'Cafeteria': 'Restaurant',
    'History Museum': 'Entertainment',
    'Food Court': 'Restaurant',
    'Jazz and Blues Venue': 'Entertainment',
    'Wine Store': 'Retail',
    'Dance Studio': 'Active',
    'Video Store': 'Entertainment',
    'Rooftop Bar': 'Entertainment',
    'Outdoor Supply Store': 'Retail',
    'Hot Dog Joint': 'Restaurant',
    'Bagel Shop': 'Cafe',
    'Aquarium': 'Entertainment',
    'Adult Store': 'Retail',
    'Flea Market': 'Retail',
    'Gay Bar': 'Entertainment',
    'Cuban Restaurant': 'Restaurant',
    'Pet Café': 'Cafe',
    'Laser Tag Center': 'Entertainment',
    'Brazilian Restaurant': 'Restaurant',
    'Polish Restaurant': 'Restaurant',
    'BBQ Joint': 'Restaurant',
    'Persian Restaurant': 'Restaurant',
    'Swimming Pool': 'Active',
    'Hookah Bar': 'Entertainment',
    'Performing Arts Venue': 'Entertainment',
    'Falafel Restaurant': 'Restaurant',
    'Meat and Seafood Store': 'Retail',
    'Distillery': 'Entertainment',
    'Concert Hall': 'Entertainment',
    'Iraqi Restaurant': 'Restaurant',
    'Sri Lankan Restaurant': 'Restaurant',
    'Malay Restaurant': 'Restaurant',
    'Bowling Alley': 'Entertainment',
    'Public Art': 'Entertainment',
    'Soup Spot': 'Restaurant',
    'Gaming Cafe': 'Entertainment',
    'Winery': 'Entertainment',
    'Venezuelan Restaurant': 'Restaurant',
    'Pie Shop': 'Cafe',
    'Supplement Store': 'Retail',
    'Argentinian Restaurant': 'Restaurant',
    'Brasserie': 'Restaurant',
    'Karaoke Bar': 'Entertainment',
    'Belgian Restaurant': 'Restaurant',
    'Comfort Food Restaurant': 'Restaurant',
    'Farmers Market': 'Foodstore',
    'Shawarma Restaurant': 'Restaurant',
    'Chocolate Store': 'Retail',
    'South Indian Restaurant': 'Restaurant',
    'Beach Bar': 'Entertainment',
    'Souvlaki Shop': 'Restaurant',
    'Latin American Restaurant': 'Restaurant',
    'Classic and Antique Car Dealership': 'Retail',
    'Romanian Restaurant': 'Restaurant',
    'Street Art': 'Entertainment',
    'Surf Store': 'Retail',
    'Water Park': 'Entertainment',
    'Souvenir Store': 'Retail',
    'Leather Goods Store': 'Retail',
    'German Restaurant': 'Restaurant',
    'Szechuan Restaurant': 'Restaurant',
    'Peruvian Restaurant': 'Restaurant',
}

# ──────────────────────────────────────────────
# 6. Apply classification
# ──────────────────────────────────────────────
print("\nClassifying Foursquare data into broad retail types...")

# Handle column name collision from join
# If 'category_n' exists in both, it will be suffixed.
category_col = FOURSQUARE_CATEGORY_COL
if category_col not in foursquare_with_rc.columns:
    # Try with single underscore suffix
    suffixed_col_1 = f"{category_col}_fsq"
    # Try with double underscore suffix (as seen in output)
    suffixed_col_2 = f"{category_col}__fsq"
    
    if suffixed_col_1 in foursquare_with_rc.columns:
        print(f"Note: Using suffixed column '{suffixed_col_1}' for classification.")
        category_col = suffixed_col_1
    elif suffixed_col_2 in foursquare_with_rc.columns:
        print(f"Note: Using suffixed column '{suffixed_col_2}' for classification.")
        category_col = suffixed_col_2
    else:
        print(f"❌ ERROR: Could not find '{category_col}', '{suffixed_col_1}', or '{suffixed_col_2}' after join.")
        print(f"Available columns: {list(foursquare_with_rc.columns)}")
        exit(1)

# Map the detailed category to a broad retail type
foursquare_with_rc["retail_type"] = foursquare_with_rc[category_col].map(mapping)

# Check for any categories NOT in the mapping
unmapped_mask = (
    foursquare_with_rc["retail_type"].isna()
    & foursquare_with_rc[category_col].notna()
)
unmapped = foursquare_with_rc.loc[unmapped_mask, category_col].unique()

if len(unmapped) > 0:
    print(f"\n⚠ {len(unmapped)} category name(s) NOT found in the mapping:")
    for name in sorted(unmapped):
        print(f"  - {name}")
else:
    print("All categories successfully mapped!")

# ──────────────────────────────────────────────
# 7. Keep only stores that are INSIDE a retail
#    centre (i.e. drop those with no match)
# ──────────────────────────────────────────────
stores_in_rc = foursquare_with_rc.dropna(subset=[join_col]).copy()
print(f"\nStores inside a retail centre: {len(stores_in_rc)}")

# ──────────────────────────────────────────────
# 8. Count retail types per retail centre
# ──────────────────────────────────────────────
print("\nCounting retail types per retail centre...")

# We need the retail centre identifier. 
# If join_col is the index from the right df, we might need to fetch the ID column 
# from the retail_centres df if it's not in the joined result.
# However, usually sjoin keeps columns from right df.
# Let's verify we have the RC_ID column in stores_in_rc.

# Check if RETAIL_CENTRE_NAME_COL is in stores_in_rc
rc_id_col = RETAIL_CENTRE_NAME_COL
# Check standard, suffixed, and double-suffixed versions
possible_id_cols = [
    rc_id_col,
    f"{rc_id_col}_rc",
    f"{rc_id_col}__rc"
]

found_id_col = None
for col in possible_id_cols:
    if col in stores_in_rc.columns:
        found_id_col = col
        break

if found_id_col:
    rc_id_col = found_id_col
else:
    # If we joined by index, and RC_ID wasn't preserved, we might be in trouble.
    # But for now let's hope it's there.
    print(f"⚠ WARNING: Could not find '{RETAIL_CENTRE_NAME_COL}' in joined data.")
    print(f"Available columns: {list(stores_in_rc.columns)}")
    # Try to fallback to join_col if it seems to be the ID
    rc_id_col = join_col

# Pivot: rows = retail centres, columns = retail types, values = counts
type_counts = (
    stores_in_rc
    .groupby([rc_id_col, "retail_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename the ID column back to standard if it was suffixed
if rc_id_col != RETAIL_CENTRE_NAME_COL:
    type_counts.rename(columns={rc_id_col: RETAIL_CENTRE_NAME_COL}, inplace=True)

# Merge the counts back onto the retail centre polygons so we keep the geometry
# First, check if any of our count columns already exist in retail_centres
# If so, they will collide (e.g. 'Retail' column). 
# We should probably drop the original ones from retail_centres or rename them
# to avoid confusion, as we want the *newly calculated* counts.
cols_to_drop = [c for c in type_counts.columns if c in retail_centres.columns and c != RETAIL_CENTRE_NAME_COL]
if cols_to_drop:
    print(f"Dropping pre-existing columns from retail_centres to avoid collision: {cols_to_drop}")
    retail_centres_clean = retail_centres.drop(columns=cols_to_drop)
else:
    retail_centres_clean = retail_centres

rc_with_counts = retail_centres_clean.merge(type_counts, on=RETAIL_CENTRE_NAME_COL, how="left")

# Fill any NaN counts with 0 (centres with no Foursquare stores at all)
count_cols = [c for c in type_counts.columns if c != RETAIL_CENTRE_NAME_COL]
rc_with_counts[count_cols] = rc_with_counts[count_cols].fillna(0).astype(int)

print(rc_with_counts[[RETAIL_CENTRE_NAME_COL] + count_cols].head(10))

# ──────────────────────────────────────────────
# 9. Save as GeoPackage
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 9. Save as GeoPackage
# ──────────────────────────────────────────────
# Clean up duplicate columns if any (e.g. from join quirks)
# Also rename 'fid' if it exists, as it can conflict with GeoPackage internal feature ids
if "fid" in rc_with_counts.columns:
    print("Renaming 'fid' column to 'fid_original' to avoid GeoPackage conflict...")
    rc_with_counts.rename(columns={"fid": "fid_original"}, inplace=True)

# Ensure geometry column is set correctly (just in case)
if "geometry" in rc_with_counts.columns:
    rc_with_counts.set_geometry("geometry", inplace=True)

output_path = os.path.join(PROCESSED_DIR, "retail_centre_type_counts.gpkg")
print(f"\nSaving results to {output_path}...")
try:
    rc_with_counts.to_file(output_path, driver="GPKG", layer="retail_centre_counts")
    print(f"✅ Success! File saved to: {output_path}")
except Exception as e:
    print(f"❌ ERROR saving file: {e}")
    # Fallback to Shapefile if GPKG fails
    shp_path = "retail_centre_type_counts.shp"
    print(f"Attempting to save as Shapefile instead: {shp_path}")
    # Shapefiles have column length limits (10 chars), so some names might get truncated
    rc_with_counts.to_file(shp_path)
    print("✅ Success! Saved as Shapefile.")
