import sys; sys.path.insert(0, '.')
import config, agent
import pandas as pd, geopandas as gpd

def _is_centre_col(name):
    try: float(name); return True
    except: return False

def _split_matrix(df, household_col='household'):
    centre_cols = [c for c in df.columns if _is_centre_col(c)]
    meta_cols   = [c for c in df.columns if not _is_centre_col(c)]
    matrix = df[centre_cols].copy()
    matrix.columns = [c[:-2] if c.endswith('.0') else c for c in matrix.columns]
    if household_col in df.columns:
        matrix.index = df[household_col]
    return df[meta_cols], matrix

consumers, matrix_bulk = _split_matrix(pd.read_parquet(config.UTILITY_SCORES_BULK_WITH_TRIPS))
matrix_bulk = matrix_bulk.fillna(0)
_, matrix_con = _split_matrix(pd.read_parquet(config.UTILITY_SCORES_CONV))
matrix_con = matrix_con.fillna(0)
_, matrix_avg = _split_matrix(pd.read_parquet(config.UTILITY_SCORES_AVG))
matrix_avg = matrix_avg.fillna(0)

gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
gdf['RC_ID'] = gdf['RC_ID'].apply(lambda x: str(x)[:-2] if str(x).endswith('.0') else str(x))
gdf = gdf.set_index('RC_ID')
amenity_cols = ['Foodstore','Personal Service','Professional Services','Entertainment','Convenience Store','Retail','Restaurant','Cafe']
amenity_binary = {col: (gdf[col]>0).astype(float) for col in amenity_cols if col in gdf.columns}
utility_matrices = {'bulk': matrix_bulk, 'convenience': matrix_con, 'average': matrix_avg}

# Check ID alignment
overlap = set(matrix_avg.columns) & set(gdf.index)
print(f"Centre ID overlap: {len(overlap)} / {len(matrix_avg.columns)} matrix cols match GPKG")
print(f"Entertainment centres in overlap: {sum(gdf.loc[gdf.index.isin(overlap), 'Entertainment'] > 0)}")

n = 500
state_df, sampled = agent.initialize_agent_state(n, consumers)
state_df['Stock'] = 5.0

all_visits = []
for day in range(1, 4):
    state_df = agent.consume(state_df)
    needs_grocery = agent.check_shopping_need(state_df)
    if needs_grocery.any():
        mode_series  = agent.choose_mode(sampled, needs_grocery)
        destinations = agent.choose_destination(state_df, needs_grocery, mode_series, utility_matrices)
        valid = destinations.notna() & (mode_series != 'online')
        if valid.any():
            idx = destinations[valid].index
            all_visits.append(pd.DataFrame({'Day':day,'AgentID':state_df.loc[idx,'AgentID'].values,'Trip_Type':'grocery','Retail_Centre':destinations[valid].values,'Mode':mode_series[valid].values,'Distance':0.0}))
        state_df = agent.update_stock_after_shop(state_df, needs_grocery)
    triggered = agent.trigger_trips(sampled)
    for trip_type, mask in triggered.items():
        if not mask.any(): continue
        dests = agent.choose_destination_for_trip(trip_type, mask, sampled, utility_matrices, amenity_binary)
        valid = dests.notna()
        if valid.any():
            idx = dests[valid].index
            all_visits.append(pd.DataFrame({'Day':day,'AgentID':sampled.loc[idx,'household'].values,'Trip_Type':trip_type,'Retail_Centre':dests[valid].values,'Mode':'physical','Distance':0.0}))

df = pd.concat(all_visits, ignore_index=True)
print()
print("=== FINAL: Trip Type Breakdown (3 days, 500 agents) ===")
print(df['Trip_Type'].value_counts().to_string())
print()
print("=== Grocery: max trips per agent per day (must = 1) ===")
groc = df[df['Trip_Type']=='grocery'].groupby(['Day','AgentID']).size()
print(f"Max: {groc.max()}  Mean: {groc.mean():.2f}")
print()
print("Sample rows:")
print(df.groupby('Trip_Type').first().to_string())
