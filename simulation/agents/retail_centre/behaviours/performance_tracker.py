import pandas as pd
import numpy as np
from simulation.core.constants import GROCERY_MODES, TRANSPORT_MODES, TRIP_TYPE_CONFIG

def rank_retail_centres(visits_df, retail_gdf, amenity_binary):
    """
    Ranks each retail centre against its peers based on visit counts.
    Returns a dict {centre_id: {trip_type: failed_boolean}}.
    """
    if visits_df.empty or retail_gdf.empty:
        return {}, {}

    all_centres = [str(c) for c in retail_gdf.index]
    visit_counts = visits_df.groupby(['Retail_Centre', 'Trip_Type']).size().unstack(fill_value=0)
    visit_counts.index = visit_counts.index.astype(str)
    visit_counts = visit_counts.reindex(all_centres, fill_value=0)

    trip_types_to_check = {
        'grocery':       [f'{g}_{t}' for g in GROCERY_MODES for t in TRANSPORT_MODES],
        'comparison':    [f'comparison_{t}' for t in TRANSPORT_MODES],
        'service':       [f'service_{t}' for t in TRANSPORT_MODES],
        'entertainment': [f'entertainment_{t}' for t in TRANSPORT_MODES],
        'food_drink':    [f'food_drink_{t}' for t in TRANSPORT_MODES],
    }

    failed_categories = {c: [] for c in all_centres}
    participating_categories = {c: [] for c in all_centres}

    for trip_type, _ in trip_types_to_check.items():
        counts = visit_counts[trip_type] if trip_type in visit_counts.columns else pd.Series(0, index=all_centres)
        
        cfg = TRIP_TYPE_CONFIG.get(trip_type, {})
        filter_cols = cfg.get('filter_col', [])
        if isinstance(filter_cols, str): filter_cols = [filter_cols]
        
        can_provide = pd.Series(False, index=all_centres)
        for col in filter_cols:
            if col in amenity_binary:
                can_provide |= (amenity_binary[col].reindex(all_centres, fill_value=0) == 1.0)
        
        eval_targets = can_provide[can_provide].index.tolist()
        for c in eval_targets:
            participating_categories[c].append(trip_type)

        if not eval_targets:
            continue

        for centre in eval_targets:
            poi_count = retail_gdf.loc[centre, 'Total_POI_'] if centre in retail_gdf.index else 0
            if pd.isna(poi_count): poi_count = 0
            my_visits = counts.get(centre, 0)
            
            # --- Size Peer Ranking ---
            s_lower, s_upper = poi_count * 0.8, poi_count * 1.2
            size_peers = retail_gdf[(retail_gdf['Total_POI_'] >= s_lower) & 
                                    (retail_gdf['Total_POI_'] <= s_upper)].index
            size_peers = [str(p) for p in size_peers if str(p) in eval_targets]
            
            if len(size_peers) > 1:
                p_visits = counts[size_peers]
                size_rank = (sum(p_visits < my_visits) + 0.5 * sum(p_visits == my_visits)) / len(p_visits)
            else:
                size_rank = 0.5

            # --- Spatial Peer Ranking ---
            dists = retail_gdf.distance(retail_gdf.loc[centre, 'geometry'])
            spatial_peers = dists[eval_targets].nsmallest(6).index.tolist()
            if centre in spatial_peers: spatial_peers.remove(centre)
            spatial_peers = [str(p) for p in spatial_peers[:5]]
            
            if len(spatial_peers) > 0:
                p_visits = counts[spatial_peers]
                spatial_rank = (sum(p_visits < my_visits) + 0.5 * sum(p_visits == my_visits)) / len(p_visits)
            else:
                spatial_rank = 0.5

            # --- Weighting ---
            size_weight = np.interp(poi_count, [10, 200], [0.3, 0.7])
            final_rank = (size_rank * size_weight) + (spatial_rank * (1.0 - size_weight))

            if final_rank <= 0.10:
                failed_categories[centre].append(trip_type)

    return failed_categories, participating_categories
