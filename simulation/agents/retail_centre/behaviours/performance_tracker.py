import pandas as pd
import numpy as np
import config
from simulation.core.constants import GROCERY_MODES, TRANSPORT_MODES, TRIP_TYPE_CONFIG

def rank_retail_centres(visits_df, retail_gdf, amenity_binary,
                        dist_matrix, centre_list, centre_pos,
                        poi_series, poi_sorted_idx, poi_sorted_vals, poi_sorted_ids):
    """
    Ranks each retail centre against its peers based on visit counts.
    Uses pre-computed distance matrix and sorted POI index for O(log N) lookups.
    Returns a dict {centre_id: [failed_trip_types]}, {centre_id: [participating_trip_types]}.
    """
    if visits_df.empty or retail_gdf.empty:
        return {}, {}

    all_centres = [str(c) for c in retail_gdf.index]
    visit_counts = visits_df.groupby(['Retail_Centre', 'Trip_Type']).size().unstack(fill_value=0)
    visit_counts.index = visit_counts.index.astype(str)
    visit_counts = visit_counts.reindex(all_centres, fill_value=0)

    failure_limit = getattr(config, 'RETAIL_FAILURE_THRESHOLD', 0.10)
    weights_range = getattr(config, 'RETAIL_PEER_SIZE_WEIGHT', [0.3, 0.7])

    failed_categories        = {c: [] for c in all_centres}
    participating_categories = {c: [] for c in all_centres}


    for trip_type in TRIP_TYPE_CONFIG:
        cfg = TRIP_TYPE_CONFIG.get(trip_type, {})
        filter_cols = cfg.get('filter_col', [])
        if isinstance(filter_cols, str):
            filter_cols = [filter_cols]

        # Determine which centres can provide this trip type (vectorised OR)
        can_provide = np.zeros(len(all_centres), dtype=bool)
        for col in filter_cols:
            if col in amenity_binary:
                vec = amenity_binary[col].reindex(all_centres, fill_value=0).values
                can_provide |= (vec == 1.0)

        eval_targets = [c for c, flag in zip(all_centres, can_provide) if flag]
        eval_set     = set(eval_targets)

        for c in eval_targets:
            participating_categories[c].append(trip_type)

        if not eval_targets:
            continue

        counts = visit_counts[trip_type] if trip_type in visit_counts.columns else pd.Series(0, index=all_centres)

        for centre in eval_targets:
            my_visits = counts.get(centre, 0)
            poi_count = poi_series.get(centre, 0)
            if pd.isna(poi_count):
                poi_count = 0

            # --- Size Peer Ranking via searchsorted (O(log N)) ---
            s_lower = poi_count * 0.8
            s_upper = poi_count * 1.2
            lo = int(np.searchsorted(poi_sorted_vals, s_lower, side='left'))
            hi = int(np.searchsorted(poi_sorted_vals, s_upper, side='right'))
            size_peer_ids = [p for p in poi_sorted_ids[lo:hi]
                             if p != centre and p in eval_set]

            if size_peer_ids:
                p_visits  = counts[size_peer_ids].values
                size_rank = (np.sum(p_visits < my_visits) + 0.5 * np.sum(p_visits == my_visits)) / len(p_visits)
            else:
                size_rank = 0.5

            # --- Spatial Peer Ranking via pre-computed distance matrix ---
            c_idx = centre_pos.get(centre)
            if c_idx is not None:
                # Get distances from this centre to all others in eval_targets (excluding self)
                eval_peers = [e for e in eval_targets if e != centre and e in centre_pos]
                if eval_peers:
                    peer_idxs   = np.array([centre_pos[e] for e in eval_peers])
                    peer_dists  = dist_matrix[c_idx, peer_idxs]
                    nearest_5   = np.argsort(peer_dists)[:5]
                    spatial_peer_ids = [eval_peers[j] for j in nearest_5]
                else:
                    spatial_peer_ids = []
            else:
                spatial_peer_ids = []

            if spatial_peer_ids:
                p_visits     = counts[spatial_peer_ids].values
                spatial_rank = (np.sum(p_visits < my_visits) + 0.5 * np.sum(p_visits == my_visits)) / len(p_visits)
            else:
                spatial_rank = 0.5

            # --- Weighted Score (Eq. 7 & 8) ---
            size_weight = float(np.interp(poi_count, [10, 200], weights_range))
            final_rank  = size_rank * size_weight + spatial_rank * (1.0 - size_weight)

            if final_rank <= failure_limit:
                failed_categories[centre].append(trip_type)

    return failed_categories, participating_categories
