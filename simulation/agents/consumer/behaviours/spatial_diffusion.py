import pandas as pd
import numpy as np
import config
# NOTE: demographic_similarity_scores is moved to word_of_mouth.py or utility_engine.py
from simulation.core.utility_engine import demographic_similarity_scores

_centre_decay_lookup = None

def get_centre_decays(centre_ids):
    global _centre_decay_lookup
    if _centre_decay_lookup is None:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(config.RETAIL_CENTRES_GPKG, layer='retail_centre_counts')
            def clean_id(x):
                s = str(x).strip()
                return s[:-2] if s.endswith('.0') else s
            gdf['RC_ID'] = gdf['RC_ID'].apply(clean_id)
            gdf = gdf.set_index('RC_ID')
            
            # Estimate POI count per centre
            poi_col = 'Total_POI_' if 'Total_POI_' in gdf.columns else None
            if not poi_col:
                amenity_cols = ['Foodstore', 'Personal Service', 'Professional Services', 
                                'Entertainment', 'Convenience Store', 'Retail', 'Restaurant', 'Cafe']
                existing_cols = [col for col in amenity_cols if col in gdf.columns]
                gdf['POI_Count'] = gdf[existing_cols].sum(axis=1)
                poi_series = gdf['POI_Count']
            else:
                poi_series = gdf[poi_col]
                
            # Scale decay between 0.60 (small, quick decay) and 0.90 (large, slow decay)
            min_poi, max_poi = poi_series.min(), poi_series.max()
            if max_poi > min_poi:
                log_pois = np.log1p(poi_series)
                log_min, log_max = log_pois.min(), log_pois.max()
                decay_series = 0.60 + 0.30 * (log_pois - log_min) / (log_max - log_min)
            else:
                decay_series = pd.Series(0.75, index=gdf.index)
            _centre_decay_lookup = decay_series.to_dict()
        except Exception as e:
            print(f"Warning: Could not compute centre-specific decays. Error: {e}")
            _centre_decay_lookup = {}
    return {str(cid): _centre_decay_lookup.get(str(cid), 0.75) for cid in centre_ids}


def apply_spatial_diffusion_bonus(visits_df, attributes_df, utility_matrices, base_utility_matrices=None):
    """
    Implements a socially mediated utility adjustment mechanism.
    Includes a DECAY mechanism to ensure long-term convergence.

    Logic:
    1. Decay existing social influence: U = Base + (U_old - Base) * Decay
    2. Add new influence: U_new = U + alpha * I(i,c)

    Performance optimisations (mathematically identical to the original):
    - (#2) Decay applied in-place on the raw NumPy array, avoiding 3 large
      temporary DataFrames per matrix (54 temporaries/day → 1 per matrix).
    - (#3) Demographic similarity computed via a single scipy cdist call over
      all centres at once, replacing 239 individual per-centre broadcast ops.
    - (#1) Boost applied to each utility matrix with a single NumPy fancy-index
      update, replacing ~4,302 slow pandas .loc column updates per day.
    """
    if visits_df.empty or 'Postcode' not in attributes_df.columns:
        return []

    messages = []
    alpha = getattr(config, 'SOCIAL_SCALING_ALPHA', 0.05)
    decay = getattr(config, 'SOCIAL_DECAY_FACTOR', 1.0)

    # -------------------------------------------------------------------------
    # OPTIMISATION #2: In-place NumPy decay
    # Original: matrix.update(base + (matrix - base) * decay_array)
    # Created 3 large float64 DataFrames per matrix (54/day). Now uses a single
    # float32 temporary array and writes back to the float16 matrix in-place.
    # -------------------------------------------------------------------------
    if decay < 1.0 and base_utility_matrices is not None:
        for key, matrix in utility_matrices.items():
            if key in base_utility_matrices:
                base = base_utility_matrices[key]
                decays = get_centre_decays(matrix.columns)
                decay_arr = np.array(
                    [decays.get(col, 0.75) for col in matrix.columns], dtype=np.float32
                )
                # Work in float32 for precision, write back as float16
                arr32   = matrix.values.astype(np.float32)
                base32  = base.values.astype(np.float32)
                arr32  -= base32
                arr32  *= decay_arr          # broadcast: (n_agents, n_centres) * (n_centres,)
                arr32  += base32
                matrix.values[:] = arr32.astype(np.float16)

    # -------------------------------------------------------------------------
    # 1. Pre-process mapping: Agent -> Postcode Sector (first 3 chars)
    # -------------------------------------------------------------------------
    if 'household' in attributes_df.columns:
        agent_to_postcode = (
            attributes_df.drop_duplicates(subset=['household'])
            .set_index('household')['Postcode'].str[:3]
        )
    else:
        agent_to_postcode = attributes_df['Postcode'].str[:3]
        agent_to_postcode = agent_to_postcode[~agent_to_postcode.index.duplicated()]

    # -------------------------------------------------------------------------
    # 2. Calculate Local Popularity P(s,c) for ALL centres simultaneously
    # -------------------------------------------------------------------------
    visits_with_pc = visits_df.assign(
        Postcode_Sector=visits_df['AgentID'].astype(str).map(agent_to_postcode)
    ).dropna(subset=['Postcode_Sector'])

    if visits_with_pc.empty:
        return []

    v_sc       = visits_with_pc.groupby(['Postcode_Sector', 'Retail_Centre']).size().reset_index(name='V_sc')
    v_s_total  = visits_with_pc.groupby('Postcode_Sector').size().reset_index(name='V_s_total')
    pop_df     = v_sc.merge(v_s_total, on='Postcode_Sector')
    pop_df['P_sc'] = pop_df['V_sc'] / pop_df['V_s_total']

    # Pivot to (Postcode_Sector × Retail_Centre) -> dense popularity matrix
    p_pivot        = pop_df.pivot(index='Postcode_Sector', columns='Retail_Centre', values='P_sc').fillna(0.0)
    affected_centres = list(p_pivot.columns)
    n_affected     = len(affected_centres)

    if n_affected == 0:
        return []

    # Map each agent to their sector's row: produces (n_agents, n_affected) array
    agent_sectors    = agent_to_postcode.values                       # (n_agents,) sector strings
    sector_positions = p_pivot.index.get_indexer(agent_sectors)       # int positions; -1 = not found
    valid_agents     = sector_positions >= 0

    P_full = np.zeros((len(agent_sectors), n_affected), dtype=np.float32)
    if valid_agents.any():
        P_full[valid_agents] = p_pivot.values[sector_positions[valid_agents]]

    # -------------------------------------------------------------------------
    # OPTIMISATION #3: Vectorized demographic similarity via a single cdist call
    # Original: computed (n_agents,) distance vector inside a Python loop over
    # 239 centres. Now computes the full (n_agents × n_affected) distance matrix
    # in one scipy call, then broadcasts exp(-D / B) across all centres at once.
    # -------------------------------------------------------------------------
    demo_cols = ['age_years', 'salary_yearly']
    has_demo  = all(col in attributes_df.columns for col in demo_cols)

    if has_demo:
        if 'household' in attributes_df.columns:
            demo_lookup = (
                attributes_df.drop_duplicates('household').set_index('household')[demo_cols]
            )
        else:
            demo_lookup = attributes_df[demo_cols]
            demo_lookup = demo_lookup[~demo_lookup.index.duplicated()]

        # Normalisation parameters (computed once)
        age_min, age_max = demo_lookup['age_years'].min(),     demo_lookup['age_years'].max()
        inc_min, inc_max = demo_lookup['salary_yearly'].min(), demo_lookup['salary_yearly'].max()
        age_range = age_max - age_min if age_max > age_min else 1.0
        inc_range = inc_max - inc_min if inc_max > inc_min else 1.0

        # Normalise all agent demographics once: (n_agents, 2)
        agent_demo_norm = np.column_stack([
            (demo_lookup['age_years'].values     - age_min) / age_range,
            (demo_lookup['salary_yearly'].values - inc_min) / inc_range,
        ]).astype(np.float32)

        # Compute visitor centroid per centre (normalised) -> (n_affected, 2)
        visits_with_demo = visits_with_pc.join(demo_lookup, on='AgentID', how='inner')
        visits_with_demo['age_norm'] = (visits_with_demo['age_years']     - age_min) / age_range
        visits_with_demo['inc_norm'] = (visits_with_demo['salary_yearly'] - inc_min) / inc_range
        centre_centroids_df = visits_with_demo.groupby('Retail_Centre')[['age_norm', 'inc_norm']].mean()
        centre_centroids    = (
            centre_centroids_df.reindex(affected_centres).fillna(0.5).values.astype(np.float32)
        )  # (n_affected, 2)

        # Single cdist call replaces 239 individual (n_agents, 2) broadcasts
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(agent_demo_norm, centre_centroids, metric='euclidean').astype(np.float32)
        # (n_agents, n_affected)

        b_open = (
            attributes_df['Social_Openness'].values.astype(np.float32)
            if 'Social_Openness' in attributes_df.columns
            else np.full(len(agent_sectors), 0.5, dtype=np.float32)
        )
        b_open = np.maximum(b_open, 1e-6)
        S_full = np.exp(-dist_matrix / b_open[:, None])  # (n_agents, n_affected)
    else:
        # Fallback: no demographic influence
        S_full = np.ones((len(agent_sectors), n_affected), dtype=np.float32)

    # -------------------------------------------------------------------------
    # OPTIMISATION #1: Compute full boost matrix (n_agents × n_affected) at once,
    # then apply to each utility matrix with a single NumPy fancy-index update.
    # Original: looped over 239 centres × 18 matrices = 4,302 pandas .loc updates.
    # -------------------------------------------------------------------------
    c_coeff = (
        attributes_df['Conformity_Coefficient'].values.astype(np.float32)
        if 'Conformity_Coefficient' in attributes_df.columns
        else np.full(len(agent_sectors), 0.3, dtype=np.float32)
    )

    # Sigmoid S-curve of social adoption (Bass-like diffusion)
    P0    = getattr(config, 'DIFFUSION_POPULATION_RATIO', 0.05)
    k     = 100.0
    sig_p = 1.0 / (1.0 + np.exp(-k * (P_full - P0)))
    sig_p *= (P_full > 0)   # Zero popularity -> zero influence

    I_full       = c_coeff[:, None] * S_full * sig_p   # (n_agents, n_affected)
    boost_matrix = (alpha * I_full).astype(np.float16)  # (n_agents, n_affected)

    affected_centres_arr = np.array(affected_centres)

    total_updates = 0
    for matrix in utility_matrices.values():
        # Find which affected centres exist as columns in this matrix
        col_positions = matrix.columns.get_indexer(affected_centres_arr)
        valid_col_mask = col_positions >= 0
        if not valid_col_mask.any():
            continue

        # Find which agents (rows of agent_to_postcode) map to rows in this matrix
        row_positions = matrix.index.get_indexer(agent_to_postcode.index.astype(str))
        valid_row_mask = row_positions >= 0
        if not valid_row_mask.any():
            continue

        # Integer positions for numpy fancy indexing
        valid_row_indices = row_positions[valid_row_mask]               # positions in matrix rows
        valid_col_indices = col_positions[valid_col_mask]               # positions in matrix columns

        # Sub-matrix of boost values for the valid agent/centre combinations
        agent_idxs_in_boost  = np.where(valid_row_mask)[0]             # which rows of boost_matrix
        centre_idxs_in_boost = np.where(valid_col_mask)[0]             # which cols of boost_matrix
        boost_sub = boost_matrix[np.ix_(agent_idxs_in_boost, centre_idxs_in_boost)]

        # Single in-place NumPy update (no pandas overhead)
        matrix.values[np.ix_(valid_row_indices, valid_col_indices)] += boost_sub
        total_updates += 1

    # Count centres that received a non-zero net boost
    n_boosted = int((boost_matrix.sum(axis=0) > 0).sum())
    messages.append(
        f"Social Influence: Applied additive utility updates to {n_boosted} centres "
        f"based on local popularity share and homophilic diffusion logic."
    )
    return messages
