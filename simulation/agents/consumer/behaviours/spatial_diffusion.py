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
    Now includes a DECAY mechanism to ensure long-term convergence.
    
    Logic:
    1. Decay existing social influence: U = Base + (U_old - Base) * Decay
    2. Add new influence: U_new = U + alpha * I(i,c)
    """
    if visits_df.empty or 'Postcode' not in attributes_df.columns:
        return []

    messages = []
    alpha = getattr(config, 'SOCIAL_SCALING_ALPHA', 0.05)
    decay = getattr(config, 'SOCIAL_DECAY_FACTOR', 1.0)
    
    # 0. Apply Centre-Specific Decay (reverting towards base geography)
    if decay < 1.0 and base_utility_matrices is not None:
        for key, matrix in utility_matrices.items():
            if key in base_utility_matrices:
                base = base_utility_matrices[key]
                # Get centre-specific decays for columns in matrix
                decays = get_centre_decays(matrix.columns)
                decay_array = np.array([decays.get(col, 0.75) for col in matrix.columns], dtype=np.float16)
                # U = Base + (U - Base) * Decay (vectorized along columns)
                matrix.update(base + (matrix - base) * decay_array)
    
    # 1. Pre-process mapping: Agent -> Postcode Sector (first 3 chars)
    if 'household' in attributes_df.columns:
        agent_to_postcode = attributes_df.drop_duplicates(subset=['household']).set_index('household')['Postcode'].str[:3]
    else:
        agent_to_postcode = attributes_df['Postcode'].str[:3]
        agent_to_postcode = agent_to_postcode[~agent_to_postcode.index.duplicated()]

    # 2. Calculate Local Popularity P(s,c)
    # P(s,c) = V(s,c) / sum_j(V(s,j))
    visits_with_pc = visits_df.copy()
    visits_with_pc['Postcode_Sector'] = visits_with_pc['AgentID'].astype(str).map(agent_to_postcode)
    visits_with_pc = visits_with_pc.dropna(subset=['Postcode_Sector'])
    
    if visits_with_pc.empty:
        return []

    # Visits per sector and centre
    v_sc = visits_with_pc.groupby(['Postcode_Sector', 'Retail_Centre']).size().reset_index(name='V_sc')
    # Total visits per sector
    v_s_total = visits_with_pc.groupby('Postcode_Sector').size().reset_index(name='V_s_total')
    
    # Merge to get P(s,c)
    pop_df = v_sc.merge(v_s_total, on='Postcode_Sector')
    pop_df['P_sc'] = pop_df['V_sc'] / pop_df['V_s_total']
    
    # 3. Calculate Socio-Demographic Similarity S_i
    # S_i = exp(-D_i / B_i) where D_i is distance to visitor centroid
    demo_cols = ['age_years', 'salary_yearly']
    has_demo = all(col in attributes_df.columns for col in demo_cols)
    
    if has_demo:
        if 'household' in attributes_df.columns:
            demo_lookup = attributes_df.drop_duplicates('household').set_index('household')[demo_cols]
        else:
            demo_lookup = attributes_df[demo_cols]
            
        # Normalization parameters
        age_min, age_max = demo_lookup['age_years'].min(), demo_lookup['age_years'].max()
        inc_min, inc_max = demo_lookup['salary_yearly'].min(), demo_lookup['salary_yearly'].max()
        age_range = age_max - age_min if age_max > age_min else 1.0
        inc_range = inc_max - inc_min if inc_max > inc_min else 1.0
        
        def normalize_demo(df):
            out = pd.DataFrame(index=df.index)
            out['age'] = (df['age_years'] - age_min) / age_range
            out['inc'] = (df['salary_yearly'] - inc_min) / inc_range
            return out
            
        agent_demo_norm = normalize_demo(demo_lookup)
        
        # Calculate centroids per centre
        visits_with_demo = visits_with_pc.merge(demo_lookup, left_on='AgentID', right_index=True)
        visitor_centroids = normalize_demo(visits_with_demo.groupby('Retail_Centre')[demo_cols].mean())
    else:
        # Fallback to no demographic influence if columns missing
        visitor_centroids = pd.DataFrame()

    # 4. Apply Additive Utility Updates
    # We iterate over centres to update the matrices in chunks for memory efficiency
    affected_centres = pop_df['Retail_Centre'].unique()
    total_updates = 0
    
    # Pre-fetch agent traits
    c_coeff = attributes_df['Conformity_Coefficient'].values if 'Conformity_Coefficient' in attributes_df.columns else 0.3
    b_open  = attributes_df['Social_Openness'].values if 'Social_Openness' in attributes_df.columns else 0.5
    
    for centre in affected_centres:
        centre_str = str(centre)
        
        # Popularity vector (mapped to agents via their postcode sector)
        c_pop_map = pop_df[pop_df['Retail_Centre'] == centre].set_index('Postcode_Sector')['P_sc']
        p_vec = agent_to_postcode.map(c_pop_map).fillna(0).values
        
        if (p_vec == 0).all(): continue
        
        # Similarity vector S_i
        if not visitor_centroids.empty and centre in visitor_centroids.index:
            centroid = visitor_centroids.loc[centre].values
            # Vectorized Euclidean Distance
            d_i = np.sqrt(((agent_demo_norm.values - centroid)**2).sum(axis=1))
            # Handle scalar vs array for b_open
            s_vec = np.exp(-d_i / np.maximum(b_open, 1e-6))
        else:
            s_vec = np.ones(len(agent_to_postcode))
            
        # Sigmoid social influence (S-curve of adoption / Bass-like diffusion)
        # f(P_sc) = 1 / (1 + exp(-k * (P_sc - P0)))
        # Zero popularity should always result in zero influence, so we mask by (p_vec > 0).
        P0 = getattr(config, 'DIFFUSION_POPULATION_RATIO', 0.05)
        k = 100.0  # Steepness of adoption S-curve
        sig_p = 1.0 / (1.0 + np.exp(-k * (p_vec - P0)))
        sig_p = sig_p * (p_vec > 0)

        # Total Influence I(i,c) using the non-linear popularity sigmoid
        i_vec = c_coeff * s_vec * sig_p
        
        # Additive Update: U = U + alpha * I
        boost_term = (alpha * i_vec).astype(np.float16)
        
        # Convert to Series once for efficient reindexing in the matrix loop
        boost_series = pd.Series(boost_term, index=agent_to_postcode.index)
        
        updated_any = False
        for matrix in utility_matrices.values():
            if centre_str in matrix.columns:
                # Optimized index intersection update
                common_idx = matrix.index.intersection(boost_series.index)
                if not common_idx.empty:
                    matrix.loc[common_idx, centre_str] += boost_series.loc[common_idx]
                    updated_any = True
        
        if updated_any:
            total_updates += 1

    messages.append(
        f"Social Influence: Applied additive utility updates to {total_updates} centres "
        f"based on local popularity share and homophilic diffusion logic."
    )
    return messages
