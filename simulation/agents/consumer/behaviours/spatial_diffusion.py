import pandas as pd
import numpy as np
import config
# NOTE: demographic_similarity_scores is moved to word_of_mouth.py or utility_engine.py
from simulation.core.utility_engine import demographic_similarity_scores

def apply_spatial_diffusion_bonus(visits_df, attributes_df, utility_matrices):
    """
    Implements a socially mediated utility adjustment mechanism based on:
    I(i,c) = C_i * S_i * P(s,c)
    U_new = U + alpha * I(i,c)
    """
    if visits_df.empty or 'Postcode' not in attributes_df.columns:
        return []

    messages = []
    alpha = getattr(config, 'SOCIAL_SCALING_ALPHA', 0.05)
    
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
    c_coeff = attributes_df['Conformity_Coefficient'] if 'Conformity_Coefficient' in attributes_df.columns else 0.3
    b_open  = attributes_df['Social_Openness'] if 'Social_Openness' in attributes_df.columns else 0.5
    
    for centre in affected_centres:
        centre_str = str(centre)
        
        # Popularity vector (mapped to agents via their postcode sector)
        c_pop_map = pop_df[pop_df['Retail_Centre'] == centre].set_index('Postcode_Sector')['P_sc']
        p_vec = agent_to_postcode.map(c_pop_map).fillna(0)
        
        if (p_vec == 0).all(): continue
        
        # Similarity vector S_i
        if not visitor_centroids.empty and centre in visitor_centroids.index:
            centroid = visitor_centroids.loc[centre].values
            # Vectorized Euclidean Distance
            d_i = np.sqrt(((agent_demo_norm.values - centroid)**2).sum(axis=1))
            s_vec = np.exp(-d_i / np.maximum(b_open.values, 1e-6))
        else:
            s_vec = np.ones(len(agent_to_postcode))
            
        # Total Influence I(i,c)
        i_vec = c_coeff * s_vec * p_vec
        
        # Additive Update: U = U + alpha * I
        boost_term = (alpha * i_vec).astype(np.float16)
        
        updated_any = False
        for matrix in utility_matrices.values():
            if centre_str in matrix.columns:
                # Align indices: only update agents present in the matrix
                common_idx = matrix.index.intersection(boost_term.index)
                if not common_idx.empty:
                    matrix.loc[common_idx, centre_str] += boost_term.loc[common_idx]
                    updated_any = True
        
        if updated_any:
            total_updates += 1

    messages.append(
        f"Social Influence: Applied additive utility updates to {total_updates} centres "
        f"based on local popularity share and homophilic diffusion logic."
    )
    return messages
