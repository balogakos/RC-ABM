import pandas as pd
import numpy as np
import config
# NOTE: demographic_similarity_scores is moved to word_of_mouth.py or utility_engine.py
from simulation.core.utility_engine import demographic_similarity_scores

def apply_spatial_diffusion_bonus(visits_df, attributes_df, utility_matrices):
    """
    Applies spatial diffusion of preferences by Postcode Sector (first 3 chars).
    Identifies the most popular retail centre per area and boosts utility for 
    agents living in that area.
    """
    # -- Threshold Logic ----------------
    pop_ratio  = getattr(config, 'DIFFUSION_POPULATION_RATIO', 0.05)
    min_floor  = getattr(config, 'DIFFUSION_MIN_VISITS', 3)
    boost_mult = getattr(config, 'DIFFUSION_BOOST_MULTIPLIER', 1.05)
    
    if visits_df.empty or 'Postcode' not in attributes_df.columns:
        return []

    messages      = []
    # Global fallbacks
    g_demo_weight   = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.0)
    g_demo_bw       = getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5)

    if 'household' in attributes_df.columns:
        df_unique         = attributes_df.drop_duplicates(subset=['household'])
        agent_to_postcode = df_unique.set_index('household')['Postcode'].str[:3]
    else:
        agent_to_postcode = attributes_df['Postcode'].str[:3]
        if agent_to_postcode.index.duplicated().any():
            agent_to_postcode = agent_to_postcode[~agent_to_postcode.index.duplicated()]

    # Calculate population per postcode sector for dynamic threshold
    pop_per_sector = agent_to_postcode.value_counts()

    visits_with_pc            = visits_df.copy()
    visits_with_pc['Postcode'] = visits_with_pc['AgentID'].map(agent_to_postcode)
    visits_with_pc             = visits_with_pc.dropna(subset=['Postcode'])
    if visits_with_pc.empty:
        return []

    counts = (visits_with_pc
              .groupby(['Postcode', 'Retail_Centre'])
              .size().reset_index(name='Visits'))

    # Determine trending centres using dynamic thresholds
    def is_trending(row):
        sector_pop = pop_per_sector.get(row['Postcode'], 0)
        dynamic_threshold = max(min_floor, sector_pop * pop_ratio)
        return row['Visits'] >= dynamic_threshold

    idx = counts.groupby('Postcode')['Visits'].idxmax()
    top_centres = counts.loc[idx]
    trending = top_centres[top_centres.apply(is_trending, axis=1)]

    if trending.empty:
        return []

    # -- Demographic setup ----------------
    # Check if we have any demographic weighting (either global or agent-specific)
    has_demo_cols = (
        'age_years'     in attributes_df.columns
        and 'salary_yearly' in attributes_df.columns
    )
    
    # We apply demo logic if either the global weight is > 0 OR if we have the agent-specific column
    has_demo = has_demo_cols and (g_demo_weight > 0.0 or 'Diffusion_Weight' in attributes_df.columns)

    if has_demo:
        demo_cols = ['age_years', 'salary_yearly']
        if 'household' in attributes_df.columns:
            demo_lookup = (attributes_df
                           .drop_duplicates('household')
                           .set_index('household')[demo_cols])
        else:
            demo_lookup = attributes_df[demo_cols].copy()

        age_min   = demo_lookup['age_years'].min()
        age_range = demo_lookup['age_years'].max()     - age_min
        inc_min   = demo_lookup['salary_yearly'].min()
        inc_range = demo_lookup['salary_yearly'].max() - inc_min

    # -- Main diffusion loop -------------------------------------------------
    num_postcodes_affected = 0
    total_boosts           = 0

    for _, row in trending.iterrows():
        pc     = row['Postcode']
        centre = str(row['Retail_Centre'])

        affected_agents = agent_to_postcode[agent_to_postcode == pc].index
        if len(affected_agents) == 0:
            continue

        if has_demo:
            visitor_ids = visits_with_pc.loc[
                (visits_with_pc['Postcode'] == pc) &
                (visits_with_pc['Retail_Centre'].astype(str) == centre),
                'AgentID'
            ].unique()

            # Extract agent-specific traits if available
            if 'Diffusion_Bandwidth' in attributes_df.columns:
                bw = attributes_df.loc[affected_agents, 'Diffusion_Bandwidth'].values
            else:
                bw = g_demo_bw

            raw_sim = demographic_similarity_scores(
                demo_lookup, visitor_ids, affected_agents,
                age_min, age_range, inc_min, inc_range, bw)

            if 'Diffusion_Weight' in attributes_df.columns:
                w = attributes_df.loc[affected_agents, 'Diffusion_Weight'].values
            else:
                w = g_demo_weight

            effective_sim = (1.0 - w) + w * raw_sim
        else:
            effective_sim = np.ones(len(affected_agents))

        per_agent_mults = 1.0 + (boost_mult - 1.0) * effective_sim

        boosted = False
        for matrix in utility_matrices.values():
            if centre in matrix.columns:
                col_idx     = matrix.columns.get_loc(centre)
                valid_mask  = affected_agents.isin(matrix.index)
                valid_agents = affected_agents[valid_mask]
                if len(valid_agents) > 0:
                    row_locs = matrix.index.get_indexer_for(valid_agents)
                    mults    = per_agent_mults[valid_mask]
                    matrix.values[row_locs, col_idx] *= mults
                    boosted = True

        if boosted:
            num_postcodes_affected += 1
            total_boosts           += len(affected_agents)

    if num_postcodes_affected > 0:
        demo_tag = " [demographic-weighted]" if has_demo else ""
        messages.append(
            f"Spatial Diffusion{demo_tag}: Word-of-Mouth boosted top centres in "
            f"{num_postcodes_affected} areas (affected {total_boosts} agent utilities).")

    return messages
