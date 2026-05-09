import numpy as np
import pandas as pd
import config

def apply_choice_modifiers(relevant_utils: pd.DataFrame, state_df_or_attribs: pd.DataFrame, 
                           shoppers_idx: pd.Index, has_postcode: bool = True) -> pd.DataFrame:
    """
    Applies intervention, neighbourhood conformity, and distance-sensitivity
    modifications to a utility matrix slice. 
    
    This function handles the "Echo Chamber" effect by computing local max utilities 
    within demographic/geographic groups.
    """
    # Intervention Control: boost globally largest centre
    intervention_mult = getattr(config, 'RETAIL_INTERVENTION', 1.0)
    if intervention_mult != 1.0 and not relevant_utils.empty:
        top_center = relevant_utils.sum(axis=0).idxmax()
        relevant_utils[top_center] *= intervention_mult

    # Neighbourhood Conformity ("Echo Chamber" effect)
    # Use agent-specific conformity weight if available, else fallback to config
    if 'Conformity_Weight' in state_df_or_attribs.columns:
        conformity = state_df_or_attribs.loc[shoppers_idx, 'Conformity_Weight']
    else:
        conformity = getattr(config, 'NEIGHBOURHOOD_CONFORMITY', 0.0)

    if (isinstance(conformity, pd.Series) or conformity > 0.0) and has_postcode and 'Postcode' in state_df_or_attribs.columns:
        # Use agent-specific diffusion weight if available
        if 'Diffusion_Weight' in state_df_or_attribs.columns:
            demo_weight = state_df_or_attribs.loc[shoppers_idx, 'Diffusion_Weight']
        else:
            demo_weight = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.0)
        agent_pcs   = state_df_or_attribs.loc[shoppers_idx, 'Postcode'].str[:3].fillna('UNK')

        has_age = 'age_years'      in state_df_or_attribs.columns
        has_inc = 'salary_yearly'  in state_df_or_attribs.columns

        is_demo_active = (demo_weight > 0.0).any() if isinstance(demo_weight, pd.Series) else demo_weight > 0.0
        if is_demo_active and has_age and has_inc:
            # Age bands: young (<30), mid (30-50), senior (>50)
            ages = state_df_or_attribs.loc[shoppers_idx, 'age_years'].fillna(35)
            age_bands = pd.cut(ages, bins=[0, 30, 50, 150],
                               labels=['Y', 'M', 'S'], right=False)

            # Income quartiles computed on the active subsample
            incomes = state_df_or_attribs.loc[shoppers_idx, 'salary_yearly'].fillna(0)
            try:
                inc_bands = pd.qcut(incomes, q=4,
                                    labels=['q1', 'q2', 'q3', 'q4'],
                                    duplicates='drop')
    # Distance Sensitivity (beta scale)
    dist_sens = getattr(config, 'DISTANCE_SENSITIVITY', 1.0)
    if dist_sens != 1.0:
        mask = relevant_utils > 0
        relevant_utils[mask] = relevant_utils[mask] ** dist_sens

    return relevant_utils


def sample_destinations(relevant_utils, shoppers_idx):
    """
    Draws one destination per agent using softmax over utility scores.
    Returns a Series (index = shoppers_idx) mapping agent → chosen centre ID.
    """
    row_sums = relevant_utils.sum(axis=1)
    valid = row_sums > 0
    if not valid.any():
        return pd.Series(None, index=shoppers_idx, dtype=object)

    valid_utils = relevant_utils[valid]
    valid_idx   = shoppers_idx[valid.values]

    beta = getattr(config, 'SOFTMAX_BETA', 5.0)
    arr  = valid_utils.values.astype(np.float32)

    scaled = beta * arr
    mask = (arr <= 0)
    scaled[mask] = -1e10
    
    scaled -= pd.DataFrame(scaled).max(axis=1).values.reshape(-1, 1)
    exp_u  = np.exp(scaled)
    exp_u[mask] = 0.0
    
    denom = exp_u.sum(axis=1, keepdims=True)
    probs = exp_u / np.where(denom == 0, 1.0, denom)

    cumsum = probs.cumsum(axis=1)
    cumsum /= cumsum[:, -1:]

    rand = np.random.rand(len(cumsum), 1)
    choice_indices = (rand < cumsum).argmax(axis=1)
    choice_labels  = valid_utils.columns[choice_indices]

    result = pd.Series(None, index=shoppers_idx, dtype=object)
    result.loc[valid_idx] = choice_labels.values
    return result


def joint_mode_destination_choice(aligned_mode_utils, shoppers_idx):
    """
    Joint transport-mode and destination choice via proportional sampling.
    """
    destinations  = pd.Series(None, index=shoppers_idx, dtype=object)
    modes_used    = pd.Series(None, index=shoppers_idx, dtype=object)
    chosen_scores = pd.Series(0.0,  index=shoppers_idx)

    from .constants import TRANSPORT_MODES
    all_modes = [m for m in TRANSPORT_MODES
                 if m in aligned_mode_utils
                 and aligned_mode_utils[m] is not None
                 and not aligned_mode_utils[m].empty]
    if not all_modes:
        return destinations, modes_used, chosen_scores

    centres = aligned_mode_utils[all_modes[0]].columns
    n_centres = len(centres)

    stacked  = np.concatenate([aligned_mode_utils[m].values for m in all_modes], axis=1)
    row_sums = stacked.sum(axis=1)
    valid    = row_sums > 0

    if not valid.any():
        return destinations, modes_used, chosen_scores

    valid_stacked = stacked[valid]
    valid_idx     = shoppers_idx[valid]

    beta   = getattr(config, 'SOFTMAX_BETA', 5.0)
    scaled = (beta * valid_stacked).astype(np.float32)
    
    mask = (valid_stacked <= 0)
    scaled[mask] = -1e10
    
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_u  = np.exp(scaled)
    exp_u[mask] = 0.0
    
    denom = exp_u.sum(axis=1, keepdims=True)
    probs = exp_u / np.where(denom == 0, 1.0, denom)

    cumsum = probs.cumsum(axis=1)
    cumsum /= cumsum[:, -1:]

    rand        = np.random.rand(len(cumsum), 1)
    choice_flat = (rand < cumsum).argmax(axis=1)

    mode_indices   = choice_flat // n_centres
    centre_indices = choice_flat  % n_centres

    chosen_modes   = [all_modes[i] for i in mode_indices]
    chosen_centres = centres[centre_indices]
    
    picked_utils = valid_stacked[np.arange(len(choice_flat)), choice_flat]

    destinations.loc[valid_idx]  = chosen_centres.values
    modes_used.loc[valid_idx]    = chosen_modes
    chosen_scores.loc[valid_idx] = picked_utils

    return destinations, modes_used, chosen_scores


def demographic_similarity_scores(demo_lookup, influencer_ids, target_ids,
                                  age_min, age_range, inc_min, inc_range,
                                  bandwidth):
    """
    Computes a Gaussian similarity score in [0, 1] for each target agent relative
    to the demographic centroid of the influencer group.
    """
    inf_data = demo_lookup.reindex(influencer_ids).dropna()
    tgt_data = demo_lookup.reindex(target_ids).fillna(demo_lookup.mean())

    if inf_data.empty:
        return np.ones(len(target_ids))

    def _normalise(df):
        out = df.copy()
        if age_range > 0:
            out['age_years']     = (out['age_years']     - age_min) / age_range
        if inc_range > 0:
            out['salary_yearly'] = (out['salary_yearly'] - inc_min) / inc_range
        return out

    inf_norm = _normalise(inf_data)
    tgt_norm = _normalise(tgt_data)

    centroid = inf_norm.mean(axis=0).values
    diffs    = tgt_norm.values - centroid
    distances = np.sqrt((diffs ** 2).sum(axis=1))

    # Support for agent-specific bandwidth (vector)
    bw = np.array(bandwidth)
    return np.exp(-distances / np.maximum(bw, 1e-6))


def apply_feedback(utility_matrix, agent_ids, chosen_centres):
    """
    Applies a simple stochastic feedback loop to a utility matrix in-place.
    """
    if len(agent_ids) == 0:
        return np.array([])

    mults = np.where(np.random.rand(len(agent_ids)) > 0.5, 1.05, 0.95)

    updates = pd.DataFrame({
        'agent':  agent_ids,
        'centre': chosen_centres,
        'mult':   mults
    })

    grouped = updates.groupby(['agent', 'centre'])['mult'].prod().reset_index()

    valid_agents  = grouped['agent'].isin(utility_matrix.index)
    valid_centres = grouped['centre'].isin(utility_matrix.columns)
    valid = grouped[valid_agents & valid_centres]

    if valid.empty:
        return mults

    agent_locs  = utility_matrix.index.get_indexer_for(valid['agent'])
    centre_locs = utility_matrix.columns.get_indexer_for(valid['centre'])
    utility_matrix.values[agent_locs, centre_locs] *= valid['mult'].values

    return mults
