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
        # Cast to np.float16 to prevent upcasting of the column
        relevant_utils[top_center] *= np.float16(intervention_mult)

    # --- MODIFIED: Daily Neighbourhood Conformity is now handled by the 
    # periodic additive mechanism in spatial_diffusion.py to speed up daily steps.
    
    # Distance Sensitivity (beta scale)
    dist_sens = getattr(config, 'DISTANCE_SENSITIVITY', 1.0)
    if dist_sens != 1.0:
        # Perform in-place on the numpy array to prevent pandas upcasting to float64
        vals = relevant_utils.values
        mask = vals > 0
        vals[mask] = vals[mask] ** np.float16(dist_sens)

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
    # Use float16 for massive memory savings in intermediate arrays
    arr  = valid_utils.values.astype(np.float16)

    scaled = np.float16(beta) * arr
    mask = (arr <= 0)
    # Use a safe minimum for float16 (-1e4 is usually enough for softmax)
    scaled[mask] = np.float16(-10000.0) 
    
    # Subtract max for numerical stability
    row_max = scaled.max(axis=1, keepdims=True)
    scaled -= row_max
    
    exp_u  = np.exp(scaled.astype(np.float32)) # exp needs float32 for range
    exp_u[mask] = 0.0
    
    denom = exp_u.sum(axis=1, keepdims=True)
    actually_valid = (denom > 0).flatten()
    
    if not actually_valid.any():
        return pd.Series(None, index=shoppers_idx, dtype=object)

    exp_u_filt = exp_u[actually_valid]
    denom_filt = denom[actually_valid]
    v_idx_filt = valid_idx[actually_valid]
    v_utils_filt = valid_utils[actually_valid]

    probs  = exp_u_filt / denom_filt
    cumsum = probs.cumsum(axis=1)
    cumsum /= cumsum[:, -1:]

    rand = np.random.rand(len(cumsum), 1)
    choice_indices = (rand < cumsum).argmax(axis=1)
    choice_labels  = v_utils_filt.columns[choice_indices]

    result = pd.Series(None, index=shoppers_idx, dtype=object)
    result.loc[v_idx_filt] = choice_labels.values
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
    # Memory optimization: Use float16 for the massive stacked array
    valid_stacked = valid_stacked.astype(np.float16)
    scaled = (np.float16(beta) * valid_stacked)
    
    mask = (valid_stacked <= 0)
    scaled[mask] = np.float16(-10000.0)
    
    # Numerical stability: Subtract max
    row_max = scaled.max(axis=1, keepdims=True)
    scaled -= row_max
    
    exp_u  = np.exp(scaled.astype(np.float32)) # exp needs float32
    exp_u[mask] = 0.0
    
    # 6. Softmax Probabilities
    denom = exp_u.sum(axis=1, keepdims=True)
    # If an agent has no valid destination (denom=0), they should be excluded
    actually_valid = (denom > 0).flatten()
    
    if not actually_valid.any():
        return destinations, modes_used, chosen_scores

    # Filter down to agents who have at least one valid destination
    exp_u_filt = exp_u[actually_valid]
    denom_filt = denom[actually_valid]
    v_idx_filt = valid_idx[actually_valid]
    v_stacked_filt = valid_stacked[actually_valid]
    
    probs = exp_u_filt / denom_filt
    cumsum = probs.cumsum(axis=1)
    # Ensure cumulative sum ends at exactly 1.0 to avoid floating point issues
    cumsum /= cumsum[:, -1:]

    # 7. Sampling
    rand = np.random.rand(len(cumsum), 1)
    choice_flat = (rand < cumsum).argmax(axis=1)

    mode_indices   = choice_flat // n_centres
    centre_indices = choice_flat  % n_centres

    chosen_modes   = [all_modes[i] for i in mode_indices]
    chosen_centres = centres[centre_indices]
    
    picked_utils = v_stacked_filt[np.arange(len(choice_flat)), choice_flat]

    # 8. Assign results
    destinations.loc[v_idx_filt]  = chosen_centres.values
    modes_used.loc[v_idx_filt]    = chosen_modes
    chosen_scores.loc[v_idx_filt] = picked_utils

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
