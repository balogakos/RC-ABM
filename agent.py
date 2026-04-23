"""
Retail ABM - Agent Logic Library

Defines how agents behave: consuming stock, deciding to shop, choosing a transport
mode, choosing a destination, and triggering non-grocery NTS trips.

Transport Modes
---------------
Each grocery type (bulk / convenience) and NTS trip type (average) now has THREE
per-mode utility matrices, one per transport mode:
    bulk_walk, bulk_drive, bulk_pt
    convenience_walk, convenience_drive, convenience_pt
    average_walk, average_drive, average_pt

The utility_matrices dict passed into every destination-choice function must use
these flat key names.

All functions are designed to operate on the full agent population at once (vectorised).
"""

import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSPORT_MODES = ['drive']
GROCERY_MODES   = ['bulk', 'convenience']


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_agent_state(num_agents, attributes_df):
    """
    Initialises the dynamic state DataFrame for agents.
    If attributes_df has fewer rows than num_agents, rows are resampled.
    Returns (state_df, attributes_df).
    """
    if len(attributes_df) < num_agents:
        attributes_df = attributes_df.sample(n=num_agents, replace=True).reset_index(drop=True)
    else:
        attributes_df = attributes_df.sample(n=num_agents, replace=False).reset_index(drop=True)

    state_df = pd.DataFrame({
        'AgentID':         attributes_df['household'].values
                           if 'household' in attributes_df.columns
                           else range(num_agents),
        'Postcode':        attributes_df['Postcode'].values
                           if 'Postcode' in attributes_df.columns
                           else 'UNKNOWN',
        'Stock':           np.random.uniform(50, 100, num_agents),
        'Consumption_Rate': np.clip(
                               np.random.normal(config.DAILY_CONSUMPTION_MEAN,
                                                config.DAILY_CONSUMPTION_STD,
                                                num_agents),
                               0.5, None),
        'Last_Shop_Day':   -1,
        'Shopping_Mode':   None,
    })

    return state_df, attributes_df


# ---------------------------------------------------------------------------
# Grocery system (stock-based)
# ---------------------------------------------------------------------------

def consume(state_df):
    """Reduces each agent's stock by their daily consumption rate (floor 0)."""
    state_df['Stock'] = (state_df['Stock'] - state_df['Consumption_Rate']).clip(lower=0.0)
    return state_df


def check_shopping_need(state_df):
    """Returns a boolean Series: True where Stock < REORDER_THRESHOLD."""
    return state_df['Stock'] < config.REORDER_THRESHOLD


def choose_mode(attributes_df, shopping_mask):
    """
    Draws a grocery shopping mode (online / bulk / convenience) for each agent
    whose index appears in shopping_mask (True entries).
    Uses per-agent probabilities from prob_online, prob_bulk, prob_convenience columns.
    Returns a Series aligned to attributes_df.index (None for non-shoppers).
    """
    modes = pd.Series(index=attributes_df.index, data=None, dtype=object)
    shoppers = attributes_df[shopping_mask]
    if shoppers.empty:
        return modes

    probs = shoppers[['prob_online', 'prob_bulk', 'prob_convenience']].copy()
    row_sums = probs.sum(axis=1)
    probs = probs.div(row_sums, axis=0).fillna(0)

    rand = np.random.random(len(shoppers))
    cumsum = probs.cumsum(axis=1)

    chosen = pd.Series(index=shoppers.index, data='convenience')  # default
    chosen[rand < cumsum['prob_online']] = 'online'
    chosen[(rand >= cumsum['prob_online']) & (rand < cumsum['prob_bulk'])] = 'bulk'

    modes.update(chosen)
    return modes


def choose_transport_mode(attributes_df, physical_mask):
    """
    Draws a transport mode (walk / drive / pt) for each physically-shopping agent.
    Uses per-agent probabilities from prob_walk, prob_drive, prob_pt columns.
    Falls back to equal probability if columns are absent.

    Returns a Series aligned to attributes_df.index (None for non-physical agents).
    """
    # Since we moved to a single distance-decayed drive model, 
    # all physical trips are now recorded as 'drive'.
    transport = pd.Series(index=attributes_df.index, data=None, dtype=object)
    shoppers = attributes_df[physical_mask]
    if shoppers.empty:
        return transport

    chosen = pd.Series(index=shoppers.index, data='drive')
    transport.update(chosen)
    return transport


def _apply_choice_modifiers(relevant_utils, state_df_or_attribs, shoppers_idx,
                            has_postcode=True):
    """
    Applies intervention, neighbourhood conformity, and distance-sensitivity
    modifications to a utility matrix slice, in-place on a copy.
    Returns the modified copy.
    """
    # Intervention Control: boost globally largest centre
    intervention_mult = getattr(config, 'RETAIL_INTERVENTION', 1.0)
    if intervention_mult != 1.0 and not relevant_utils.empty:
        top_center = relevant_utils.sum(axis=0).idxmax()
        relevant_utils[top_center] *= intervention_mult

    # Neighbourhood Conformity ("Echo Chamber" effect)
    # When DEMOGRAPHIC_DIFFUSION_WEIGHT > 0 and age/income columns exist, peers are
    # defined by postcode-sector × age-band × income-quartile rather than postcode alone,
    # so only demographically similar neighbours pull each other's utilities together.
    conformity = getattr(config, 'NEIGHBOURHOOD_CONFORMITY', 0.0)
    if conformity > 0.0 and has_postcode and 'Postcode' in state_df_or_attribs.columns:
        demo_weight = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.0)
        agent_pcs   = state_df_or_attribs.loc[shoppers_idx, 'Postcode'].str[:3].fillna('UNK')

        has_age = 'age_years'      in state_df_or_attribs.columns
        has_inc = 'salary_yearly'  in state_df_or_attribs.columns

        if demo_weight > 0.0 and has_age and has_inc:
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
            except ValueError:
                inc_bands = pd.Series('q2', index=shoppers_idx)

            group_key = (agent_pcs.values + '_' +
                         age_bands.astype(str).values + '_' +
                         inc_bands.astype(str).values)
        else:
            # Fallback: group by postcode sector only (original behaviour)
            group_key = agent_pcs.values

        temp_df = relevant_utils.copy()
        temp_df['_group'] = group_key
        local_maxes = (temp_df.groupby('_group')
                              .transform('max')
                              .drop(columns=['_group']))
        relevant_utils = (relevant_utils * (1 - conformity)) + (local_maxes * conformity)

    # Distance Sensitivity (beta scale)
    dist_sens = getattr(config, 'DISTANCE_SENSITIVITY', 1.0)
    if dist_sens != 1.0:
        # We mask to ensure 0.0 utility stays 0.0 even if dist_sens is 0.0 (0^0 = 1.0)
        mask = relevant_utils > 0
        relevant_utils[mask] = relevant_utils[mask] ** dist_sens

    return relevant_utils


def _sample_destinations(relevant_utils, shoppers_idx):
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

    # Softmax with log-sum-exp trick for numerical stability
    scaled = beta * arr
    
    # Strictly enforce 0-utility being impossible to select by making it -inf
    mask = (arr <= 0)
    scaled[mask] = -1e10
    
    scaled -= pd.DataFrame(scaled).max(axis=1).values.reshape(-1, 1)   # subtract row max before exp
    exp_u  = np.exp(scaled)
    # Re-apply mask to exp_u to be safe against float precision
    exp_u[mask] = 0.0
    
    denom = exp_u.sum(axis=1, keepdims=True)
    probs = exp_u / np.where(denom == 0, 1.0, denom)  # guard division by zero

    cumsum = probs.cumsum(axis=1)
    cumsum /= cumsum[:, -1:]   # guard float drift

    rand = np.random.rand(len(cumsum), 1)
    choice_indices = (rand < cumsum).argmax(axis=1)
    choice_labels  = valid_utils.columns[choice_indices]

    result = pd.Series(None, index=shoppers_idx, dtype=object)
    result.loc[valid_idx] = choice_labels.values
    return result


def _joint_mode_destination_choice(aligned_mode_utils, shoppers_idx):
    """
    Joint transport-mode and destination choice via proportional sampling.

    Instead of pre-selecting a mode from demographic probabilities, every
    (destination × mode) combination is treated as a distinct alternative.
    An agent's probability of choosing pair (c, m) is proportional to
    util[agent, c, m], so the mode that offers the highest utility *for the
    best reachable centre* is the one that tends to win.

    Parameters
    ----------
    aligned_mode_utils : dict {'walk': DataFrame, 'drive': DataFrame, 'pt': DataFrame}
                         Each DataFrame has rows = shoppers_idx, cols = RC IDs.
                         Matrices must already be reindexed, filtered (amenity masks
                         where applicable), and have choice modifiers applied.
    shoppers_idx       : pandas Index matching the rows of each DataFrame.

    Returns
    -------
    destinations  : pd.Series (index = shoppers_idx) → chosen RC ID or None
    modes_used    : pd.Series (index = shoppers_idx) → 'walk'|'drive'|'pt' or None
    chosen_scores : pd.Series (index = shoppers_idx) → raw utility score of the choice
    """
    destinations  = pd.Series(None, index=shoppers_idx, dtype=object)
    modes_used    = pd.Series(None, index=shoppers_idx, dtype=object)
    chosen_scores = pd.Series(0.0,  index=shoppers_idx)

    all_modes = [m for m in ['drive']
                 if m in aligned_mode_utils
                 and aligned_mode_utils[m] is not None
                 and not aligned_mode_utils[m].empty]
    if not all_modes:
        return destinations, modes_used, chosen_scores

    centres = aligned_mode_utils[all_modes[0]].columns
    n_centres = len(centres)

    # Stack: shape (n_agents, n_centres × n_modes)
    stacked  = np.concatenate([aligned_mode_utils[m].values for m in all_modes], axis=1)
    row_sums = stacked.sum(axis=1)
    valid    = row_sums > 0

    if not valid.any():
        return destinations, modes_used

    valid_stacked = stacked[valid]
    valid_idx     = shoppers_idx[valid]

    # Softmax over all (centre × mode) pairs — log-sum-exp trick for stability
    beta   = getattr(config, 'SOFTMAX_BETA', 5.0)
    scaled = (beta * valid_stacked).astype(np.float32)
    
    # Strictly enforce 0-utility being impossible to select by making it -inf
    mask = (valid_stacked <= 0)
    scaled[mask] = -1e10
    
    scaled -= scaled.max(axis=1, keepdims=True)   # subtract row max before exp
    exp_u  = np.exp(scaled)
    # Re-apply mask to exp_u to be safe against float precision
    exp_u[mask] = 0.0
    
    denom = exp_u.sum(axis=1, keepdims=True)
    probs = exp_u / np.where(denom == 0, 1.0, denom)  # guard division by zero

    cumsum = probs.cumsum(axis=1)
    cumsum /= cumsum[:, -1:]              # guard float drift

    rand        = np.random.rand(len(cumsum), 1)
    choice_flat = (rand < cumsum).argmax(axis=1)

    mode_indices   = choice_flat // n_centres
    centre_indices = choice_flat  % n_centres

    chosen_modes   = [all_modes[i] for i in mode_indices]
    chosen_centres = centres[centre_indices]
    
    # Extract the utility score of the actual chosen alternative
    picked_utils = valid_stacked[np.arange(len(choice_flat)), choice_flat]

    destinations.loc[valid_idx]  = chosen_centres.values
    modes_used.loc[valid_idx]    = chosen_modes
    chosen_scores.loc[valid_idx] = picked_utils

    return destinations, modes_used, chosen_scores


def choose_destination(state_df, consumers_mask, grocery_mode_series,
                       utility_matrices, amenity_binary):
    """
    Jointly selects a destination AND transport mode for each grocery shopper.

    Transport mode is no longer pre-drawn from demographic probabilities
    (prob_walk / prob_drive / prob_pt).  Instead, every (retail centre × mode)
    pair is treated as a distinct alternative and one pair is sampled
    proportionally to its utility score.  The mode associated with the winning
    pair is recorded as the agent's transport mode for that trip.

    Parameters
    ----------
    state_df            : agent state DataFrame
    consumers_mask      : boolean mask of agents that need to shop
    grocery_mode_series : Series — 'online' | 'bulk' | 'convenience'
    utility_matrices    : dict with flat keys like 'bulk_walk', 'bulk_drive', etc.

    Returns
    -------
    destinations    : Series (index = state_df.index) → chosen centre ID or None
    transport_modes : Series (index = state_df.index) → 'walk'|'drive'|'pt' or None
    utility_scores  : Series (index = state_df.index) → utility value of the choice
    """
    destinations    = pd.Series(index=state_df.index, data=None, dtype=object)
    transport_modes = pd.Series(index=state_df.index, data=None, dtype=object)
    utility_scores  = pd.Series(index=state_df.index, data=0.0,  dtype=float)

    physical_mask = consumers_mask & grocery_mode_series.isin(['bulk', 'convenience'])
    if not physical_mask.any():
        return destinations, transport_modes

    for gmode in GROCERY_MODES:
        gmode_mask = physical_mask & (grocery_mode_series == gmode)
        if not gmode_mask.any():
            continue

        shoppers_idx = state_df[gmode_mask].index
        agent_ids    = state_df.loc[shoppers_idx, 'AgentID']

        # Build per-mode utility matrices (reindexed + modifiers) for this grocery type
        # We now use the SINGLE averaged utility matrices for all grocery types
        # and apply the amenity filter dynamically.
        filter_col = 'Foodstore' if gmode == 'bulk' else 'Convenience Store'
        
        aligned = {}
        for tmode in TRANSPORT_MODES:
            mat = utility_matrices.get(f'{gmode}_{tmode}')
            if mat is None or mat.empty:
                continue
                
            relevant = mat.reindex(agent_ids).fillna(0).copy()
            relevant = _apply_choice_modifiers(
                relevant, state_df, shoppers_idx, has_postcode=True)
            
            # Apply dynamic store-type filter (Foodstore for bulk, Conv Store for convenience)
            if filter_col in amenity_binary:
                binary_vec = amenity_binary[filter_col].reindex(
                    relevant.columns, fill_value=0)
                relevant = relevant.mul(binary_vec.values, axis=1)
                
            aligned[tmode] = relevant

        dests, modes, scores = _joint_mode_destination_choice(aligned, shoppers_idx)
        destinations.loc[dests.index]    = dests.values
        transport_modes.loc[modes.index] = modes.values
        utility_scores.loc[scores.index] = scores.values

    return destinations, transport_modes, utility_scores


def update_stock_after_shop(state_df, shopping_mask):
    """Replenishes stock to MAX_STOCK_CAPACITY for all agents who shopped."""
    state_df.loc[shopping_mask, 'Stock'] = config.MAX_STOCK_CAPACITY
    return state_df


# ---------------------------------------------------------------------------
# NTS frequency-based trip system
# ---------------------------------------------------------------------------

# Maps each trip type to its daily-probability column, which utility matrix to
# use (grocery-type prefix), and which amenity column(s) to filter on.
TRIP_TYPE_CONFIG = {
    'service': {
        'prob_col':    'prob_trip_service',
        'util_prefix': 'service',                                      
        'filter_col':  ['Personal Service', 'Professional Services'],  # OR logic
    },
    'comparison': {
        'prob_col':    'prob_trip_comparison',
        'util_prefix': 'comparison',
        'filter_col':  'Retail',
    },
    'entertainment': {
        'prob_col':    'prob_trip_entertainment',
        'util_prefix': 'entertainment',
        'filter_col':  'Entertainment',
    },
    'food_drink': {
        'prob_col':    'prob_trip_food_drink',
        'util_prefix': 'food_drink',
        'filter_col':  ['Cafe', 'Restaurant'],                         # OR logic
    },
    'grocery': {
        'prob_col':    None,                                           # Handled by stock logic
        'util_prefix': None,                                           # Handled by bulk/conv logic
        'filter_col':  'Foodstore',
    },
}


def trigger_trips(attributes_df):
    """
    For each NTS trip type, draws a random number per agent against its daily
    probability.  Multiple trip types can fire on the same day for the same agent.

    Returns dict {trip_type -> boolean Series over attributes_df.index}.
    """
    triggered = {}
    rand_matrix = np.random.random((len(attributes_df), len(TRIP_TYPE_CONFIG)))

    for i, (trip_type, cfg) in enumerate(TRIP_TYPE_CONFIG.items()):
        prob_col = cfg['prob_col']
        probs = attributes_df[prob_col].values if prob_col in attributes_df.columns \
                else np.zeros(len(attributes_df))
        triggered[trip_type] = pd.Series(rand_matrix[:, i] < probs,
                                         index=attributes_df.index)

    return triggered


def choose_destination_for_trip(trip_type, triggered_mask, attributes_df,
                                utility_matrices, amenity_binary,
                                transport_mode_series=None):
    """
    Jointly selects a destination AND transport mode per triggered agent for
    a given NTS trip type.

    Transport mode now emerges from the utility landscape rather than being
    pre-drawn from demographic probabilities.  For each agent, every
    (retail centre × mode) pair (filtered to centres that offer the required
    amenity) is sampled proportionally to its utility score.

    `transport_mode_series` is accepted for backward compatibility but ignored;
    mode selection always uses the joint utility approach.

    Returns
    -------
    destinations : Series (index = attributes_df.index) → chosen centre or None
    modes_used   : Series (index = attributes_df.index) → 'walk'|'drive'|'pt' or None
    scores       : Series (index = attributes_df.index) → chosen utility value
    """
    cfg         = TRIP_TYPE_CONFIG[trip_type]
    util_prefix = cfg['util_prefix']
    filter_col  = cfg['filter_col']

    destinations = pd.Series(index=attributes_df.index, data=None, dtype=object)
    modes_used   = pd.Series(index=attributes_df.index, data=None, dtype=object)
    utility_scores = pd.Series(index=attributes_df.index, data=0.0,  dtype=float)

    if not triggered_mask.any():
        return destinations, modes_used, utility_scores

    shoppers_idx = attributes_df[triggered_mask].index
    agent_ids    = (attributes_df.loc[shoppers_idx, 'household'].values
                    if 'household' in attributes_df.columns
                    else shoppers_idx.values)

    def _apply_amenity_filter(mat):
        """Zero-out columns for centres that don't offer the required amenity."""
        if not filter_col:
            return mat
        if isinstance(filter_col, list):
            or_mask = pd.Series(0.0, index=mat.columns)
            for col in filter_col:
                if col in amenity_binary:
                    vec = amenity_binary[col].reindex(mat.columns, fill_value=0)
                    or_mask = or_mask.combine(
                        vec,
                        lambda a, b: 1.0 if (a == 1.0 or b == 1.0) else 0.0)
            return mat.mul(or_mask.values, axis=1)
        else:
            if filter_col in amenity_binary:
                binary_vec = amenity_binary[filter_col].reindex(
                    mat.columns, fill_value=0)
                return mat.mul(binary_vec.values, axis=1)
        return mat

    # Build per-mode utility matrices: reindex → modifiers → amenity filter
    aligned = {}
    for tmode in TRANSPORT_MODES:
        mat = utility_matrices.get(f'{util_prefix}_{tmode}')
        if mat is None or mat.empty:
            continue
        relevant = mat.reindex(agent_ids).fillna(0).copy()
        relevant = _apply_choice_modifiers(
            relevant, attributes_df, shoppers_idx, has_postcode=True)
        relevant = _apply_amenity_filter(relevant)
        aligned[tmode] = relevant

    dests, modes, scores = _joint_mode_destination_choice(aligned, shoppers_idx)
    destinations.loc[dests.index] = dests.values
    modes_used.loc[modes.index]   = modes.values
    utility_scores.loc[scores.index] = scores.values

    return destinations, modes_used, utility_scores


def choose_chained_destination(trip_list, shoppers_idx, grocery_modes,
                               utility_matrices, amenity_binary, attributes_df):
    """
    Selects a single destination and transport mode for an agent performing
    multiple trips (trip chaining).

    The centre must satisfy the requirements of ALL trip types in the list.
    If no such centre exists, returns (None, None, 0.0).

    The utility of a centre is calculated as the AVERAGE utility across all
    the trips being chained.
    """
    destinations   = pd.Series(index=shoppers_idx, data=None, dtype=object)
    modes_used     = pd.Series(index=shoppers_idx, data=None, dtype=object)
    utility_scores = pd.Series(index=shoppers_idx, data=0.0,  dtype=float)

    if not shoppers_idx.any() or not trip_list:
        return destinations, modes_used, utility_scores

    agent_ids = (attributes_df.loc[shoppers_idx, 'household'].values
                 if 'household' in attributes_df.columns
                 else shoppers_idx.values)

    # 1. Resolve trip-specific metadata (util keys and amenity filters)
    trip_metas = []
    for trip in trip_list:
        if trip == 'grocery':
            # For chaining logic, we'll look at the dominant gmode in the provided grocery_modes
            # or handle it per-agent if needed. For now, assume consistent mode per batch
            gmode = grocery_modes.loc[shoppers_idx].iloc[0] if not grocery_modes.loc[shoppers_idx].empty else 'bulk'
            trip_metas.append({
                'prefix': gmode,
                'filter': 'Foodstore' if gmode == 'bulk' else 'Convenience Store'
            })
        else:
            cfg = TRIP_TYPE_CONFIG[trip]
            trip_metas.append({
                'prefix': cfg['util_prefix'],
                'filter': cfg['filter_col']
            })

    # 2. Build Joint Utility Matrices per Transport Mode
    aligned = {}
    for tmode in TRANSPORT_MODES:
        composite_utility = None
        joint_amenity_mask = None

        for meta in trip_metas:
            key = f"{meta['prefix']}_{tmode}"
            mat = utility_matrices.get(key)
            if mat is None or mat.empty:
                continue
                
            rel = mat.reindex(agent_ids).fillna(0).copy()
            if composite_utility is None:
                composite_utility = rel
            else:
                composite_utility += rel
            
            fcol = meta['filter']
            mask = pd.Series(1.0, index=rel.columns)
            if isinstance(fcol, list):
                or_mask = pd.Series(0.0, index=rel.columns)
                for c in fcol:
                    if c in amenity_binary:
                        vec = amenity_binary[c].reindex(rel.columns, fill_value=0)
                        or_mask = or_mask.combine(vec, lambda a, b: 1.0 if (a == 1.0 or b == 1.0) else 0.0)
                mask = or_mask
            elif fcol and fcol in amenity_binary:
                mask = amenity_binary[fcol].reindex(rel.columns, fill_value=0)
            
            if joint_amenity_mask is None:
                joint_amenity_mask = mask
            else:
                joint_amenity_mask = joint_amenity_mask.combine(mask, lambda a, b: 1.0 if (a == 1.0 and b == 1.0) else 0.0)

        if composite_utility is not None:
            composite_utility /= len(trip_metas)
            composite_utility = _apply_choice_modifiers(
                composite_utility, attributes_df, shoppers_idx, has_postcode=True)
            if joint_amenity_mask is not None:
                composite_utility = composite_utility.mul(joint_amenity_mask.values, axis=1)
            aligned[tmode] = composite_utility

    # 3. Sample
    dests, modes, scores = _joint_mode_destination_choice(aligned, shoppers_idx)
    return dests, modes, scores


# ---------------------------------------------------------------------------
# Feedback Loop System
# ---------------------------------------------------------------------------

def apply_feedback(utility_matrix, agent_ids, chosen_centres):
    """
    Applies a simple stochastic feedback loop to a utility matrix in-place.
    For each visit, there is a 50% chance of a good experience (utility +5%)
    and a 50% chance of a bad experience (utility -5%).
    Returns an array of the multipliers applied.
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


# ---------------------------------------------------------------------------
# Retail Centre evaluation
# ---------------------------------------------------------------------------

def evaluate_retail_centres(visits_df, retail_gdf, utility_matrices, amenity_binary,
                             tracker=None, cumulative_boosts=None):
    """
    Evaluates each retail centre against size-peers and spatial-peers.
    Hierarchical Weighting: Small centres compete mostly by distance; 
    Large centres compete mostly by size/functional similarity.

    New Logic:
    - Strike System: A centre only receives a strike if it fails in >=2 categories 
      (or 1 category if it ONLY has 1 category total).
    - Persistent Memory: Requires 2 strikes in a row to trigger a boost.
    - Intervention Ceiling: Boost is capped at 1.30 (30% total).
    """
    if visits_df.empty or retail_gdf.empty:
        return []

    # Initialise trackers if not provided
    if tracker is None: tracker = {}
    if cumulative_boosts is None: cumulative_boosts = {}

    messages = []
    
    # 1. Collate visits by trip type
    visit_counts = visits_df.groupby(['Retail_Centre', 'Trip_Type']).size().unstack(fill_value=0)
    all_centres  = [str(c) for c in retail_gdf.index]
    visit_counts.index = visit_counts.index.astype(str)
    visit_counts = visit_counts.reindex(all_centres, fill_value=0)

    # 2. Track failures across all monitored categories
    trip_types_to_check = {
        'grocery':       [f'{g}_{t}' for g in GROCERY_MODES for t in TRANSPORT_MODES],
        'comparison':    [f'comparison_{t}' for t in TRANSPORT_MODES],
        'service':       [f'service_{t}' for t in TRANSPORT_MODES],
        'entertainment': [f'entertainment_{t}' for t in TRANSPORT_MODES],
        'food_drink':    [f'food_drink_{t}' for t in TRANSPORT_MODES],
    }

    # Centre -> List of categories where it ranks in bottom 10%
    failed_categories = {c: [] for c in all_centres}
    # Centre -> List of categories where it is eligible/participating
    participating_categories = {c: [] for c in all_centres}

    for trip_type, matrix_keys in trip_types_to_check.items():
        counts = visit_counts[trip_type] if trip_type in visit_counts.columns else pd.Series(0, index=all_centres)
        
        cfg = TRIP_TYPE_CONFIG.get(trip_type, {})
        filter_cols = cfg.get('filter_col', [])
        if isinstance(filter_cols, str): filter_cols = [filter_cols]
        
        # Binary mask: which centres can even provide this service?
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
            
            # --- Peer Ranking ---
            s_lower, s_upper = poi_count * 0.8, poi_count * 1.2
            size_peers = retail_gdf[(retail_gdf['Total_POI_'] >= s_lower) & 
                                    (retail_gdf['Total_POI_'] <= s_upper)].index
            size_peers = [str(p) for p in size_peers if str(p) in eval_targets]
            
            if len(size_peers) > 1:
                p_visits = counts[size_peers]
                size_rank = (sum(p_visits < my_visits) + 0.5 * sum(p_visits == my_visits)) / len(p_visits)
            else:
                size_rank = 0.5

            dists = retail_gdf.distance(retail_gdf.loc[centre, 'geometry'])
            spatial_peers = dists[eval_targets].nsmallest(6).index.tolist()
            if centre in spatial_peers: spatial_peers.remove(centre)
            spatial_peers = [str(p) for p in spatial_peers[:5]]
            
            if len(spatial_peers) > 0:
                p_visits = counts[spatial_peers]
                spatial_rank = (sum(p_visits < my_visits) + 0.5 * sum(p_visits == my_visits)) / len(p_visits)
            else:
                spatial_rank = 0.5

            size_weight = np.interp(poi_count, [10, 200], [0.3, 0.7])
            final_rank = (size_rank * size_weight) + (spatial_rank * (1.0 - size_weight))

            if final_rank <= 0.10:
                failed_categories[centre].append(trip_type)

    # 3. Apply Multi-Strike / Persistence Logic
    for centre in all_centres:
        n_participating = len(participating_categories[centre])
        n_failed        = len(failed_categories[centre])
        
        # Strike Logic
        if n_participating == 0:
            continue
            
        # Strike Rule: Fail in >= 2 categories OR (Fail in 1 cat AND only has 1 cat total)
        current_failure = (n_failed >= 2) or (n_participating == 1 and n_failed == 1)
        
        if current_failure:
            tracker[centre] = tracker.get(centre, 0) + 1
            if tracker[centre] >= 2:
                # Trigger Boost
                current_total_boost = cumulative_boosts.get(centre, 1.0)
                if current_total_boost < 1.30:
                    for trip_type, m_keys in trip_types_to_check.items():
                        # Boost attractiveness across all matrices for this centre
                        for m_key in m_keys:
                            matrix = utility_matrices.get(m_key)
                            if matrix is not None and centre in matrix.columns:
                                col_idx = matrix.columns.get_loc(centre)
                                matrix.values[:, col_idx] *= 1.10
                    
                    cumulative_boosts[centre] = current_total_boost * 1.10
                    messages.append(
                        f"Intervention: Centre {centre} boosted (Failing {n_failed}/{n_participating} categories for 2 periods). "
                        f"Total Boost: {cumulative_boosts[centre]:.1f}x")
                    
                else:
                    messages.append(f"Welfare Trap: Centre {centre} hit boost ceiling (1.3x). Intervention stopped.")
                
                # Reset strikes after intervention trigger (win or lose)
                tracker[centre] = 0
        else:
            # Rank recovered or not bad enough -> Reset strikes
            tracker[centre] = 0

    return messages

# ---------------------------------------------------------------------------
# Demographic similarity helper
# ---------------------------------------------------------------------------

def _demographic_similarity_scores(demo_lookup, influencer_ids, target_ids,
                                   age_min, age_range, inc_min, inc_range,
                                   bandwidth):
    """
    Computes a Gaussian similarity score in [0, 1] for each target agent relative
    to the demographic centroid of the influencer group.

    Both age_years and salary_yearly are normalised to [0, 1] using the global
    (sampled-population) min/range supplied by the caller.

    Falls back to 1.0 (full boost) for any target with missing demographic data
    or when the influencer group is empty.

    Parameters
    ----------
    demo_lookup    : DataFrame indexed by household ID, columns [age_years, salary_yearly]
    influencer_ids : array-like of household IDs who visited the trending centre
    target_ids     : Index of household IDs to score
    age_min / age_range / inc_min / inc_range : global normalisation parameters
    bandwidth      : Gaussian decay bandwidth (smaller => steeper drop-off)

    Returns
    -------
    numpy array of float, length = len(target_ids), values in (0, 1]
    """
    inf_data = demo_lookup.reindex(influencer_ids).dropna()
    tgt_data = demo_lookup.reindex(target_ids).fillna(demo_lookup.mean())

    if inf_data.empty:
        return np.ones(len(target_ids))

    # Normalise to [0, 1]
    def _normalise(df):
        out = df.copy()
        if age_range > 0:
            out['age_years']     = (out['age_years']     - age_min) / age_range
        if inc_range > 0:
            out['salary_yearly'] = (out['salary_yearly'] - inc_min) / inc_range
        return out

    inf_norm = _normalise(inf_data)
    tgt_norm = _normalise(tgt_data)

    centroid = inf_norm.mean(axis=0).values               # shape (2,)
    diffs    = tgt_norm.values - centroid                  # shape (n_targets, 2)
    distances = np.sqrt((diffs ** 2).sum(axis=1))          # Euclidean distance

    return np.exp(-distances / max(bandwidth, 1e-6))


# ---------------------------------------------------------------------------
# Spatial + demographic word-of-mouth diffusion
# ---------------------------------------------------------------------------

def apply_spatial_diffusion_bonus(visits_df, attributes_df, utility_matrices,
                                  threshold_visits=3, boost_multiplier=1.05):
    """
    Applies spatial diffusion of preferences by Postcode Sector (first 3 chars).
    Identifies the most popular retail centre per area from recent visits and
    boosts utility for agents living in that area.

    Demographic Homophily (new):
    ----------------------------
    When config.DEMOGRAPHIC_DIFFUSION_WEIGHT > 0 and age_years / salary_yearly
    columns are present, the boost applied to each agent is *scaled* by how
    demographically similar that agent is to the actual visitors of the trending
    centre (the "influencers").

      effective_boost = 1 + (boost_multiplier - 1) * effective_similarity
      effective_similarity = (1 - weight) + weight * gaussian_similarity

    This means:
      - weight=0 : all agents in the area get the full boost (original behaviour)
      - weight=1 : only demographically close agents get the full boost;
                   dissimilar agents receive little or no boost

    Returns a list of summary messages.
    """
    if visits_df.empty or 'Postcode' not in attributes_df.columns:
        return []

    messages      = []
    demo_weight   = getattr(config, 'DEMOGRAPHIC_DIFFUSION_WEIGHT', 0.0)
    demo_bw       = getattr(config, 'DEMOGRAPHIC_BANDWIDTH', 0.5)

    # Build postcode-sector lookup (household_id -> 3-char sector)
    if 'household' in attributes_df.columns:
        df_unique         = attributes_df.drop_duplicates(subset=['household'])
        agent_to_postcode = df_unique.set_index('household')['Postcode'].str[:3]
    else:
        agent_to_postcode = attributes_df['Postcode'].str[:3]
        if agent_to_postcode.index.duplicated().any():
            agent_to_postcode = agent_to_postcode[~agent_to_postcode.index.duplicated()]

    visits_with_pc            = visits_df.copy()
    visits_with_pc['Postcode'] = visits_with_pc['AgentID'].map(agent_to_postcode)
    visits_with_pc             = visits_with_pc.dropna(subset=['Postcode'])
    if visits_with_pc.empty:
        return []

    counts = (visits_with_pc
              .groupby(['Postcode', 'Retail_Centre'])
              .size().reset_index(name='Visits'))

    idx         = counts.groupby('Postcode')['Visits'].idxmax()
    top_centres = counts.loc[idx]
    trending    = top_centres[top_centres['Visits'] >= threshold_visits]

    if trending.empty:
        return []

    # -- Pre-compute demographic lookup (globally normalised) ----------------
    has_demo = (
        demo_weight > 0.0
        and 'age_years'     in attributes_df.columns
        and 'salary_yearly' in attributes_df.columns
    )

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

        # Demographic similarity weights for each affected agent
        if has_demo:
            # Influencers = agents who actually visited this centre in this sector
            visitor_ids = visits_with_pc.loc[
                (visits_with_pc['Postcode'] == pc) &
                (visits_with_pc['Retail_Centre'].astype(str) == centre),
                'AgentID'
            ].unique()

            raw_sim = _demographic_similarity_scores(
                demo_lookup, visitor_ids, affected_agents,
                age_min, age_range, inc_min, inc_range, demo_bw)

            # Blend: weight=0 => flat 1.0 (original); weight=1 => raw_sim
            effective_sim = (1.0 - demo_weight) + demo_weight * raw_sim
        else:
            effective_sim = np.ones(len(affected_agents))

        # Per-agent multiplier: 1 + (boost - 1) * similarity
        per_agent_mults = 1.0 + (boost_multiplier - 1.0) * effective_sim

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
