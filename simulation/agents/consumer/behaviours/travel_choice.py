import pandas as pd
from simulation.core.constants import TRANSPORT_MODES, GROCERY_MODES, TRIP_TYPE_CONFIG
from simulation.core.utility_engine import (
    apply_choice_modifiers, 
    joint_mode_destination_choice
)

def choose_destination(state_df, consumers_mask, grocery_mode_series,
                       utility_matrices, amenity_binary):
    """
    Jointly selects a destination AND transport mode for each grocery shopper.
    """
    destinations    = pd.Series(index=state_df.index, data=None, dtype=object)
    transport_modes = pd.Series(index=state_df.index, data=None, dtype=object)
    utility_scores  = pd.Series(index=state_df.index, data=0.0,  dtype=float)

    physical_mask = consumers_mask & grocery_mode_series.isin(['bulk', 'convenience'])
    if not physical_mask.any():
        return destinations, transport_modes, utility_scores

    for gmode in GROCERY_MODES:
        gmode_mask = physical_mask & (grocery_mode_series == gmode)
        if not gmode_mask.any():
            continue

        shoppers_idx = state_df[gmode_mask].index
        agent_ids    = state_df.loc[shoppers_idx, 'AgentID']

        filter_col = 'Foodstore' if gmode == 'bulk' else 'Convenience Store'
        
        aligned = {}
        for tmode in TRANSPORT_MODES:
            mat = utility_matrices.get(f'{gmode}_{tmode}')
            if mat is None or mat.empty:
                continue
                
            relevant = mat.reindex(agent_ids).fillna(0).copy()
            relevant = apply_choice_modifiers(
                relevant, state_df, shoppers_idx, has_postcode=True)
            
            if filter_col in amenity_binary:
                binary_vec = amenity_binary[filter_col].reindex(
                    relevant.columns, fill_value=0)
                relevant = relevant.mul(binary_vec.values, axis=1)
                
            aligned[tmode] = relevant

        dests, modes, scores = joint_mode_destination_choice(aligned, shoppers_idx)
        destinations.loc[dests.index]    = dests.values
        transport_modes.loc[modes.index] = modes.values
        utility_scores.loc[scores.index] = scores.values

    return destinations, transport_modes, utility_scores


def choose_destination_for_trip(trip_type, triggered_mask, attributes_df,
                                utility_matrices, amenity_binary):
    """
    Jointly selects a destination AND transport mode for a given NTS trip type.
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

    aligned = {}
    for tmode in TRANSPORT_MODES:
        mat = utility_matrices.get(f'{util_prefix}_{tmode}')
        if mat is None or mat.empty:
            continue
        relevant = mat.reindex(agent_ids).fillna(0).copy()
        relevant = apply_choice_modifiers(
            relevant, attributes_df, shoppers_idx, has_postcode=True)
        relevant = _apply_amenity_filter(relevant)
        aligned[tmode] = relevant

    dests, modes, scores = joint_mode_destination_choice(aligned, shoppers_idx)
    destinations.loc[dests.index] = dests.values
    modes_used.loc[modes.index]   = modes.values
    utility_scores.loc[scores.index] = scores.values

    return destinations, modes_used, utility_scores


def choose_chained_destination(trip_list, shoppers_idx, grocery_modes,
                               utility_matrices, amenity_binary, attributes_df):
    """
    Selects a single destination and transport mode for an agent performing
    multiple trips (trip chaining).
    """
    destinations   = pd.Series(index=shoppers_idx, data=None, dtype=object)
    modes_used     = pd.Series(index=shoppers_idx, data=None, dtype=object)
    utility_scores = pd.Series(index=shoppers_idx, data=0.0,  dtype=float)

    if not shoppers_idx.any() or not trip_list:
        return destinations, modes_used, utility_scores

    agent_ids = (attributes_df.loc[shoppers_idx, 'household'].values
                 if 'household' in attributes_df.columns
                 else shoppers_idx.values)

    trip_metas = []
    for trip in trip_list:
        if trip == 'grocery':
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
            composite_utility = apply_choice_modifiers(
                composite_utility, attributes_df, shoppers_idx, has_postcode=True)
            if joint_amenity_mask is not None:
                composite_utility = composite_utility.mul(joint_amenity_mask.values, axis=1)
            aligned[tmode] = composite_utility

    dests, modes, scores = joint_mode_destination_choice(aligned, shoppers_idx)
    return dests, modes, scores
