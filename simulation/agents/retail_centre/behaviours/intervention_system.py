from simulation.core.constants import GROCERY_MODES, TRANSPORT_MODES

def apply_intervention_policy(failed_categories, participating_categories, 
                              utility_matrices, tracker, cumulative_boosts):
    """
    Applies the multi-strike and welfare boost logic based on performance data.
    """
    messages = []
    all_centres = list(participating_categories.keys())

    trip_types_to_check = {
        'grocery':       [f'{g}_{t}' for g in GROCERY_MODES for t in TRANSPORT_MODES],
        'comparison':    [f'comparison_{t}' for t in TRANSPORT_MODES],
        'service':       [f'service_{t}' for t in TRANSPORT_MODES],
        'entertainment': [f'entertainment_{t}' for t in TRANSPORT_MODES],
        'food_drink':    [f'food_drink_{t}' for t in TRANSPORT_MODES],
    }

    for centre in all_centres:
        n_participating = len(participating_categories[centre])
        n_failed        = len(failed_categories[centre])
        
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
                    for _, m_keys in trip_types_to_check.items():
                        for m_key in m_keys:
                            matrix = utility_matrices.get(m_key)
                            if matrix is not None and centre in matrix.columns:
                                col_idx = matrix.columns.get_loc(centre)
                                matrix.values[:, col_idx] *= 1.10
                    
                    cumulative_boosts[centre] = current_total_boost * 1.10
                    messages.append(
                        f"Intervention: Centre {centre} boosted (Failing {n_failed}/{n_participating} categories for 2 periods). "
                        f"Total Boost: {cumulative_boosts[centre]:.2f}x")
                    
                else:
                    messages.append(f"Welfare Trap: Centre {centre} hit boost ceiling (1.3x). Intervention stopped.")
                
                # Reset strikes after intervention trigger
                tracker[centre] = 0
        else:
            # Rank recovered -> Reset strikes
            tracker[centre] = 0

    return messages
