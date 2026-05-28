import config
from simulation.core.constants import GROCERY_MODES, TRANSPORT_MODES

def apply_intervention_policy(failed_categories, participating_categories, 
                              utility_matrices, tracker, cumulative_boosts, death_spirals):
    """
    Applies the multi-strike, welfare boost, and death spiral logic based on performance data.
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
        # If centre is in a death spiral, it gets penalized every evaluation period
        if death_spirals.get(centre, False):
            decline_val = getattr(config, 'RETAIL_DECLINE_PENALTY', 0.965)  # -3.5%
            for _, m_keys in trip_types_to_check.items():
                for m_key in m_keys:
                    matrix = utility_matrices.get(m_key)
                    if matrix is not None and centre in matrix.columns:
                        col_idx = matrix.columns.get_loc(centre)
                        matrix.values[:, col_idx] *= decline_val
            
            cumulative_boosts[centre] = cumulative_boosts.get(centre, 1.0) * decline_val
            messages.append(
                f"Death Spiral: Centre {centre} utility penalized by {((1 - decline_val)*100):.1f}% (Total factor: {cumulative_boosts[centre]:.2f}x)."
            )
            continue

        n_participating = len(participating_categories[centre])
        n_failed        = len(failed_categories[centre])
        
        if n_participating == 0:
            continue
            
        # Strike Rule: Fail in >= 2 categories OR (Fail in 1 cat AND only has 1 cat total)
        current_failure = (n_failed >= 2) or (n_participating == 1 and n_failed == 1)
        
        if current_failure:
            tracker[centre] = tracker.get(centre, 0) + 1
            
            # Check 3 consecutive strikes first
            if tracker[centre] >= 3:
                death_spirals[centre] = True
                messages.append(
                    f"Death Spiral: Centre {centre} accumulated 3 consecutive strikes. Entering permanent structural decline."
                )
                tracker[centre] = 0
            
            # Check 2 strikes (intervention attempt)
            elif tracker[centre] == 2:
                # 85/15 split: 15% success probability (Dolega et al., 2026)
                import numpy as np
                success_prob = getattr(config, 'INTERVENTION_SUCCESS_PROBABILITY', 0.15)
                is_successful = np.random.rand() < success_prob

                if is_successful:
                    # Trigger Boost
                    current_total_boost = cumulative_boosts.get(centre, 1.0)
                    ceiling = getattr(config, 'RETAIL_BOOST_CEILING', 1.30)
                    boost_val = getattr(config, 'RETAIL_INTERVENTION_BOOST', 1.10)

                    if current_total_boost < ceiling:
                        for _, m_keys in trip_types_to_check.items():
                            for m_key in m_keys:
                                matrix = utility_matrices.get(m_key)
                                if matrix is not None and centre in matrix.columns:
                                    col_idx = matrix.columns.get_loc(centre)
                                    matrix.values[:, col_idx] *= boost_val
                        
                        cumulative_boosts[centre] = current_total_boost * boost_val
                        messages.append(
                            f"Intervention Success: Centre {centre} boosted (Failing {n_failed}/{n_participating} categories for 2 periods). "
                            f"Total Boost: {cumulative_boosts[centre]:.2f}x")
                    else:
                        messages.append(f"Welfare Trap: Centre {centre} hit boost ceiling (1.3x). Intervention stopped.")
                    # Reset strikes after successful intervention
                    tracker[centre] = 0
                else:
                    death_spirals[centre] = True
                    messages.append(
                        f"Intervention Failure: Centre {centre} was eligible for rescue (Failing {n_failed}/{n_participating} categories for 2 periods), "
                        f"but the intervention failed (85% failure probability). Entering permanent structural decline."
                    )
                    # Reset strikes
                    tracker[centre] = 0
        else:
            # Rank recovered -> Reset strikes
            tracker[centre] = 0

    return messages
