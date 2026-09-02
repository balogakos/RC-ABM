import pandas as pd
import numpy as np
import config
from simulation.core.constants import TRIP_TYPE_CONFIG

def check_shopping_need(state_df):
    """Returns a boolean Series: True where Stock < Shopping_Threshold."""
    return state_df['Stock'] < state_df['Shopping_Threshold']

def choose_mode(attributes_df, shopping_mask):
    """
    Draws a grocery shopping mode (online / bulk / convenience) for each agent
    whose index appears in shopping_mask.
    Probabilities are pre-normalised at load time, so no re-normalisation needed.
    """
    modes = pd.Series(index=attributes_df.index, data=None, dtype=object)
    shoppers = attributes_df[shopping_mask]
    if shoppers.empty:
        return modes

    probs  = shoppers[['prob_online', 'prob_bulk', 'prob_convenience']].values
    rand   = np.random.random(len(shoppers))
    cumsum = probs.cumsum(axis=1)

    chosen = np.where(rand < cumsum[:, 0], 'online',
             np.where(rand < cumsum[:, 1], 'bulk', 'convenience'))
    modes.iloc[shoppers.index.get_indexer(shoppers.index)] = chosen
    modes.update(pd.Series(chosen, index=shoppers.index))
    return modes

def trigger_trips(attributes_df, multiplier=1.0):
    """
    For each NTS trip type, draws a random number per agent against its daily
    probability (multiplied by a global daily pulse).
    """
    triggered = {}
    rand_matrix = np.random.random((len(attributes_df), len(TRIP_TYPE_CONFIG)))

    for i, (trip_type, cfg) in enumerate(TRIP_TYPE_CONFIG.items()):
        prob_col = cfg['prob_col']
        if prob_col:
            probs = attributes_df[prob_col].values if prob_col in attributes_df.columns \
                    else np.zeros(len(attributes_df))
            
            # Apply temporal variability
            m = multiplier.get(trip_type, 1.0) if isinstance(multiplier, dict) else multiplier
            effective_probs = np.clip(probs * m, 0, 1)
            
            triggered[trip_type] = pd.Series(rand_matrix[:, i] < effective_probs,
                                             index=attributes_df.index)
        else:
            # grocery is handled by stock logic
            triggered[trip_type] = pd.Series(False, index=attributes_df.index)

    return triggered
