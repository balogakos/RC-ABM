import config
import numpy as np

def consume(state_df):
    """Reduces each agent's stock by their daily consumption rate (floor 0)."""
    state_df['Stock'] = (state_df['Stock'] - state_df['Consumption_Rate']).clip(lower=0.0)
    return state_df

def update_stock_after_shop(state_df, shopping_mask, shopping_modes):
    """
    Replenishes stock based on shopping mode (Bulk/Online vs. Convenience).
    Adds a random quantity within the configured ranges.
    """
    if not shopping_mask.any():
        return state_df
        
    shoppers = state_df[shopping_mask]
    modes = shopping_modes[shopping_mask]
    
    # Identify trip types
    is_bulk = modes.isin(['bulk', 'online'])
    is_con  = modes == 'convenience'
    
    # Initialize gain array
    gains = np.zeros(len(shoppers))
    
    # Apply random refill factor for bulk
    if is_bulk.any():
        bulk_range = config.BULK_REFILL_RANGE
        ratios = np.random.uniform(bulk_range[0], bulk_range[1], size=is_bulk.sum())
        gains[is_bulk] = shoppers.loc[is_bulk, 'Capacity'] * ratios
        
    # Apply random refill factor for convenience
    if is_con.any():
        con_range = config.CONVENIENCE_REFILL_RANGE
        ratios = np.random.uniform(con_range[0], con_range[1], size=is_con.sum())
        gains[is_con] = shoppers.loc[is_con, 'Capacity'] * ratios
        
    # Update stock and clip at Capacity
    state_df.loc[shopping_mask, 'Stock'] = (state_df.loc[shopping_mask, 'Stock'] + gains).clip(upper=state_df.loc[shopping_mask, 'Capacity'])
    
    return state_df
