import config

def consume(state_df):
    """Reduces each agent's stock by their daily consumption rate (floor 0)."""
    state_df['Stock'] = (state_df['Stock'] - state_df['Consumption_Rate']).clip(lower=0.0)
    return state_df

def update_stock_after_shop(state_df, shopping_mask):
    """Replenishes stock to MAX_STOCK_CAPACITY for all agents who shopped."""
    state_df.loc[shopping_mask, 'Stock'] = config.MAX_STOCK_CAPACITY
    return state_df
