# Shared constants and configurations for trip types

TRANSPORT_MODES = ['walk', 'drive', 'pt']
GROCERY_MODES   = ['bulk', 'convenience']

TRIP_TYPE_CONFIG = {
    'service': {
        'prob_col':    'prob_trip_service',
        'util_prefix': 'service',                                      
        'filter_col':  ['Personal Service', 'Professional Services'],
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
        'filter_col':  ['Cafe', 'Restaurant'],
    },
    'grocery': {
        'prob_col':    None,
        'util_prefix': None,
        'filter_col':  'Foodstore',
    },
}
