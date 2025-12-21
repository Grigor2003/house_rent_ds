# Configuration for data transformation pipeline

# Columns configuration
LOG_SCALING_COLS = ['Size']

CATEGORICAL_COLS = [
    'City', 
    'Furnishing Status', 
    'Tenant Preferred', 
    'day of week posted', 
    'quarter posted',
]

NUMERIC_COLS = ['Floor Level', 'Total Floors', 'BHK', 'Bathroom']

# Interaction combinations configuration
# Each combination: {'columns': [...], 'keep_originals': True/False}
# keep_originals=True: keeps main effects (City, Quarter, Day) + adds interaction
# keep_originals=False: removes originals, keeps only interaction
COMBINATIONS_TO_APPLY = [
    # {'columns': ['City', 'quarter posted','Tenant Preferred'], 'keep_originals': False},
]

# Pretty names for feature display
# Maps raw category values to readable names
PRETTY_NAMES = {
    # Day of week (0=Monday, 6=Sunday)
    'day of week posted': {
        0: 'Mon',
        1: 'Tue', 
        2: 'Wed',
        3: 'Thu',
        4: 'Fri',
        5: 'Sat',
        6: 'Sun',
        '0': 'Mon',
        '1': 'Tue',
        '2': 'Wed', 
        '3': 'Thu',
        '4': 'Fri',
        '5': 'Sat',
        '6': 'Sun',
    },
    # Quarter
    'quarter posted': {
        1: 'Q1',
        2: 'Q2',
        3: 'Q3', 
        4: 'Q4',
        '1': 'Q1',
        '2': 'Q2',
        '3': 'Q3',
        '4': 'Q4',
    }
}

# Short column names for display
COLUMN_SHORT_NAMES = {
    'day of week posted': 'Day',
    'quarter posted': 'Quarter',
    'Furnishing Status': 'Furnishing',
    'Tenant Preferred': 'Tenant',
}

