import pandas as pd

# 1. Pipeline Setup & Data Loading
# Creating a dynamic schema for the engine dataset [cite: 3, 4, 5]
sensor_fields = [f"sensor_{i}" for i in range(1, 22)]
column_names = ['unit_id', 'cycle', 'set1', 'set2', 'set3'] + sensor_fields

# Load the dataset using whitespace as the separator [cite: 4, 5]
df = pd.read_csv("../dataset/train_FD001.txt", sep=r"\s+", header=None, names=column_names)

# 2. Data Health & Type Casting
# Checking for missing values (NaN) across all features [cite: 5]
null_summary = df.isna().sum()

# Converting unit_id to float64 as a specific data requirement [cite: 5]
df['unit_id'] = df['unit_id'].astype('float64')

# 3. Summary Statistics & Mapping
# Describing sensor_2 and calculating its mean 
s2_stats = df['sensor_2'].describe()
s2_mean = df['sensor_2'].mean()

# Normalizing sensor_2 by centering it around the mean 
# Using a lambda function with map() for efficient processing 
df['sensor_2_centered'] = df['sensor_2'].map(lambda val: val - s2_mean)

# 4. Advanced Selection & Grouping
# Using iloc for the first entry and loc for specific subsets 
first_entry = df.iloc[0]
subset_view = df.loc[0:4, ['unit_id', 'cycle', 'sensor_2']]

# Groupwise analysis to determine engine lifespans [cite: 4]
# Sorting by end_cycle in descending order [cite: 4]
engine_lifespans = (
    df.groupby('unit_id')['cycle']
    .agg(total_cycles='count', end_cycle='max')
    .reset_index()
    .sort_values(by='end_cycle', ascending=False)
)

# 5. Assignment & Status
# Creating a new 'status' column for tracking 
df['status'] = 'active'

# 6. Diagnostic Reports
print("--- Unit Lifespan Overview ---")
print(engine_lifespans.head())

print("\n--- Sensor 2 Normalization Summary ---")
print(s2_stats)

print("\n--- Data Health Check ---")
if null_summary.sum() == 0:
    print("Database is clean: No NaN values found.")
else:
    print(null_summary[null_summary > 0])