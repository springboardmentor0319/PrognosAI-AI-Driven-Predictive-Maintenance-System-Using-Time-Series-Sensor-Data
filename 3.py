import pandas as pd

# 1. Dataset Configuration
# Constructing a dynamic schema for the CMAPSS dataset
sensor_ids = [f"sensor_{i}" for i in range(1, 22)]
data_schema = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + sensor_ids

# 2. Data Acquisition
# [cite_start]Loading raw text data into a structured DataFrame [cite: 255, 256]
raw_data = pd.read_csv(
    "../dataset/train_FD001.txt", 
    sep=r"\s+", 
    header=None, 
    names=data_schema
)

# 3. Data Integrity & Type Management
# [cite_start]Performing a global health check for missing values [cite: 329, 330]
# [cite_start]Verifying internal data types (dtypes) [cite: 265]
health_check = raw_data.isna().sum()

# [cite_start]Explicitly casting unit_id to float64 as a transformation step [cite: 322]
raw_data['unit_id'] = raw_data['unit_id'].astype('float64')

# 4. Advanced Data Selection
# Using iloc for position-based indexing (getting the first record)
first_entry = raw_data.iloc[0]

# Implementing label-based slicing with loc
initial_subset = raw_data.loc[0:9, ['unit_id', 'cycle', 'sensor_2']]

# 5. Feature Engineering: Engine Lifecycle Analysis
# [cite_start]Aggregating cycle counts to determine the 'EndOfLife' for each unit [cite: 82, 183]
# [cite_start]Sorting units by their maximum operational cycle [cite: 184]
lifecycle_summary = (
    raw_data.groupby('unit_id')['cycle']
    .agg(total_cycles='count', start_cycle='min', end_cycle='max')
    .reset_index()
    .sort_values(by='end_cycle', ascending=False)
)

# 6. Reporting Diagnostics
print("--- Data Schema Preview ---")
print(raw_data.dtypes.head(10))

print("\n--- Top 5 Units by Operational Lifespan ---")
print(lifecycle_summary.head())