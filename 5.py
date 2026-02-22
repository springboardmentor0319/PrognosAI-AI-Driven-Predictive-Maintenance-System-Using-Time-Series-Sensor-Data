import pandas as pd

# 1. Manual Data Creation (Demonstrating Series & DataFrames)
# Creating a small demo DataFrame from a dictionary
demo_data = {
    "engine_id": [1, 1, 1],
    "cycle": [1, 2, 3],
    "sensor1": [500, 505, 510]
}
df_demo = pd.DataFrame(demo_data)

# Creating a Series with a custom index
health_score = pd.Series(
    [100, 95, 80],
    index=['Cycle 1', 'Cycle 2', 'Cycle 3'],
    name='Engine Health Score'
)

# 2. Data Acquisition (Reading)
# Loading the main dataset with a defined schema
sensor_names = [f"sensor_{i}" for i in range(1, 22)]
cols = ['unit_id', 'cycle', 'set1', 'set2', 'set3'] + sensor_names

# Using a robust loading method to avoid parsing errors
main_df = pd.read_csv(
    "../dataset/train_FD001.txt", 
    sep=r"\s+", 
    header=None, 
    names=cols
)

# 3. Data Transformation & Health
# Type conversion: ensures unit_id is stored as a float
main_df['unit_id'] = main_df['unit_id'].astype('float64')

# Centering sensor_2 data using a vectorized operation
s2_mean = main_df['sensor_2'].mean()
main_df['sensor_2_adj'] = main_df['sensor_2'] - s2_mean

# 4. Selection & Analytics
# Position-based selection for the first entry
initial_row = main_df.iloc[0]

# Aggregating lifecycle data to find max engine life
lifecycle_report = (
    main_df.groupby('unit_id')['cycle']
    .agg(max_run='max', total='count')
    .reset_index()
    .sort_values(by='max_run', ascending=False)
)

# 5. Data Persistence (Writing)
# Saving the processed results to a CSV file
lifecycle_report.to_csv("engine_lifecycle_summary.csv", index=False)

# 6. Outputs
print("--- Custom Series Example ---")
print(health_score)

print("\n--- Processed Lifecycle (Top 5) ---")
print(lifecycle_report.head())

print("\n--- System Status ---")
print(f"Data successfully exported. Total rows processed: {len(main_df)}")