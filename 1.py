import pandas as pd

# 1. Dynamic Data Initialization
# Using list comprehension to build the sensor map 
sensor_labels = [f"sensor_{i}" for i in range(1, 22)]
header_labels = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + sensor_labels

# Loading the dataset with regex separator for robustness [cite: 4]
raw_data = pd.read_csv("../dataset/train_FD001.txt", sep=r"\s+", header=None, names=header_labels)

# 2. Comprehensive Engine Lifecycle Analysis
# Combining grouping and aggregation into a single descriptive summary [cite: 82]
# This replaces standard count/min/max with a more analytical view [cite: 79, 83, 85]
engine_lifecycle = (
    raw_data.groupby('engine_id')['cycle']
    .agg(total_records='count', start_point='min', end_point='max')
    .reset_index()
)

# 3. Granular Sensor Performance Tracking
# Grouping by multiple dimensions to see sensor behavior over time [cite: 119]
# We use .mean() to find the average sensor signature per engine cycle [cite: 119, 143]
sensor_metrics = (
    raw_data.groupby(['engine_id', 'cycle'])['sensor_2']
    .mean()
    .reset_index(name='avg_sensor_2')
)

# 4. Identifying High-Usage Units (Sorting)
# Sorting the engine life data to highlight the most utilized assets [cite: 182, 184]
# Ascending=False helps identify the engines with the highest 'end_point' [cite: 184]
prioritized_engines = engine_lifecycle.sort_values(by='end_point', ascending=False)

# 5. Output Preview
print("--- Unit Lifecycle Metrics ---")
print(engine_lifecycle.head())

print("\n--- Top 5 Engines by Operational Life ---")
print(prioritized_engines.head())