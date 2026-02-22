import pandas as pd

# 1. Pipeline Configuration
# [cite_start]Building a structured sensor map and header list [cite: 254]
sensor_fields = [f"sensor_{i}" for i in range(1, 22)]
column_schema = [
    'unit_id', 'cycle_count', 'setting_1', 'setting_2', 'setting_3'
] + sensor_fields

# 2. Robust Data Ingestion
# [cite_start]Loading with defined schema [cite: 255, 256]
raw_df = pd.read_csv(
    "../dataset/train_FD001.txt", 
    sep=r"\s+", 
    header=None, 
    names=column_schema
)

# 3. Integrity & Type Validation
# [cite_start]Every column in pandas has a dtype (data type) [cite: 258, 259]
# [cite_start]We cast 'unit_id' to float64 as a specific data transformation step [cite: 322]
raw_df['unit_id'] = raw_df['unit_id'].astype('float64')

# Automated Data Health Check
# [cite_start]counts missing values in each column [cite: 329, 330]
data_health = raw_df.isna().sum()
# [cite_start]The dtype property shows the internal data type of a column [cite: 261, 265]
type_map = raw_df.dtypes 

# 4. Feature Engineering: Lifecycle Aggregation
# [cite_start]groupby() allows us to group rows and apply calculations [cite: 7, 8]
engine_metrics = (
    raw_df.groupby('unit_id')['cycle_count']
    .agg(lifespan='max', start_cycle='min', entry_count='count')
    .reset_index()
    .sort_values(by='lifespan', ascending=False)
)

# 5. Diagnostic Outputs
print("--- Schema Integrity (Data Types) ---")
# [cite_start]int64 and float64 are standard internal data types [cite: 262, 267]
print(type_map.head(10))

print("\n--- Missing Value Audit ---")
# [cite_start]isnull() returns True for missing values, allowing filtering [cite: 385, 387]
if data_health.sum() == 0:
    print("Clean Dataset: No missing values (NaN) detected.")
else:
    print(data_health[data_health > 0])

print("\n--- Top 5 Units by Total Lifecycle ---")
print(engine_metrics.head())