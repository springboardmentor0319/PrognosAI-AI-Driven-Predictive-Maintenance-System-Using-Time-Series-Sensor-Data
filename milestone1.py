import pandas as pd

# Path is define to  dataset file

data_path = 'train_FD001.txt'

# Define the column names 
columns = ['unit_number', 'cycle', 'setting1', 'setting2', 'setting3']
for i in range(1, 22):
    columns.append(f'sensor_{i}')

# Loading of data useing pandas
df = pd.read_csv(data_path, sep=" ", header=None, names=columns)

print(df.head())

# calculations of Rul
max_cycle = df.groupby('unit_number')['cycle'].max()

# Merge the max_cycle back into the original dataframe
df = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_number', right_index=True)

# Calculate RUL: maximum cycle minus current cycle
df['RUL'] = df['max_cycle'] - df['cycle']

# Drop the temporary 'max_cycle' column
df = df.drop(columns=['max_cycle'])

# 6. Verify the new column
print(df.head())

from sklearn.preprocessing import MinMaxScaler

# 7. Select the columns to normalize (all sensor columns and setting columns)
# We do not want to normalize 'unit_number', 'cycle', or 'RUL'
features = columns[2:] # This selects setting1 through sensor_21

# 8. Initialize the MinMaxScaler
scaler = MinMaxScaler()

# 9. Fit and transform the feature data
df[features] = scaler.fit_transform(df[features])

# 10. Verify the normalization
print(df.head())

import numpy as np

# 11. Define the window size (how many past cycles to look at)
window_size = 30 
# We look at 30 cycles to predict the RUL

# 12. Function to create sequences for each engine
def create_sequences(engine_df, features, window_size):
    X = []
    y = []
    
    # We need at least 'window_size' rows to make a sequence
    if len(engine_df) < window_size:
        return np.array([]), np.array([])
    
    # Create the sequences
    for i in range(len(engine_df) - window_size + 1):
        # Extract the sensor data for the current window
        window = engine_df.iloc[i : i + window_size][features].values
        X.append(window)
        
        # The target RUL is the RUL at the end of this window
        target_rul = engine_df.iloc[i + window_size - 1]['RUL']
        y.append(target_rul)
        
    return np.array(X), np.array(y)

# 13. Apply the function to each engine and combine results
X_all = []
y_all = []

# Group by unit_number and process each engine separately
for unit, engine_df in df.groupby('unit_number'):
    X_engine, y_engine = create_sequences(engine_df, features, window_size)
    if X_engine.size > 0:
        X_all.append(X_engine)
        y_all.append(y_engine)

# 14. Convert lists to numpy arrays (what LSTM models expect)
X_final = np.concatenate(X_all, axis=0)
y_final = np.concatenate(y_all, axis=0)

# 15. Verify the final shape
print("Shape of input sequences (X):", X_final.shape)
print("Shape of target RULs (y):", y_final.shape)
