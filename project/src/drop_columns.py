import pandas as pd

# Load merged data
df = pd.read_csv('../data/processed/merged_data.csv')

print(f"Original shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Columns to drop (redundant features)
cols_to_drop = [
    'temperature_2m',        # Duplicate of tavg
    'wind_speed_10m',        # Duplicate of wspd
    'pressure_msl',          # Duplicate of pres
    'wind_direction_10m',    # Not meaningful after daily aggregation
    'fire_frp_mean',         # Redundant with fire_frp_total
    'fire_brightness_avg',   # Less informative than FRP
    'pm2_5_min',             # Less informative than mean and max
]

# Drop redundant columns
df_clean = df.drop(columns=cols_to_drop)

print(f"Cleaned shape: {df_clean.shape}")
print(f"\nRemaining columns ({len(df_clean.columns)}):")
for i, col in enumerate(df_clean.columns, 1):
    print(f"{i}. {col}")

# Check for any remaining issues
print(f"\n✅ Missing values: {df_clean.isnull().sum().sum()}")
print(f"✅ Duplicate rows: {df_clean.duplicated().sum()}")

# Save cleaned dataset
df_clean.to_csv('../data/processed/merged_data_clean.csv', index=False)
print(f"\n✅ Saved to: data/processed/merged_data_clean.csv")