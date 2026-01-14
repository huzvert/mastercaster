import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: Loading and Aggregating AQ Data (Hourly → Daily)")
print("=" * 60)

# Load hourly AQ data
df_aq = pd.read_csv('../data/raw/aqi.csv')
df_aq['time'] = pd.to_datetime(df_aq['time'])
df_aq['date'] = df_aq['time'].dt.date

print(f"Loaded {len(df_aq)} hourly AQ records")
print(f"Date range: {df_aq['time'].min()} to {df_aq['time'].max()}")

# Aggregate to daily level
df_aq_daily = df_aq.groupby('date').agg({
    # Target variable
    'pm2_5': ['mean', 'max', 'min', 'std'],
    
    # Other pollutants
    'pm10': 'mean',
    'ozone': 'max',  # Peak ozone matters for health
    'nitrogen_dioxide': 'mean',
    'sulphur_dioxide': 'mean',
    'carbon_monoxide': 'mean',
    
    # Weather from AQ dataset
    'temperature_2m': 'mean',
    'relative_humidity_2m': 'mean',
    'wind_speed_10m': 'mean',
    'wind_direction_10m': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),  # Most common direction
    'pressure_msl': 'mean'
}).reset_index()

# Flatten column names
df_aq_daily.columns = ['date', 'pm2_5_mean', 'pm2_5_max', 'pm2_5_min', 'pm2_5_std',
                        'pm10', 'ozone', 'nitrogen_dioxide', 'sulphur_dioxide', 
                        'carbon_monoxide', 'temperature_2m', 'relative_humidity_2m',
                        'wind_speed_10m', 'wind_direction_10m', 'pressure_msl']

print(f"\n✅ Aggregated to {len(df_aq_daily)} daily records")
print(f"\nDaily AQ data preview:")
print(df_aq_daily.head())

print("\n" + "=" * 60)
print("STEP 2: Loading Additional Weather Data")
print("=" * 60)

# Load daily weather data
df_weather = pd.read_csv('../data/raw/weather.csv')
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.date

# Drop columns with all nulls
df_weather = df_weather.drop(columns=['snow', 'tsun'], errors='ignore')

print(f"Loaded {len(df_weather)} daily weather records")
print(f"\nWeather data columns: {df_weather.columns.tolist()}")

print("\n" + "=" * 60)
print("STEP 3: Processing Fire Data")
print("=" * 60)

# Load fire data
df_fire = pd.read_csv('../data/raw/fire.csv')
df_fire['acq_date'] = pd.to_datetime(df_fire['acq_date'])

print(f"Loaded {len(df_fire)} fire detections")
print(f"Date range: {df_fire['acq_date'].min()} to {df_fire['acq_date'].max()}")

# Aggregate fires to daily
df_fire_daily = df_fire.groupby('acq_date').agg({
    'latitude': 'count',          # Fire count
    'brightness': 'mean',         # Average brightness
    'frp': ['mean', 'max', 'sum'] # Fire Radiative Power
}).reset_index()

df_fire_daily.columns = ['date', 'fire_count', 'fire_brightness_avg', 
                         'fire_frp_mean', 'fire_frp_max', 'fire_frp_total']

# Convert date to same type as other dataframes
df_fire_daily['date'] = df_fire_daily['date'].dt.date

# Create complete date range and fill missing days with 0
date_range = pd.date_range(start='2023-01-01', end='2025-05-01', freq='D')
df_dates = pd.DataFrame({'date': date_range.date})

df_fire_complete = df_dates.merge(df_fire_daily, on='date', how='left')
df_fire_complete = df_fire_complete.fillna(0)  # Days with no fires = 0

print(f"\n✅ Daily fire data: {len(df_fire_complete)} days")
print(f"Days with fires: {(df_fire_complete['fire_count'] > 0).sum()}")
print(f"Days without fires: {(df_fire_complete['fire_count'] == 0).sum()}")

print("\n" + "=" * 60)
print("STEP 4: Merging All Datasets")
print("=" * 60)

# Merge all three datasets
df_merged = df_aq_daily.copy()

# Merge weather
df_merged = df_merged.merge(df_weather, on='date', how='inner')
print(f"After merging weather: {len(df_merged)} rows")

# Merge fire data
df_merged = df_merged.merge(df_fire_complete, on='date', how='inner')
print(f"After merging fire data: {len(df_merged)} rows")

print("\n" + "=" * 60)
print("STEP 5: Data Quality Checks")
print("=" * 60)

# Check for missing values
print("\nMissing values per column:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])

# Handle missing values in precipitation (forward fill)
if 'prcp' in df_merged.columns:
    df_merged['prcp'] = df_merged['prcp'].fillna(0)  # Assume no rain if missing

# Drop columns with excessive nulls
null_threshold = 0.5
cols_to_drop = df_merged.columns[df_merged.isnull().sum() / len(df_merged) > null_threshold]
if len(cols_to_drop) > 0:
    print(f"\nDropping columns with >{null_threshold*100}% nulls: {cols_to_drop.tolist()}")
    df_merged = df_merged.drop(columns=cols_to_drop)

# Final null check
print(f"\nFinal missing values: {df_merged.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("STEP 6: Final Dataset Summary")
print("=" * 60)

print(f"\n✅ FINAL MERGED DATASET:")
print(f"   Rows: {len(df_merged)}")
print(f"   Columns: {len(df_merged.columns)}")
print(f"   Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")

print(f"\nColumn list:")
for i, col in enumerate(df_merged.columns, 1):
    print(f"   {i}. {col}")

print(f"\nTarget variable (pm2_5_mean) statistics:")
print(df_merged['pm2_5_mean'].describe())

print(f"\nFirst 5 rows:")
print(df_merged.head())

# Save merged dataset
output_file = '../data/processed/merged_data.csv'
df_merged.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")

print("\n" + "=" * 60)
print("MERGE COMPLETE! Ready for feature engineering.")
print("=" * 60)