import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("FEATURE ENGINEERING FOR LAHORE PM2.5 FORECASTING")
print("=" * 70)

# Load cleaned data
df = pd.read_csv('../data/processed/merged_data_clean.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n📊 Initial Dataset Shape: {df.shape}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# STEP 1: TEMPORAL FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Creating Temporal Features")
print("=" * 70)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Seasonal indicators (Lahore-specific)
df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)  # Dec-Feb: Worst pollution
df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)   # Mar-May
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)   # Jun-Aug: Monsoon
df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int) # Sep-Nov: Post-monsoon

# Smog season (Oct 15 - Jan 31)
df['is_smog_season'] = ((df['month'].isin([11, 12, 1])) | 
                        ((df['month'] == 10) & (df['date'].dt.day >= 15))).astype(int)

# Crop burning season (Oct 15 - Nov 15)
df['is_crop_burning'] = (((df['month'] == 10) & (df['date'].dt.day >= 15)) | 
                         ((df['month'] == 11) & (df['date'].dt.day <= 15))).astype(int)

# Cyclical encoding for month and day_of_week
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print(f"✅ Created 18 temporal features")

# ============================================================================
# STEP 2: LAG FEATURES (Previous days' values)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Creating Lag Features")
print("=" * 70)

lag_days = [1, 7, 14, 30]
lag_features = ['pm2_5_mean', 'pm2_5_max', 'pm2_5_std', 'pm10', 'ozone', 
                'nitrogen_dioxide', 'fire_count', 'fire_frp_total']

lag_count = 0
for feature in lag_features:
    for lag in lag_days:
        col_name = f'{feature}_lag{lag}'
        df[col_name] = df[feature].shift(lag)
        lag_count += 1

print(f"✅ Created {lag_count} lag features")

# ============================================================================
# STEP 3: ROLLING STATISTICS (Moving averages)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating Rolling Window Features")
print("=" * 70)

rolling_windows = [3, 7, 30]
rolling_features = ['pm2_5_mean', 'pm10', 'ozone', 'tavg', 'relative_humidity_2m', 
                    'fire_count', 'fire_frp_total']

rolling_count = 0
for feature in rolling_features:
    for window in rolling_windows:
        # Rolling mean
        col_name = f'{feature}_rolling_mean_{window}d'
        df[col_name] = df[feature].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Rolling std (only for key features)
        if feature in ['pm2_5_mean', 'pm10', 'tavg']:
            col_name_std = f'{feature}_rolling_std_{window}d'
            df[col_name_std] = df[feature].shift(1).rolling(window=window, min_periods=1).std()
            rolling_count += 1
        
        rolling_count += 1

print(f"✅ Created {rolling_count} rolling window features")

# ============================================================================
# STEP 4: DERIVED WEATHER FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Creating Derived Weather Features")
print("=" * 70)

# Temperature range (diurnal variation)
df['temp_range'] = df['tmax'] - df['tmin']

# Days since last rain
df['days_since_rain'] = 0
no_rain_count = 0
for i in range(len(df)):
    if df.loc[i, 'prcp'] > 0:
        no_rain_count = 0
    else:
        no_rain_count += 1
    df.loc[i, 'days_since_rain'] = no_rain_count

# Cumulative precipitation (last 7 days)
df['prcp_cumsum_7d'] = df['prcp'].rolling(window=7, min_periods=1).sum()

# Temperature anomaly (deviation from 30-day average)
df['tavg_anomaly'] = df['tavg'] - df['tavg'].shift(1).rolling(window=30, min_periods=1).mean()

# Wind chill effect (temperature felt with wind)
df['wind_chill'] = df['tavg'] - (df['wspd'] * 2)  # Simplified wind chill

print(f"✅ Created 6 derived weather features")

# ============================================================================
# STEP 5: INTERACTION FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Creating Interaction Features")
print("=" * 70)

# Temperature × Humidity (affects particle formation)
df['temp_humidity_interaction'] = df['tavg'] * df['relative_humidity_2m']

# Wind speed × Fire count (fire smoke dispersal)
df['wind_fire_interaction'] = df['wspd'] * df['fire_count']

# PM10 × Ozone (secondary pollutant formation)
df['pm10_ozone_interaction'] = df['pm10'] * df['ozone']

# Temperature inversion indicator (low wind + high humidity + winter)
df['inversion_risk'] = ((df['wspd'] < 1) & 
                        (df['relative_humidity_2m'] > 80) & 
                        (df['is_winter'] == 1)).astype(int)

# Fire intensity during crop burning season
df['crop_fire_intensity'] = df['fire_frp_total'] * df['is_crop_burning']

print(f"✅ Created 5 interaction features")

# ============================================================================
# STEP 6: POLLUTANT RATIOS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Creating Pollutant Ratio Features")
print("=" * 70)

# PM2.5 to PM10 ratio (fine vs coarse particles)
df['pm2_5_pm10_ratio'] = df['pm2_5_mean'] / (df['pm10'] + 1)  # +1 to avoid division by zero

# NO2 to CO ratio (traffic vs combustion indicator)
df['no2_co_ratio'] = df['nitrogen_dioxide'] / (df['carbon_monoxide'] + 1)

print(f"✅ Created 2 pollutant ratio features")

# ============================================================================
# STEP 7: FIRE-RELATED FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Creating Fire-Related Features")
print("=" * 70)

# Binary fire indicator
df['has_fire'] = (df['fire_count'] > 0).astype(int)

# Cumulative fire count (last 3 days)
df['fire_count_cumsum_3d'] = df['fire_count'].rolling(window=3, min_periods=1).sum()

# Fire intensity per fire (average)
df['fire_intensity_per_fire'] = df['fire_frp_total'] / (df['fire_count'] + 1)

print(f"✅ Created 3 fire-related features")

# ============================================================================
# STEP 8: EXTREME POLLUTION DETECTION FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Creating Extreme Pollution Detection Features")
print("=" * 70)

# Feature 1: Strong temperature inversion (low wind + conditions for inversion)
# Temperature inversion traps pollutants near ground
df['strong_inversion'] = (
    (df['wspd'] < df['wspd'].quantile(0.25)) & 
    (df['relative_humidity_2m'] > df['relative_humidity_2m'].quantile(0.75))
).astype(int)

# Feature 2: Fire + weather conditions conducive to smoke spread
df['fire_weather_risk'] = (
    (df['fire_count'] > 0) & 
    (df['wspd'] < df['wspd'].quantile(0.33))
).astype(int)

# Feature 3: High pollution streak (consecutive days above threshold)
threshold_high = df['pm2_5_mean_rolling_mean_3d'].quantile(0.75)
df['high_pollution_streak'] = (
    df['pm2_5_mean_rolling_mean_3d'] > threshold_high
).astype(int)

# Feature 4: Stagnant air conditions (very low wind + high humidity)
df['stagnant_air'] = (
    (df['wspd'] < df['wspd'].quantile(0.15)) & 
    (df['relative_humidity_2m'] > df['relative_humidity_2m'].quantile(0.85))
).astype(int)

# Feature 5: Winter smog season risk (Nov-Jan + stagnant conditions)
df['winter_smog_risk'] = (
    (df['is_smog_season'] == 1) & 
    (df['wspd'] < df['wspd'].quantile(0.25)) & 
    (df['relative_humidity_2m'] > df['relative_humidity_2m'].quantile(0.70))
).astype(int)

# Feature 6: Humidity-wind trap (high humidity + low wind combination)
df['humidity_wind_trap'] = df['temp_humidity_interaction'] * (1 / (df['wspd'] + 0.1))

print(f"Extreme pollution detection features created:")
print(f"  1. strong_inversion: {df['strong_inversion'].sum()} days")
print(f"  2. fire_weather_risk: {df['fire_weather_risk'].sum()} days")
print(f"  3. high_pollution_streak: {df['high_pollution_streak'].sum()} days")
print(f"  4. stagnant_air: {df['stagnant_air'].sum()} days")
print(f"  5. winter_smog_risk: {df['winter_smog_risk'].sum()} days")
print(f"  6. humidity_wind_trap: continuous feature")
print(f"✅ Created 6 extreme pollution detection features")

# ============================================================================
# STEP 9: DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Data Quality Checks")
print("=" * 70)

print(f"\nTotal features created: {df.shape[1] - 19} new features")
print(f"Final dataset shape: {df.shape}")
print(f"  (includes 6 extreme pollution detection features)")

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"\nInfinite values: {inf_count}")
if inf_count > 0:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("✅ Replaced infinite values with NaN")

# Check missing values
missing_values = df.isnull().sum()
missing_features = missing_values[missing_values > 0]

if len(missing_features) > 0:
    print(f"\n⚠️ Features with missing values:")
    for feature, count in missing_features.items():
        pct = (count / len(df)) * 100
        print(f"  {feature}: {count} ({pct:.2f}%)")
    
    # Fill missing values strategically
    print("\n🔧 Filling missing values...")
    
    # For lag and rolling features, missing at start is expected
    # Forward fill for first few rows, then backward fill if needed
    for col in df.columns:
        if 'lag' in col or 'rolling' in col:
            df[col].fillna(method='bfill', inplace=True)
    
    # For other features, use median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    print(f"✅ All missing values filled")
    print(f"Final missing values: {df.isnull().sum().sum()}")
else:
    print("✅ No missing values found")

# ============================================================================
# STEP 9: DROP ROWS WITH MISSING VALUES & CLEAN UP
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Removing Rows with Insufficient Historical Data")
print("=" * 70)

# Drop first 30 rows (insufficient lag/rolling features)
initial_rows = len(df)
df = df.iloc[30:].reset_index(drop=True)
rows_dropped = initial_rows - len(df)

print(f"Dropped first {rows_dropped} rows (insufficient history)")
print(f"Final dataset: {len(df)} rows")

# ============================================================================
# STEP 11: REMOVE DATA LEAKAGE (Same-day PM2.5 aggregate features)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 11: Removing Data Leakage Features")
print("=" * 70)

# Remove pm2_5_max and pm2_5_std - these are same-day values that shouldn't be used
# to predict pm2_5_mean (we can't know them before the day ends)
# We keep their lagged versions (pm2_5_max_lag1, pm2_5_max_lag7, etc.)
leakage_features = ['pm2_5_max', 'pm2_5_std']
features_before = len(df.columns)

df = df.drop(columns=leakage_features)

print(f"Removed {len(leakage_features)} same-day PM2.5 aggregate features:")
for feat in leakage_features:
    print(f"  ❌ {feat} (data leakage - same day as target)")
print(f"✅ Keeping lagged versions: pm2_5_max_lag1, pm2_5_max_lag7, pm2_5_std_lag1, etc.")
print(f"Features: {features_before} → {len(df.columns)}")

# ============================================================================
# STEP 12: SAVE ENGINEERED DATASET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 12: Saving Feature-Engineered Dataset")
print("=" * 70)

output_file = '../data/processed/features_engineered.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ Saved to: {output_file}")
print(f"\n📊 FINAL DATASET SUMMARY:")
print(f"  • Rows: {len(df)}")
print(f"  • Columns: {len(df.columns)}")
print(f"  • Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"  • Target Variable: pm2_5_mean")
print(f"  • Features: {len(df.columns) - 2} (excluding date and target)")

# Show feature categories
print(f"\n📋 FEATURE CATEGORIES:")
print(f"  • Original features: 18")
print(f"  • Temporal features: 18")
print(f"  • Lag features: {lag_count}")
print(f"  • Rolling features: {rolling_count}")
print(f"  • Derived weather: 6")
print(f"  • Interaction features: 5")
print(f"  • Pollutant ratios: 2")
print(f"  • Fire-related: 3")
print(f"  • Extreme pollution detection: 6")

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
