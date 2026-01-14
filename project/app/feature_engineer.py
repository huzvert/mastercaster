import pandas as pd
import numpy as np
from datetime import datetime

def engineer_features_for_date(df, target_date):
    """
    Engineer all 93 features for a specific date based on historical data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data with base features
    target_date : str or datetime
        The date for which to engineer features
    
    Returns:
    --------
    pandas.Series
        Engineered features for the target date (93 features)
    """
    target_date = pd.to_datetime(target_date)
    
    # Get data up to and including target date
    df_hist = df[df['date'] <= target_date].copy()
    
    if len(df_hist) < 31:
        raise ValueError(f"Insufficient historical data. Need at least 31 days, got {len(df_hist)} days.")
    
    # Get the row for target date
    target_idx = df_hist[df_hist['date'] == target_date].index[0]
    
    # ============================================================================
    # STEP 1: TEMPORAL FEATURES
    # ============================================================================
    date = df_hist.loc[target_idx, 'date']
    
    features = {}
    
    # Basic temporal features
    features['year'] = date.year
    features['day_of_week'] = date.dayofweek
    features['week_of_year'] = date.isocalendar()[1]
    features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
    
    # Seasonal indicators
    month = date.month
    features['is_winter'] = 1 if month in [12, 1, 2] else 0
    features['is_spring'] = 1 if month in [3, 4, 5] else 0
    features['is_summer'] = 1 if month in [6, 7, 8] else 0
    features['is_autumn'] = 1 if month in [9, 10, 11] else 0
    
    # Smog and crop burning season
    features['is_smog_season'] = 1 if (month in [11, 12, 1] or (month == 10 and date.day >= 15)) else 0
    features['is_crop_burning'] = 1 if ((month == 10 and date.day >= 15) or (month == 11 and date.day <= 15)) else 0
    
    # Cyclical encoding
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['dow_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
    features['dow_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
    
    # ============================================================================
    # STEP 2: CURRENT DAY BASE FEATURES
    # ============================================================================
    base_features = ['ozone', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide',
                     'relative_humidity_2m', 'tmax', 'prcp', 'wspd', 'pres',
                     'fire_frp_max', 'fire_frp_total']
    
    for feat in base_features:
        features[feat] = df_hist.loc[target_idx, feat]
    
    # ============================================================================
    # STEP 3: LAG FEATURES
    # ============================================================================
    lag_days = [1, 7, 14, 30]
    lag_feature_names = ['pm2_5_max', 'pm2_5_std', 'pm10', 'ozone', 'nitrogen_dioxide', 'fire_frp_total']
    
    for feat in lag_feature_names:
        for lag in lag_days:
            lag_idx = target_idx - lag
            if lag_idx >= 0:
                features[f'{feat}_lag{lag}'] = df_hist.loc[lag_idx, feat]
            else:
                features[f'{feat}_lag{lag}'] = 0  # Fallback for early dates
    
    # ============================================================================
    # STEP 4: ROLLING STATISTICS
    # ============================================================================
    # For rolling features, we need to look back from the day BEFORE target date
    # (to avoid data leakage - we don't know today's value when predicting tomorrow)
    
    rolling_windows = [3, 7, 30]
    
    # PM2.5 rolling features
    for window in rolling_windows:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'pm2_5_mean']
        features[f'pm2_5_mean_rolling_mean_{window}d'] = window_data.mean()
        features[f'pm2_5_mean_rolling_std_{window}d'] = window_data.std()
    
    # PM10 rolling features
    for window in rolling_windows:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'pm10']
        features[f'pm10_rolling_mean_{window}d'] = window_data.mean()
        features[f'pm10_rolling_std_{window}d'] = window_data.std()
    
    # Ozone rolling features (mean only)
    for window in rolling_windows:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'ozone']
        features[f'ozone_rolling_mean_{window}d'] = window_data.mean()
    
    # Temperature rolling features
    for window in rolling_windows:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'tavg']
        if window == 3:
            features[f'tavg_rolling_std_{window}d'] = window_data.std()
        elif window == 7:
            features[f'tavg_rolling_std_{window}d'] = window_data.std()
        elif window == 30:
            features[f'tavg_rolling_mean_{window}d'] = window_data.mean()
            features[f'tavg_rolling_std_{window}d'] = window_data.std()
    
    # Humidity rolling features
    for window in [7, 30]:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'relative_humidity_2m']
        features[f'relative_humidity_2m_rolling_mean_{window}d'] = window_data.mean()
    
    # Fire rolling features
    for window in rolling_windows:
        window_data = df_hist.loc[max(0, target_idx-window):target_idx-1, 'fire_frp_total']
        features[f'fire_frp_total_rolling_mean_{window}d'] = window_data.mean()
    
    # ============================================================================
    # STEP 5: DERIVED WEATHER FEATURES
    # ============================================================================
    features['temp_range'] = df_hist.loc[target_idx, 'tmax'] - df_hist.loc[target_idx, 'tmin']
    
    # Days since last rain
    days_since_rain = 0
    for i in range(target_idx, -1, -1):
        if df_hist.loc[i, 'prcp'] > 0:
            break
        days_since_rain += 1
    features['days_since_rain'] = days_since_rain
    
    # Cumulative precipitation (last 7 days)
    prcp_data = df_hist.loc[max(0, target_idx-6):target_idx, 'prcp']
    features['prcp_cumsum_7d'] = prcp_data.sum()
    
    # Temperature anomaly
    if target_idx >= 30:
        tavg_30d = df_hist.loc[target_idx-30:target_idx-1, 'tavg'].mean()
        features['tavg_anomaly'] = df_hist.loc[target_idx, 'tavg'] - tavg_30d
    else:
        features['tavg_anomaly'] = 0
    
    # Wind chill
    features['wind_chill'] = df_hist.loc[target_idx, 'tavg'] - (df_hist.loc[target_idx, 'wspd'] * 2)
    
    # ============================================================================
    # STEP 6: INTERACTION FEATURES
    # ============================================================================
    features['temp_humidity_interaction'] = df_hist.loc[target_idx, 'tavg'] * df_hist.loc[target_idx, 'relative_humidity_2m']
    
    fire_count = df_hist.loc[target_idx, 'fire_count'] if 'fire_count' in df_hist.columns else 0
    features['wind_fire_interaction'] = df_hist.loc[target_idx, 'wspd'] * fire_count
    
    features['pm10_ozone_interaction'] = df_hist.loc[target_idx, 'pm10'] * df_hist.loc[target_idx, 'ozone']
    
    # Inversion risk
    wspd = df_hist.loc[target_idx, 'wspd']
    humidity = df_hist.loc[target_idx, 'relative_humidity_2m']
    features['inversion_risk'] = 1 if (wspd < 1 and humidity > 80 and features['is_winter'] == 1) else 0
    
    # Crop fire intensity
    features['crop_fire_intensity'] = df_hist.loc[target_idx, 'fire_frp_total'] * features['is_crop_burning']
    
    # ============================================================================
    # STEP 7: POLLUTANT RATIOS
    # ============================================================================
    pm10_val = df_hist.loc[target_idx, 'pm10']
    features['pm2_5_pm10_ratio'] = df_hist.loc[target_idx, 'pm2_5_mean'] / (pm10_val + 1)
    
    # ============================================================================
    # STEP 8: FIRE-RELATED FEATURES
    # ============================================================================
    features['has_fire'] = 1 if fire_count > 0 else 0
    
    # Cumulative fire count (last 3 days)
    if 'fire_count' in df_hist.columns:
        fire_data = df_hist.loc[max(0, target_idx-2):target_idx, 'fire_count']
        features['fire_count_cumsum_3d'] = fire_data.sum()
        features['fire_intensity_per_fire'] = df_hist.loc[target_idx, 'fire_frp_total'] / (fire_count + 1)
    else:
        features['fire_count_cumsum_3d'] = 0
        features['fire_intensity_per_fire'] = 0
    
    # ============================================================================
    # STEP 9: EXTREME POLLUTION DETECTION FEATURES
    # ============================================================================
    # Strong inversion
    wspd_q25 = df_hist['wspd'].quantile(0.25)
    humidity_q75 = df_hist['relative_humidity_2m'].quantile(0.75)
    features['strong_inversion'] = 1 if (wspd < wspd_q25 and humidity > humidity_q75) else 0
    
    # Fire weather risk
    wspd_q33 = df_hist['wspd'].quantile(0.33)
    features['fire_weather_risk'] = 1 if (fire_count > 0 and wspd < wspd_q33) else 0
    
    # High pollution streak
    if f'pm2_5_mean_rolling_mean_3d' in features:
        threshold_high = df_hist['pm2_5_mean'].quantile(0.75)
        features['high_pollution_streak'] = 1 if features['pm2_5_mean_rolling_mean_3d'] > threshold_high else 0
    else:
        features['high_pollution_streak'] = 0
    
    # Stagnant air
    wspd_q15 = df_hist['wspd'].quantile(0.15)
    humidity_q85 = df_hist['relative_humidity_2m'].quantile(0.85)
    features['stagnant_air'] = 1 if (wspd < wspd_q15 and humidity > humidity_q85) else 0
    
    # Winter smog risk
    humidity_q70 = df_hist['relative_humidity_2m'].quantile(0.70)
    features['winter_smog_risk'] = 1 if (features['is_smog_season'] == 1 and wspd < wspd_q25 and humidity > humidity_q70) else 0
    
    # Humidity-wind trap
    features['humidity_wind_trap'] = features['temp_humidity_interaction'] * (1 / (wspd + 0.1))
    
    return pd.Series(features)


def get_feature_names():
    """Return the list of 93 feature names in the correct order."""
    return [
        'ozone', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide',
        'relative_humidity_2m', 'tmax', 'prcp', 'wspd', 'pres',
        'fire_frp_max', 'fire_frp_total',
        'year', 'day_of_week', 'week_of_year', 'is_weekend',
        'is_winter', 'is_spring', 'is_summer', 'is_autumn',
        'is_smog_season', 'is_crop_burning',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'pm2_5_max_lag1', 'pm2_5_max_lag7', 'pm2_5_max_lag14', 'pm2_5_max_lag30',
        'pm2_5_std_lag1', 'pm2_5_std_lag7', 'pm2_5_std_lag14', 'pm2_5_std_lag30',
        'pm10_lag1', 'pm10_lag7', 'pm10_lag14', 'pm10_lag30',
        'ozone_lag1', 'ozone_lag7', 'ozone_lag14', 'ozone_lag30',
        'nitrogen_dioxide_lag1', 'nitrogen_dioxide_lag7', 'nitrogen_dioxide_lag14', 'nitrogen_dioxide_lag30',
        'fire_frp_total_lag1', 'fire_frp_total_lag7', 'fire_frp_total_lag14', 'fire_frp_total_lag30',
        'pm2_5_mean_rolling_mean_3d', 'pm2_5_mean_rolling_std_3d',
        'pm2_5_mean_rolling_mean_7d', 'pm2_5_mean_rolling_std_7d',
        'pm2_5_mean_rolling_mean_30d', 'pm2_5_mean_rolling_std_30d',
        'pm10_rolling_mean_3d', 'pm10_rolling_std_3d',
        'pm10_rolling_mean_7d', 'pm10_rolling_std_7d',
        'pm10_rolling_mean_30d', 'pm10_rolling_std_30d',
        'ozone_rolling_mean_3d', 'ozone_rolling_mean_7d', 'ozone_rolling_mean_30d',
        'tavg_rolling_std_3d', 'tavg_rolling_std_7d',
        'tavg_rolling_mean_30d', 'tavg_rolling_std_30d',
        'relative_humidity_2m_rolling_mean_7d', 'relative_humidity_2m_rolling_mean_30d',
        'fire_frp_total_rolling_mean_3d', 'fire_frp_total_rolling_mean_7d', 'fire_frp_total_rolling_mean_30d',
        'temp_range', 'days_since_rain', 'prcp_cumsum_7d', 'tavg_anomaly', 'wind_chill',
        'temp_humidity_interaction', 'wind_fire_interaction', 'pm10_ozone_interaction',
        'inversion_risk', 'crop_fire_intensity', 'pm2_5_pm10_ratio',
        'has_fire', 'fire_count_cumsum_3d', 'fire_intensity_per_fire',
        'strong_inversion', 'fire_weather_risk', 'high_pollution_streak',
        'stagnant_air', 'winter_smog_risk', 'humidity_wind_trap'
    ]
