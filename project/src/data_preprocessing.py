import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
import json

print("=" * 70)
print("PREPROCESSING: Feature Selection, Outlier Handling & Scaling")
print("=" * 70)

# Load feature-engineered data
df = pd.read_csv('../data/processed/features_engineered.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n📊 Loaded Dataset: {df.shape}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# STEP 1: SANITY CHECKS (No Outlier Capping)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Data Sanity Checks")
print("=" * 70)

# Only apply basic sanity checks (physical constraints)
# No statistical outlier capping - pollution events are real!

print("Applying physical constraint checks...")

# Ensure PM2.5 and pollutants are non-negative (only clip features that exist)
df['pm2_5_mean'] = df['pm2_5_mean'].clip(lower=0)
# Note: pm2_5_max and pm2_5_std removed from features (data leakage)

if 'pm10' in df.columns:
    df['pm10'] = df['pm10'].clip(lower=0)
if 'ozone' in df.columns:
    df['ozone'] = df['ozone'].clip(lower=0)
if 'nitrogen_dioxide' in df.columns:
    df['nitrogen_dioxide'] = df['nitrogen_dioxide'].clip(lower=0)
if 'sulphur_dioxide' in df.columns:
    df['sulphur_dioxide'] = df['sulphur_dioxide'].clip(lower=0)
if 'carbon_monoxide' in df.columns:
    df['carbon_monoxide'] = df['carbon_monoxide'].clip(lower=0)

# Check for any extreme sensor errors (e.g., PM2.5 > 1000 would be suspicious)
extreme_pm25 = (df['pm2_5_mean'] > 1000).sum()
if extreme_pm25 > 0:
    print(f"  ⚠️ Warning: Found {extreme_pm25} PM2.5 values > 1000 (possible sensor error)")
    df['pm2_5_mean'] = df['pm2_5_mean'].clip(upper=1000)
else:
    print(f"  ✅ No extreme sensor errors detected")

print(f"  ✅ PM2.5 range: [{df['pm2_5_mean'].min():.2f}, {df['pm2_5_mean'].max():.2f}]")
print(f"✅ Using real pollution values without statistical capping")

outlier_summary = {
    "method": "No statistical outlier capping applied",
    "reason": "Pollution events are real environmental phenomena, not statistical outliers"
}

# ============================================================================
# STEP 2: REMOVE LOW VARIANCE FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Removing Low Variance Features")
print("=" * 70)

# Separate features from target and date
feature_cols = [col for col in df.columns if col not in ['date', 'pm2_5_mean']]
X = df[feature_cols].copy()
y = df['pm2_5_mean'].copy()
dates = df['date'].copy()

print(f"Initial features: {len(feature_cols)}")

# Remove features with variance < 0.01
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()].tolist()

removed_features = set(feature_cols) - set(selected_features)
print(f"Removed {len(removed_features)} low-variance features:")
for feat in list(removed_features)[:10]:  # Show first 10
    print(f"  - {feat}")
if len(removed_features) > 10:
    print(f"  ... and {len(removed_features) - 10} more")

X = pd.DataFrame(X_reduced, columns=selected_features)
print(f"✅ Remaining features: {len(selected_features)}")

# ============================================================================
# STEP 3: REMOVE HIGHLY CORRELATED FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Removing Highly Correlated Features (>0.95)")
print("=" * 70)

# Calculate correlation matrix
corr_matrix = X.corr().abs()

# Find pairs with correlation > 0.95
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = []
for column in upper_triangle.columns:
    correlated_features = upper_triangle.index[upper_triangle[column] > 0.95].tolist()
    if correlated_features:
        to_drop.extend(correlated_features)

to_drop = list(set(to_drop))  # Remove duplicates

if to_drop:
    print(f"Removing {len(to_drop)} highly correlated features:")
    for feat in to_drop[:10]:  # Show first 10
        print(f"  - {feat}")
    if len(to_drop) > 10:
        print(f"  ... and {len(to_drop) - 10} more")
    
    X = X.drop(columns=to_drop)
    print(f"✅ Remaining features: {len(X.columns)}")
else:
    print("✅ No highly correlated features found")

# ============================================================================
# STEP 4: CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Chronological Train/Validation/Test Split")
print("=" * 70)

# Time-series split: 80% train, 10% validation, 10% test
# This ensures training includes Nov-Dec 2023 smog season (extreme pollution events)
n = len(X)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

# Split indices
train_idx = slice(0, train_end)
val_idx = slice(train_end, val_end)
test_idx = slice(val_end, n)

# Split data
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]

dates_train = dates.iloc[train_idx]
dates_val = dates.iloc[val_idx]
dates_test = dates.iloc[test_idx]

print(f"\n📊 Split Summary:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%) | {dates_train.min()} to {dates_train.max()}")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/n*100:.1f}%) | {dates_val.min()} to {dates_val.max()}")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/n*100:.1f}%) | {dates_test.min()} to {dates_test.max()}")

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Feature Scaling (StandardScaler)")
print("=" * 70)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✅ Scaled features using StandardScaler")
print(f"   Mean: ~0, Std: ~1")

# ============================================================================
# STEP 6: SAVE PREPROCESSED DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Saving Preprocessed Datasets")
print("=" * 70)

# Create processed directory if needed
import os
os.makedirs('../data/processed', exist_ok=True)

# Save train/val/test sets
train_data = pd.concat([dates_train.reset_index(drop=True), 
                        X_train_scaled.reset_index(drop=True), 
                        y_train.reset_index(drop=True)], axis=1)
train_data.columns = ['date'] + list(X_train.columns) + ['pm2_5_mean']
train_data.to_csv('../data/processed/train_data.csv', index=False)

val_data = pd.concat([dates_val.reset_index(drop=True), 
                      X_val_scaled.reset_index(drop=True), 
                      y_val.reset_index(drop=True)], axis=1)
val_data.columns = ['date'] + list(X_val.columns) + ['pm2_5_mean']
val_data.to_csv('../data/processed/val_data.csv', index=False)

test_data = pd.concat([dates_test.reset_index(drop=True), 
                       X_test_scaled.reset_index(drop=True), 
                       y_test.reset_index(drop=True)], axis=1)
test_data.columns = ['date'] + list(X_test.columns) + ['pm2_5_mean']
test_data.to_csv('../data/processed/test_data.csv', index=False)

print(f"✅ Saved train_data.csv ({len(train_data)} rows)")
print(f"✅ Saved val_data.csv ({len(val_data)} rows)")
print(f"✅ Saved test_data.csv ({len(test_data)} rows)")

# Save scaler for later use
joblib.dump(scaler, '../models/scaler.pkl')
print(f"✅ Saved scaler.pkl")

# Save feature names
feature_info = {
    'selected_features': list(X_train.columns),
    'n_features': len(X_train.columns),
    'removed_low_variance': list(removed_features),
    'removed_high_correlation': to_drop,
    'outlier_summary': outlier_summary
}

with open('../data/processed/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✅ Saved feature_info.json")

# ============================================================================
# STEP 7: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)

print(f"\n📊 FINAL SUMMARY:")
print(f"  • Total samples: {n}")
print(f"  • Final features: {len(X_train.columns)}")
print(f"  • Train samples: {len(X_train)} ({dates_train.min().date()} to {dates_train.max().date()})")
print(f"  • Val samples: {len(X_val)} ({dates_val.min().date()} to {dates_val.max().date()})")
print(f"  • Test samples: {len(X_test)} ({dates_test.min().date()} to {dates_test.max().date()})")

print(f"\n📋 TARGET VARIABLE STATISTICS:")
print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, Range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  Val   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}, Range: [{y_val.min():.2f}, {y_val.max():.2f}]")
print(f"  Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, Range: [{y_test.min():.2f}, {y_test.max():.2f}]")

print(f"\n✅ Ready for model training!")
print(f"\nNext steps:")
print(f"  1. Train baseline models (mean, moving average)")
print(f"  2. Train classical ML models (Ridge, Random Forest, SVR)")
print(f"  3. Train ensemble models (XGBoost, LightGBM)")
print(f"  4. Evaluate and compare all models")