import joblib
import pandas as pd
import numpy as np

# Load the trained Lasso model
lasso_model = joblib.load('../models/lasso_model.pkl')

# Load feature names (from training data)
train_data = pd.read_csv('../data/processed/train_data.csv')
feature_names = [col for col in train_data.columns if col not in ['date', 'pm2_5_mean']]

# Get coefficients (feature importance for Lasso)
coefficients = lasso_model.coef_

# Create DataFrame with feature names and their absolute coefficients
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
})

# Sort by absolute coefficient (importance)
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

# Filter out features with zero coefficients (Lasso sets unimportant features to 0)
non_zero_features = feature_importance[feature_importance['abs_coefficient'] > 0]

print("=" * 80)
print("LASSO REGRESSION - TOP 10 MOST IMPORTANT FEATURES")
print("=" * 80)
print(f"\nTotal features: {len(feature_names)}")
print(f"Non-zero features: {len(non_zero_features)}")
print(f"Features set to zero: {len(feature_names) - len(non_zero_features)}")

print("\n" + "=" * 80)
print(f"{'Rank':<6} {'Feature':<50} {'Coefficient':<15} {'Abs Coef':<15}")
print("=" * 80)

for idx, row in non_zero_features.head(10).iterrows():
    rank = non_zero_features.index.get_loc(idx) + 1
    print(f"{rank:<6} {row['feature']:<50} {row['coefficient']:<15.6f} {row['abs_coefficient']:<15.6f}")

print("\n" + "=" * 80)

# Save full feature importance to CSV
feature_importance.to_csv('../models/lasso_feature_importance.csv', index=False)
print(f"✅ Saved full feature importance to: models/lasso_feature_importance.csv")