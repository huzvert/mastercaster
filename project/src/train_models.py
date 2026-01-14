import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LAHORE PM2.5 FORECASTING - MODEL TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Loading Preprocessed Data")
print("=" * 80)

train_data = pd.read_csv('../data/processed/train_data.csv')
val_data = pd.read_csv('../data/processed/val_data.csv')
test_data = pd.read_csv('../data/processed/test_data.csv')

# Separate features and target
X_train = train_data.drop(columns=['date', 'pm2_5_mean'])
y_train = train_data['pm2_5_mean']

X_val = val_data.drop(columns=['date', 'pm2_5_mean'])
y_val = val_data['pm2_5_mean']

X_test = test_data.drop(columns=['date', 'pm2_5_mean'])
y_test = test_data['pm2_5_mean']

print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Val set:   {X_val.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")
print(f"\nTarget variable (PM2.5) statistics:")
print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"  Val   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
print(f"  Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# ============================================================================
# STEP 2: CREATE SAMPLE WEIGHTS FOR HIGH POLLUTION FOCUS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Creating Sample Weights for High Pollution Focus")
print("=" * 80)

# Create sample weights: higher weight for EXTREME pollution days only
# Weight based on absolute PM2.5 level - only truly extreme days (>150 µg/m³) get 2x weight
# This is gentler than 3x on top 33%, focusing on actual smog events
extreme_pollution_threshold = 150  # WHO's 24h PM2.5 guideline is 15, 150 is 10x that

sample_weights = np.ones(len(y_train))
extreme_pollution_mask = y_train >= extreme_pollution_threshold

sample_weights[extreme_pollution_mask] = 2.0  # 2x weight for extreme pollution days

print(f"\nSample weighting strategy (refined):")
print(f"  Extreme pollution threshold: {extreme_pollution_threshold:.2f} µg/m³ (top ~{(extreme_pollution_mask.sum() / len(y_train) * 100):.1f}%)")
print(f"  Normal samples: {(~extreme_pollution_mask).sum()} (weight = 1.0)")
print(f"  Extreme pollution samples: {extreme_pollution_mask.sum()} (weight = 2.0)")
print(f"  Effective extreme pollution emphasis: {extreme_pollution_mask.sum() * 2.0 / sample_weights.sum() * 100:.1f}% of total weight")

# ============================================================================
# STEP 3: EVALUATION METRICS
# ============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R2
    n = len(y_true)
    p = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2,
        'mape': mape
    }

# ============================================================================
# STEP 4: BASELINE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Training Baseline Models")
print("=" * 80)

baseline_results = []

# Baseline 1: Mean Predictor
print("\n[1/4] Mean Predictor...")
mean_pred = np.full(len(y_val), y_train.mean())
baseline_results.append(evaluate_model(y_val, mean_pred, 'Mean Baseline'))
print(f"  Val R²: {baseline_results[-1]['r2']:.4f}, MAE: {baseline_results[-1]['mae']:.2f}")

# Baseline 2: Median Predictor
print("[2/4] Median Predictor...")
median_pred = np.full(len(y_val), y_train.median())
baseline_results.append(evaluate_model(y_val, median_pred, 'Median Baseline'))
print(f"  Val R²: {baseline_results[-1]['r2']:.4f}, MAE: {baseline_results[-1]['mae']:.2f}")

# Baseline 3: Last Value Persistence (using lag features)
print("[3/4] Last Value Persistence...")
# Use pm2_5_mean_lag1 from validation set (need unscaled data)
# Simple persistence: tomorrow's value = today's value
train_data_full = pd.read_csv('../data/processed/train_data.csv')
val_data_full = pd.read_csv('../data/processed/val_data.csv')

# For persistence, we need the actual last known values
# Use a simple approach: predict val as shifted version
persistence_pred = np.concatenate([[y_train.iloc[-1]], y_val.values[:-1]])
baseline_results.append(evaluate_model(y_val, persistence_pred, 'Persistence Baseline'))
print(f"  Val R²: {baseline_results[-1]['r2']:.4f}, MAE: {baseline_results[-1]['mae']:.2f}")

# Baseline 4: 7-day Moving Average
print("[4/4] 7-day Moving Average...")
# Use last 7 days from training to predict first val, then roll forward
moving_avg_pred = []
recent_values = list(y_train.tail(7).values)

for i in range(len(y_val)):
    pred = np.mean(recent_values[-7:])
    moving_avg_pred.append(pred)
    recent_values.append(y_val.iloc[i])  # Use actual value for next prediction

baseline_results.append(evaluate_model(y_val, moving_avg_pred, '7-day Moving Avg'))
print(f"  Val R²: {baseline_results[-1]['r2']:.4f}, MAE: {baseline_results[-1]['mae']:.2f}")

print(f"\n✅ Baseline models complete")

# ============================================================================
# STEP 5: CLASSICAL ML MODELS WITH SAMPLE WEIGHTING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Training Classical ML Models with Sample Weighting")
print("=" * 80)

classical_results = []
trained_models = {}

# Model 1: Ridge Regression
print("\n[1/4] Ridge Regression with GridSearchCV...")
start_time = time.time()

ridge_params = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}

tscv = TimeSeriesSplit(n_splits=5)
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid.fit(X_train, y_train, sample_weight=sample_weights)

ridge_best = ridge_grid.best_estimator_
y_val_pred = ridge_best.predict(X_val)
ridge_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'Ridge Regression')
result['train_time'] = ridge_time
result['best_params'] = ridge_grid.best_params_
classical_results.append(result)
trained_models['ridge'] = ridge_best

print(f"  Best params: {ridge_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {ridge_time:.2f}s")

# Model 2: Lasso Regression
print("\n[2/4] Lasso Regression with GridSearchCV...")
start_time = time.time()

lasso_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lasso, lasso_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid.fit(X_train, y_train, sample_weight=sample_weights)

lasso_best = lasso_grid.best_estimator_
y_val_pred = lasso_best.predict(X_val)
lasso_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'Lasso Regression')
result['train_time'] = lasso_time
result['best_params'] = lasso_grid.best_params_
classical_results.append(result)
trained_models['lasso'] = lasso_best

print(f"  Best params: {lasso_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {lasso_time:.2f}s")

# Model 3: Random Forest
print("\n[3/4] Random Forest with GridSearchCV...")
start_time = time.time()

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train, sample_weight=sample_weights)

rf_best = rf_grid.best_estimator_
y_val_pred = rf_best.predict(X_val)
rf_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'Random Forest')
result['train_time'] = rf_time
result['best_params'] = rf_grid.best_params_
classical_results.append(result)
trained_models['random_forest'] = rf_best

print(f"  Best params: {rf_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {rf_time:.2f}s")

# Model 4: Support Vector Regression
print("\n[4/4] Support Vector Regression with GridSearchCV...")
start_time = time.time()

svr_params = {
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf']
}

svr_model = SVR()
svr_grid = GridSearchCV(svr_model, svr_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid.fit(X_train, y_train)

svr_best = svr_grid.best_estimator_
y_val_pred = svr_best.predict(X_val)
svr_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'SVR')
result['train_time'] = svr_time
result['best_params'] = svr_grid.best_params_
classical_results.append(result)
trained_models['svr'] = svr_best

print(f"  Best params: {svr_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {svr_time:.2f}s")

print(f"\n✅ Classical ML models complete")

# ============================================================================
# STEP 6: ENSEMBLE/ADVANCED MODELS WITH SAMPLE WEIGHTING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Training Ensemble/Advanced Models with Sample Weighting")
print("=" * 80)

ensemble_results = []

# Model 1: XGBoost
print("\n[1/3] XGBoost with GridSearchCV...")
start_time = time.time()

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train, sample_weight=sample_weights)

xgb_best = xgb_grid.best_estimator_
y_val_pred = xgb_best.predict(X_val)
xgb_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'XGBoost')
result['train_time'] = xgb_time
result['best_params'] = xgb_grid.best_params_
ensemble_results.append(result)
trained_models['xgboost'] = xgb_best

print(f"  Best params: {xgb_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {xgb_time:.2f}s")

# Model 2: LightGBM
print("\n[2/3] LightGBM with GridSearchCV...")
start_time = time.time()

lgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63]
}

lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
lgb_grid.fit(X_train, y_train, sample_weight=sample_weights)

lgb_best = lgb_grid.best_estimator_
y_val_pred = lgb_best.predict(X_val)
lgb_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'LightGBM')
result['train_time'] = lgb_time
result['best_params'] = lgb_grid.best_params_
ensemble_results.append(result)
trained_models['lightgbm'] = lgb_best

print(f"  Best params: {lgb_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {lgb_time:.2f}s")

# Model 3: Gradient Boosting
print("\n[3/3] Gradient Boosting with GridSearchCV...")
start_time = time.time()

gb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

gb_model = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(gb_model, gb_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train, sample_weight=sample_weights)

gb_best = gb_grid.best_estimator_
y_val_pred = gb_best.predict(X_val)
gb_time = time.time() - start_time

result = evaluate_model(y_val, y_val_pred, 'Gradient Boosting')
result['train_time'] = gb_time
result['best_params'] = gb_grid.best_params_
ensemble_results.append(result)
trained_models['gradient_boosting'] = gb_best

print(f"  Best params: {gb_grid.best_params_}")
print(f"  Val R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, Time: {gb_time:.2f}s")

print(f"\n✅ Ensemble models complete")

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Model Comparison on Validation Set")
print("=" * 80)

# Combine all results
all_results = baseline_results + classical_results + ensemble_results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('r2', ascending=False)

print("\n" + "=" * 80)
print("VALIDATION SET PERFORMANCE COMPARISON")
print("=" * 80)
print(f"\n{'Model':<25} {'R²':<8} {'Adj R²':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Time(s)':<10}")
print("-" * 85)

for _, row in results_df.iterrows():
    train_time = row.get('train_time', 0)
    print(f"{row['model']:<25} {row['r2']:<8.4f} {row['adj_r2']:<8.4f} {row['mae']:<8.2f} "
          f"{row['rmse']:<8.2f} {row['mape']:<8.2f} {train_time:<10.2f}")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Feature Importance Analysis")
print("=" * 80)

feature_importance = {}

# Random Forest importance
rf_importances = trained_models['random_forest'].feature_importances_
feature_importance['random_forest'] = dict(zip(X_train.columns, rf_importances))

# XGBoost importance
xgb_importances = trained_models['xgboost'].feature_importances_
feature_importance['xgboost'] = dict(zip(X_train.columns, xgb_importances))

# LightGBM importance
lgb_importances = trained_models['lightgbm'].feature_importances_
feature_importance['lightgbm'] = dict(zip(X_train.columns, lgb_importances))

# Get top 10 features from best model
best_model_name = results_df.iloc[0]['model'].lower().replace(' ', '_')
if 'random_forest' in best_model_name:
    top_features = sorted(feature_importance['random_forest'].items(), key=lambda x: x[1], reverse=True)[:10]
elif 'xgboost' in best_model_name:
    top_features = sorted(feature_importance['xgboost'].items(), key=lambda x: x[1], reverse=True)[:10]
elif 'lightgbm' in best_model_name:
    top_features = sorted(feature_importance['lightgbm'].items(), key=lambda x: x[1], reverse=True)[:10]
else:
    top_features = sorted(feature_importance['xgboost'].items(), key=lambda x: x[1], reverse=True)[:10]

print(f"\nTop 10 Most Important Features:")
print("-" * 60)
for i, (feature, importance) in enumerate(top_features, 1):
    print(f"{i:2d}. {feature:<45} {importance:.4f}")

# ============================================================================
# STEP 9: ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Error Analysis")
print("=" * 80)

# Use best model for error analysis
best_model_row = results_df.iloc[0]
best_model_name_key = best_model_row['model'].lower().replace(' ', '_')

# Map model names to keys
model_key_map = {
    'ridge_regression': 'ridge',
    'lasso_regression': 'lasso',
    'random_forest': 'random_forest',
    'svr': 'svr',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'gradient_boosting': 'gradient_boosting'
}

best_model_key = model_key_map.get(best_model_name_key, 'xgboost')
best_model = trained_models[best_model_key]

y_val_pred = best_model.predict(X_val)
errors = y_val - y_val_pred
abs_errors = np.abs(errors)

# Get worst predictions
worst_indices = abs_errors.nlargest(10).index
val_data_full = pd.read_csv('../data/processed/val_data.csv')

print(f"\nBest Model: {best_model_row['model']}")
print(f"\nTop 10 Worst Predictions:")
print("-" * 80)
print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Abs Error':<10}")
print("-" * 80)

for idx in worst_indices:
    date = val_data_full.iloc[idx]['date']
    actual = y_val.iloc[idx]
    predicted = y_val_pred[idx]
    error = errors.iloc[idx]
    abs_error = abs_errors.iloc[idx]
    print(f"{date:<12} {actual:<10.2f} {predicted:<10.2f} {error:<10.2f} {abs_error:<10.2f}")

# Analyze by pollution level
print(f"\nError Analysis by Pollution Level:")
print("-" * 60)

low_mask = y_val < y_val.quantile(0.33)
mid_mask = (y_val >= y_val.quantile(0.33)) & (y_val < y_val.quantile(0.67))
high_mask = y_val >= y_val.quantile(0.67)

print(f"Low PM2.5 (<{y_val.quantile(0.33):.2f}):  MAE = {abs_errors[low_mask].mean():.2f}, n = {low_mask.sum()}")
print(f"Mid PM2.5:   MAE = {abs_errors[mid_mask].mean():.2f}, n = {mid_mask.sum()}")
print(f"High PM2.5 (>{y_val.quantile(0.67):.2f}): MAE = {abs_errors[high_mask].mean():.2f}, n = {high_mask.sum()}")

# ============================================================================
# STEP 10: SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: Saving Models and Results")
print("=" * 80)

import os
os.makedirs('../models', exist_ok=True)

# Save all trained models
for name, model in trained_models.items():
    joblib.dump(model, f'../models/{name}_model.pkl')
    print(f"✅ Saved {name}_model.pkl")

# Save results
results_dict = {
    'baseline_results': baseline_results,
    'classical_results': classical_results,
    'ensemble_results': ensemble_results,
    'best_model': best_model_row['model'],
    'feature_importance': {k: {feat: float(imp) for feat, imp in v.items()} 
                          for k, v in feature_importance.items()},
    'timestamp': datetime.now().isoformat()
}

with open('../models/training_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"✅ Saved training_results.json")

# Save comparison table
results_df.to_csv('../models/model_comparison.csv', index=False)
print(f"✅ Saved model_comparison.csv")

# ============================================================================
# STEP 11: FINAL TEST SET EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: Final Evaluation on Test Set")
print("=" * 80)

print(f"\nEvaluating best model ({best_model_row['model']}) on test set...")

y_test_pred = best_model.predict(X_test)
test_metrics = evaluate_model(y_test, y_test_pred, best_model_row['model'])

print(f"\n{'Metric':<15} {'Value':<10}")
print("-" * 30)
print(f"{'R²':<15} {test_metrics['r2']:<10.4f}")
print(f"{'Adjusted R²':<15} {test_metrics['adj_r2']:<10.4f}")
print(f"{'MAE':<15} {test_metrics['mae']:<10.2f}")
print(f"{'RMSE':<15} {test_metrics['rmse']:<10.2f}")
print(f"{'MAPE':<15} {test_metrics['mape']:<10.2f}%")

# Save test predictions
test_results = pd.DataFrame({
    'date': test_data['date'],
    'actual': y_test.values,
    'predicted': y_test_pred
})
test_results.to_csv('../models/test_predictions.csv', index=False)
print(f"\n✅ Saved test_predictions.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 SUMMARY:")
print(f"  • Total models trained: {len(all_results)}")
print(f"  • Best model: {best_model_row['model']}")
print(f"  • Best validation R²: {best_model_row['r2']:.4f}")
print(f"  • Test set R²: {test_metrics['r2']:.4f}")
print(f"  • Test set MAE: {test_metrics['mae']:.2f} µg/m³")

print(f"\n📁 Saved Files:")
print(f"  • models/*_model.pkl (trained models)")
print(f"  • models/training_results.json (all metrics)")
print(f"  • models/model_comparison.csv (comparison table)")
print(f"  • models/test_predictions.csv (test set predictions)")

print(f"\n✅ Ready for deployment and visualization!")
