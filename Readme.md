# Master Forecaster Dashboard

> **Interactive Streamlit application for PM2.5 air quality prediction in Lahore, Pakistan**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## Overview

Multi-page interactive dashboard for PM2.5 forecasting system built for **CS-245 Machine Learning** and **CS-366 Data Visualization** courses. The application provides:

- **Real-time Predictions:** Input current conditions and get tomorrow's PM2.5 forecast using 7 ML models
- **Data Visualizations:** 10 diverse interactive charts exploring pollution patterns and trends  
- **ML Performance Analytics:** Model comparison, prediction accuracy, and feature importance analysis
- **Complete Documentation:** Detailed ML/DV pipeline methodology and technical insights
- **Professional UI:** Clean design with AQI color-coding and health recommendations

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to app directory:**
```bash
cd project/app
```

2. **Install dependencies:**
```bash
pip install -r ../requirements.txt
```

3. **Run the application:**
```bash
streamlit run Predict.py
```

4. **Open in browser:**
The app will automatically open at `http://localhost:8501`

---

## Application Structure

This is a **multi-page Streamlit app** with 3 pages:

### **1. Predict (Main Page)**
**File:** `Predict.py`

**Purpose:** Make next-day PM2.5 predictions using machine learning models

**Features:**
- **Model Selection:** Choose from 7 ML models:
  - Lasso Regression (Best: RMSE 19.69, R² 0.837)
  - Ridge Regression
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Random Forest
  - SVR
  
- **Input Parameters:**
  - **Air Quality:** PM10, Ozone, CO, SO₂, NO₂
  - **Weather:** Max Temperature, Humidity, Precipitation, Wind Speed, Pressure
  - **Fire Activity:** FRP Max, FRP Total

- **Prediction Output:**
  - Predicted PM2.5 for next day
  - AQI category with EPA color-coding
  - Health implications and recommendations
  - Actual vs Predicted comparison (if historical data available)
  - Prediction accuracy metrics
  
- **Export:** Download results as CSV

**Design Highlights:**
- Professional color palette with gradient backgrounds
- AQI badges color-coded by severity (Good=Green, Hazardous=Maroon)
- Comparison charts using Plotly
- Input validation with warnings for unusual values

---

### **2. Visualize**
**File:** `pages/1_Visualize.py`

**Purpose:** Comprehensive data exploration through 10 diverse visualizations

**Features:**

**Executive Dashboard:**
- 5 KPI metric cards with gradient backgrounds
- Dataset overview (854 days: Jan 2023 - May 2025)
- Average PM2.5, Maximum PM2.5, Unhealthy days count, etc.

**10 Interactive Visualizations:**

1. **Time Series Analysis:** PM2.5 trends over time with range slider
2. **Seasonal Box Plots:** Monthly pollution distribution patterns
3. **Correlation Heatmap:** Relationships between pollutants and weather
4. **Calendar Heatmap:** Month-day grid showing pollution intensity
5. **AQI Category Distribution:** Pie chart of air quality categories
6. **Wind Rose Polar Chart:** Wind direction and pollution relationship
7. **Year-over-Year Comparison:** Bar chart comparing annual averages
8. **ML Model Comparison:** 4-panel subplot (RMSE, MAE, R², MAPE) for 11 models
9. **Prediction Accuracy:** Actual vs Predicted scatter + error histogram
10. **Feature Importance:** Top 15 Lasso coefficients with color-coding

**Key Insights:**
- Each visualization includes professional insight boxes
- Color-coded findings (environmental patterns, seasonal trends, ML performance)
- Summary section with all 10 key takeaways

**Design:**
- White headers on dark backgrounds for visibility
- Consistent color scheme across all charts
- Plotly interactive features (zoom, pan, hover)

---

### **3. About**
**File:** `pages/2_About.py`

**Purpose:** Complete documentation of the ML and DV pipeline

**Content:**

**Project Context:**
- Problem statement from both ML and DV perspectives
- Technical objectives and challenges

**Dataset Information:**
- 3 data sources: AQI (Open Meteo API), Weather (Meteostat), Fire (NASA FIRMS)
- 852 days of data (Jan 2023 - May 2025)
- 19 columns in cleaned dataset

**Data Processing Pipeline:**
- **Step 1:** Data Collection & Merging (hourly to daily aggregation)
- **Step 2:** Data Cleaning (sanity checks, no IQR capping, 19 columns output)
- **Step 3:** Feature Engineering (100+ features created: temporal, lag, rolling, interactions, extreme pollution)
- **Step 3.5:** Preprocessing (variance/correlation filtering → 93 final features, z-score scaling, 80/10/10 split, sample weighting)
- **Step 4:** Dataset Reduction for DV (kept 19 original columns for interpretability)

**Model Training:**
- 11 models trained (4 baselines, 3 classical, 4 ensemble)
- GridSearchCV with TimeSeriesSplit (5 folds)
- Sample weighting: 2× for extreme pollution (PM2.5 > 150)
- Lasso achieves best performance

**Visualization Strategy:**
- 10 visualization types
- Mix of exploratory (data patterns) and explanatory (ML insights)
- Interactivity, professional design, color-coded insights

**Key Findings:**
- Environmental insights (winter smog, monsoon cleansing, temperature inversion)
- ML insights (PM10×Ozone strongest predictor, linear models outperform ensembles)

**Design:**
- Colored boxes matching Visualize page theme
- Technical depth with code references
- Factually verified against actual src/ implementation

---

## Data Requirements

### Required Files

The app expects the following files:

**In `../data/processed/`:**
1. **`merged_data_extended.csv`**
   - Extended dataset with lag/rolling features for ML prediction
   - Used by Predict.py for generating features from historical context
   
2. **`merged_data_clean.csv`**
   - Cleaned dataset (854 rows, 19 columns)
   - Used by Visualize.py for charts
   - Columns: date, pm2_5_mean, pm2_5_max, pm2_5_std, pm10, ozone, nitrogen_dioxide, sulphur_dioxide, carbon_monoxide, relative_humidity_2m, tavg, tmin, tmax, prcp, wspd, pres, fire_count, fire_frp_max, fire_frp_total

3. **`feature_info.json`**
   - List of 93 selected features after variance/correlation filtering

**In `../models/`:**
1. **Model Files (7 total):**
   - `lasso_model.pkl` (Best model)
   - `ridge_model.pkl`
   - `gradient_boosting_model.pkl`
   - `xgboost_model.pkl`
   - `lightgbm_model.pkl`
   - `random_forest_model.pkl`
   - `svr_model.pkl`

2. **Preprocessing:**
   - `scaler.pkl` - StandardScaler fitted on training data

3. **Metrics & Analysis:**
   - `training_results.json` - Detailed training metrics
   - `model_comparison.csv` - Performance comparison (11 models ranked)
   - `lasso_feature_importance.csv` - 93 features with coefficients
   - `test_predictions.csv` - Actual vs predicted values

### File Structure
```
master-forecaster/
├── LICENSE
├── README.md                            # This file
├── project/
│   ├── requirements.txt                 # Python dependencies
│   ├── app/
│   │   ├── Predict.py                   # Main prediction page
│   │   ├── feature_engineer.py          # Feature engineering module
│   │   ├── predictor.py                 # Prediction utilities
│   │   └── pages/
│   │       ├── 1_Visualize.py           # Data visualization page
│   │       └── 2_About.py               # Documentation page
│   ├── data/
│   │   ├── raw/
│   │   │   ├── aqi.csv                  # Hourly AQI data
│   │   │   ├── weather.csv              # Daily weather data
│   │   │   └── fire.csv                 # NASA FIRMS fire data
│   │   └── processed/
│   │       ├── merged_data.csv          # Initial merge (19 columns)
│   │       ├── merged_data_clean.csv    # After cleaning (854 rows, 19 cols)
│   │       ├── merged_data_extended.csv # With lag/rolling features
│   │       ├── features_engineered.csv  # 100+ raw features
│   │       ├── feature_info.json        # 93 selected features
│   │       ├── train_data.csv           # 80% training set
│   │       ├── val_data.csv             # 10% validation set
│   │       └── test_data.csv            # 10% test set
│   ├── models/
│   │   ├── *.pkl                        # 7 trained models
│   │   ├── scaler.pkl                   # Feature scaler
│   │   ├── training_results.json        # Training metrics
│   │   ├── model_comparison.csv         # Performance table
│   │   ├── lasso_feature_importance.csv # Coefficients
│   │   └── test_predictions.csv         # Model outputs
│   ├── notebook/                        # Jupyter notebooks (development)
│   └── src/
│       ├── merge.py                     # Data merging script
│       ├── drop_columns.py              # Column removal
│       ├── data_preprocessing.py        # Filtering, scaling, splitting
│       ├── feature_engineering.py       # Feature creation
│       ├── extend_dataset.py            # Add lag/rolling features
│       ├── train_models.py              # Model training pipeline
│       └── featureimp.py                # Feature importance analysis
```

---

## Technical Details

### Dataset
- **Size:** 852 days (Jan 1, 2023 - May 1, 2025)
- **Sources:** 
  - Lahore AQI (Open-Meteo API) (PM2.5, PM10, O₃, CO, SO₂, NO₂)
  - Lahore Weather (Meteostat) (temperature, humidity, wind, pressure, precipitation)
  - NASA FIRMS (fire radiative power)
- **Features:** 93 engineered features (after variance/correlation filtering)
- **Split:** 80% train / 10% validation / 10% test (chronological, no shuffle)

### Feature Engineering
- **Temporal (18):** Year, month, day_of_week, week_of_year, is_weekend, season indicators, cyclical encodings
- **Lag Features (32):** 1, 7, 14, 30-day lags for PM2.5, PM10, O₃, NO₂, fire
- **Rolling Statistics:** 3, 7, 30-day means and std deviations
- **Interactions (5):** Temp×Humidity, Wind×Fire, PM10×O₃, inversion risk, crop fire intensity
- **Extreme Pollution (6):** High pollution indicators, streaks, stagnant air, winter smog risk
- **Ratios:** PM2.5/PM10, NO₂/CO

### Best Model: Lasso Regression
- **RMSE:** 19.69 µg/m³
- **MAE:** 16.31 µg/m³
- **R²:** 0.837 (explains 83.7% of variance)
- **MAPE:** 17.85%
- **Alpha:** 0.1 (from GridSearchCV)
- **Strongest Predictor:** pm10_ozone_interaction (coefficient: 28.32)

### Why Lasso Won
Linear models outperformed complex ensembles because:
1. Extensive feature engineering captured non-linear patterns explicitly
2. L1 regularization performed automatic feature selection
3. Variance/correlation filtering removed redundant features
4. Z-score standardization ensured equal feature weighting
5. Sample weighting (2× for PM2.5 > 150) focused learning on extreme events

---

## Usage Guide

### Making Predictions

1. **Launch App:** Run `streamlit run Predict.py` from `project/app/`
2. **Select Model:** Choose from sidebar (Lasso recommended)
3. **Pick Reference Date:** Select date with 30+ days prior history
4. **Enter Values:** Input current air quality and weather data
   - Or use Quick Presets for testing
5. **Get Prediction:** Click "Predict Next Day PM2.5"
6. **View Results:** 
   - Predicted PM2.5 with AQI category
   - Health implications
   - Actual vs Predicted comparison (if available)
7. **Export:** Download CSV for records

### Exploring Visualizations

1. **Navigate:** Click "Visualize" in sidebar
2. **View Dashboard:** See 5 KPI metric cards
3. **Scroll Through Charts:** 10 visualizations with insights
4. **Interact:** 
   - Hover for details
   - Zoom and pan on time-series
   - Click legend to toggle traces
5. **Read Insights:** Each chart has colored insight boxes

### Understanding Methodology

1. **Navigate:** Click "About" in sidebar
2. **Read Sections:**
   - Problem Statement
   - Dataset Information
   - Data Processing Pipeline (Steps 1-4)
   - Model Training
   - Visualization Strategy
   - Key Findings
3. **Cross-Reference:** All claims verified against actual src/ code

---

## Development

### Running from Source

```bash
# Clone repository
git clone https://github.com/Abdullah-Farooq-5/master-forecaster.git
cd master-forecaster

# Install dependencies
cd project/app
pip install -r ../requirements.txt

# Run pipeline (if rebuilding from scratch)
cd ../src
python merge.py
python drop_columns.py
python data_preprocessing.py
python feature_engineering.py
python extend_dataset.py
python train_models.py
python featureimp.py

# Launch dashboard
cd ../app
streamlit run Predict.py
```

### Adding New Features

1. Modify `feature_engineering.py` to add feature
2. Update `data_preprocessing.py` to handle new feature
3. Retrain models: `python train_models.py`
4. Update `feature_info.json` with new count
5. Test predictions in dashboard

### Customizing UI

- **Colors:** Modify CSS in `st.markdown()` blocks at top of each page
- **Charts:** Edit Plotly figure configurations in visualization functions
- **Layout:** Adjust `st.columns()` ratios for responsive design
- **Content:** Update info boxes with new insights

---

## Performance

### Load Times
- **Initial load:** 2-3 seconds (caching data and models)
- **Page navigation:** <1 second
- **Prediction:** <0.5 seconds
- **Chart rendering:** <1 second per visualization

### Resource Usage
- **Memory:** ~400 MB (with all 7 models loaded)
- **CPU:** <5% on modern systems
- **Storage:** ~60 MB (models + data)

---

## Troubleshooting

### App won't start
**Error:** `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install -r ../requirements.txt
```

### Data files not found
**Error:** `FileNotFoundError: ../data/processed/merged_data_clean.csv`
- Make sure you're running from `project/app/` directory
- Check that data files exist: `ls ../data/processed/`
- Run preprocessing pipeline from `project/src/`

### Models not loading
**Error:** `Error loading model lasso`
- Ensure all 7 `.pkl` files exist in `../models/`
- Check scikit-learn version compatibility
- Verify `scaler.pkl` exists

### Prediction fails
**Error:** `Prediction Error: ...`
- Reference date needs 30+ days of prior data (select after Jan 31, 2023)
- Verify all input values are within reasonable ranges
- Check detailed error in expandable section

### Charts not displaying
- Update plotly: `pip install --upgrade plotly`
- Clear browser cache
- Try different browser (Chrome recommended)

### Wrong column count in data
- `merged_data_clean.csv` should have 19 columns (not 12)
- If you see 12, re-run `drop_columns.py` from `src/`

---

## Known Issues & Limitations

1. **Historical Data Requirement:** Predictions require 30-day history, limiting forecasts to dates after Jan 31, 2023
2. **Single Day Forecast:** Models predict only next-day PM2.5 (not multi-day)
3. **Feature Dependencies:** 93 features must match exact training configuration
4. **No Real-Time Updates:** Data is static (last update: May 1, 2025)
5. **Lahore-Specific:** Models trained on Lahore data only, not generalizable to other cities

---

## Future Enhancements

- [ ] Multi-day forecasting (7-day predictions)
- [ ] Real-time API integration (live AQI updates)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Mobile-responsive design improvements
- [ ] User authentication and prediction history
- [ ] Additional cities (Islamabad, Karachi)
- [ ] Alert system (email/SMS for hazardous AQI)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Credits

**Development Team:**
- **Abdullah Farooq** - ML Pipeline, Feature Engineering, Model Training
- **Huzaifa Ali Satti** - Data Collection, Preprocessing, Visualization
- **M. Shahzil Asif** - Dashboard Development, UI/UX Design

**Course Information:**
- **CS-245 Machine Learning** 
- **Instructor:** Mr. Usama Athar
- **CS-366 Data Visualization**
- **Instructor:** Mrs. Nikhar Azhar 
- **Institution:** NUST School of Electrical Engineering and Computer Science (SEECS)

**Data Sources:**
- Open Meteo API
- Meteostat
- NASA FIRMS (Fire Information for Resource Management System)

---

## Contact & Support

**GitHub Repository:** [master-forecaster](https://github.com/Abdullah-Farooq-5/master-forecaster)

For issues or questions:
1. Check this README thoroughly
2. Review the **About** page in the dashboard
3. Open an issue on GitHub
4. Contact: chaudharyabdullah387@gmail.com

---

