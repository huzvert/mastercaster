import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Documentation",
    page_icon="📖",
    layout="wide"
)

# CSS matching Visualize.py
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem auto 1rem auto;
        padding: 2rem 1rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        text-shadow: 0 6px 12px rgba(0,0,0,0.35);
        line-height: 1.1;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .info-box h4 {
        color: #000000;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .info-box strong {
        color: #000000;
    }
    
    .info-box ul, .info-box ol {
        color: #333333;
        margin-left: 1.5rem;
    }
    
    .info-box p {
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    .info-box li {
        margin-bottom: 0.5rem;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .warning-box h4 {
        color: #000000;
        margin-top: 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .success-box h4 {
        color: #000000;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<p class="main-header">About Master Forecaster</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #cccccc; font-size: 1.1rem; margin-bottom: 2rem;">Lahore Air Quality Prediction & Visualization System</p>', unsafe_allow_html=True)
    
    # ==================== PROBLEM STATEMENT ====================
    st.markdown('<p class="section-header">Problem Statement</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h4>Machine Learning</h4>
            <p><strong>Goal:</strong> Build a predictive model to forecast next-day PM2.5 air quality levels in Lahore using historical air quality, weather, and fire data.</p>
            <p><strong>Challenge:</strong> Air quality is influenced by multiple interacting factors (weather conditions, pollutant interactions, seasonal patterns). We need to:</p>
            <ul>
                <li>Engineer meaningful features that capture these complex relationships</li>
                <li>Handle missing data and outliers in real-world sensor data</li>
                <li>Select the best performing model from multiple ML algorithms</li>
                <li>Achieve accurate predictions (low RMSE/MAE, high R²)</li>
            </ul>
            <p><strong>Target:</strong> Predict PM2.5 concentration (µg/m³) for the next day</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Data Visualization</h4>
            <p><strong>Goal:</strong> Create diverse, interactive visualizations to reveal patterns, trends, and insights in Lahore's air quality data.</p>
            <p><strong>Challenge:</strong> Transform 850+ days of multi-dimensional pollution data into clear, actionable insights. We need to:</p>
            <ul>
                <li>Show temporal patterns (seasonal trends, year-over-year changes)</li>
                <li>Reveal correlations between weather and pollution</li>
                <li>Demonstrate model performance and prediction accuracy</li>
                <li>Use variety of chart types (avoid repetition)</li>
                <li>Make visualizations interactive and user-friendly</li>
            </ul>
            <p><strong>Target:</strong> 10+ diverse visualizations with dashboard + ML performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== DATASET INFORMATION ====================
    st.markdown('<p class="section-header">Dataset Information</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Data Sources</h4>
        <p>Our project combines three datasets covering January 2023 to May 2025 (850+ days):</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Air Quality Index Data (Open-Meteo API)**")
        st.markdown("""
        - PM2.5 (mean, max, min)
        - PM10 (mean, max, min)
        - Ozone (O₃)
        - Carbon Monoxide (CO)
        - **Source:** https://www.kaggle.com/datasets/alihussnainnawaz/lahore-weather-and-aqi-dataset-from-open-meteo-api
        - **Frequency:** Daily measurements
        """)
    
    with col2:
        st.markdown("**Weather Data (Meteostat)**")
        st.markdown("""
        - Temperature (max, min, mean)
        - Precipitation (rain)
        - Wind speed (wspd)
        - Humidity
        - **Source:** https://meteostat.net/en/place/pk/lahore?s=41640&t=2023-01-01/2025-05-01
        - **Frequency:** Daily readings
        """)
    
    with col3:  
        st.markdown("**Fire Activity Data (NASA FIRMS (VIIRS Satellite))**")
        st.markdown("""
        - Fire count (daily)
        - Brightness
        - FRP (Fire Radiative Power)
        - **Source:** https://firms.modaps.eosdis.nasa.gov/ https://firms.modaps.eosdis.nasa.gov/data/download/DL_FIRE_J1V-C2_693026.zip
        - **Frequency:** Daily detections
        """)
    
    st.markdown("""
    <div class="success-box">
        <h4>Final Dataset Statistics</h4>
        <ul>
            <li><strong>Total Records:</strong> 850+ days (Jan 2023 - May 2025)</li>
            <li><strong>Features After Engineering:</strong> 100+ features created, reduced after variance/correlation filtering</li>
            <li><strong>Target Variable:</strong> PM2.5 (next day)</li>
            <li><strong>Train/Val/Test Split:</strong> 80% / 10% / 10% (chronological, no shuffle)</li>
            <li><strong>Data Quality:</strong> Cleaned, no missing values, no statistical outlier capping (real pollution events preserved)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== DATA PIPELINE ====================
    st.markdown('<p class="section-header">Data Processing Pipeline</p>', unsafe_allow_html=True)
    
    st.markdown("### Step 1: Data Collection & Merging")
    st.markdown("""
    <div class="info-box">
        <p><strong>Process:</strong></p>
        <ul>
            <li>Collected <code>aqi.csv</code>, <code>weather.csv</code>, <code>fire.csv</code> from different sources</li>
            <li>Merged all three datasets on the <code>date</code> column using outer join</li>
            <li>Aligned temporal data to ensure consistency across sources</li>
            <li><strong>Output:</strong> <code>merged_data.csv</code> (850 rows, 19 initial columns)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 2: Data Cleaning")
    st.markdown("""
    <div class="info-box">
        <p><strong>Process:</strong></p>
        <ul>
            <li><strong>Missing Value Handling:</strong>
                <ul>
                    <li>Fire data: Filled with 0 (no fires detected on missing days)</li>
                    <li>Precipitation: Filled with 0 (assume no rain if missing)</li>
                    <li>Dropped columns with >50% missing values</li>
                </ul>
            </li>
            <li><strong>Outlier Handling Philosophy:</strong> NO statistical outlier capping applied - pollution spikes are real environmental events, not statistical anomalies</li>
            <li><strong>Sanity Checks Only:</strong> PM2.5 clipped at 1000 µg/m³ max (sensor error threshold), ensured non-negative pollutant values</li>
            <li><strong>Redundant Column Removal:</strong> Dropped duplicate features (temperature_2m, wind_speed_10m, pressure_msl, wind_direction_10m, fire_frp_mean, fire_brightness_avg, pm2_5_min)</li>
            <li><strong>Data Type Correction:</strong> Converted date strings to datetime, ensured numeric types</li>
            <li><strong>Output:</strong> <code>merged_data_clean.csv</code> (850 rows, 12 columns, 0 missing values)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 3: Feature Engineering (ML)")
    st.markdown("""
    <div class="info-box">
        <p><strong>Process:</strong> Created 100+ engineered features to capture complex relationships</p>
        <ul>
            <li><strong>Temporal Features (18):</strong> Year, month, day_of_week, day_of_year, week_of_year, is_weekend, seasonal indicators (is_winter, is_spring, is_summer, is_autumn), is_smog_season, is_crop_burning, cyclical encodings (month_sin/cos, dow_sin/cos)</li>
            <li><strong>Lag Features (32):</strong> PM2.5 (mean/max/std), PM10, O₃, NO₂, fire count, fire FRP for 1, 7, 14, 30 days back</li>
            <li><strong>Rolling Statistics (many):</strong> 3-day, 7-day, 30-day mean/std for PM2.5, PM10, O₃, temperature, humidity, fire activity</li>
            <li><strong>Derived Weather (6):</strong> Temperature range, days since rain, 7-day cumulative precipitation, temperature anomaly, wind chill</li>
            <li><strong>Interaction Features (5):</strong> Temp×Humidity, Wind×Fire, PM10×O₃, inversion risk, crop fire intensity</li>
            <li><strong>Pollutant Ratios (2):</strong> PM2.5/PM10, NO₂/CO ratios</li>
            <li><strong>Fire Features (3):</strong> Binary fire indicator, 3-day cumulative fire count, fire intensity per fire</li>
            <li><strong>Extreme Pollution Detection (6):</strong> Strong inversion, fire weather risk, high pollution streak, stagnant air, winter smog risk, humidity-wind trap</li>
            <li><strong>Data Leakage Removal:</strong> Dropped pm2_5_max and pm2_5_std (same-day values), kept only lagged versions</li>
        </ul>
        <p><strong>Rationale:</strong> Linear models need explicit feature engineering. Interaction terms (PM10×O₃) became strongest predictor (coef: 28.3). Extreme pollution detection features help identify smog events.</p>
        <p><strong>Output:</strong> <code>features_engineered.csv</code> (100+ raw features before preprocessing)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 3.5: Feature Selection & Preprocessing (ML)")
    st.markdown("""
    <div class="info-box">
        <p><strong>Process:</strong> Reduced feature set and prepared data for training</p>
        <ul>
            <li><strong>Low Variance Removal:</strong> Dropped features with variance < 0.01 (constant or near-constant values)</li>
            <li><strong>High Correlation Removal:</strong> Eliminated highly correlated features (correlation > 0.95) to reduce multicollinearity</li>
            <li><strong>Z-Score Standardization:</strong> Scaled all features to mean=0, std=1 for better model convergence</li>
            <li><strong>Train/Val/Test Split:</strong> Chronological split - 80% train / 10% validation / 10% test (time-series aware)</li>
            <li><strong>Sample Weighting:</strong> Applied 2× weight to extreme pollution days (PM2.5 > 150 µg/m³) to improve model focus on critical events</li>
        </ul>
        <p><strong>Rationale:</strong> Remove redundant features, ensure numerical stability, preserve temporal order (no shuffle in time-series), emphasize learning from extreme pollution events.</p>
        <p><strong>Output:</strong> <code>train_data.csv</code>, <code>val_data.csv</code>, <code>test_data.csv</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Step 4: Dataset Reduction (DV)")
    st.markdown("""
    <div class="info-box">
        <p><strong>For Visualization:</strong> Used the cleaned dataset without heavy feature engineering</p>
        <ul>
            <li><strong>Why?</strong> Visualizations focus on interpretable patterns in original variables (temperature, wind, PM2.5)</li>
            <li><strong>What we kept:</strong> All 19 columns from <code>merged_data_clean.csv</code> (date + 18 features: PM2.5 metrics, pollutants, weather, fire data)</li>
            <li><strong>What we added:</strong> Simple derived columns (year, month, season, AQI category) for grouping</li>
            <li><strong>Benefit:</strong> Charts remain interpretable - users understand "Temperature vs PM2.5" better than "Lag7_PM2.5_rolling14_mean vs Interaction_PM10_O3"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== MODEL TRAINING ====================
    st.markdown('<p class="section-header">Model Training & Selection</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Models Trained</h4>
        <p>We trained and compared 11 models across three categories:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Baseline Models**")
        st.markdown("""
        - Mean Baseline
        - Median Baseline
        - Persistence (yesterday's value)
        - 7-day Moving Average
        
        **Purpose:** Establish minimum performance threshold
        """)
    
    with col2:
        st.markdown("**Classical ML Models**")
        st.markdown("""
        - Lasso Regression ⭐
        - Ridge Regression
        - Support Vector Regression (SVR)
        
        **Hyperparameter Tuning:** GridSearchCV with 5-fold TimeSeriesSplit CV
                    
        **Sample Weighting:** 2× weight for PM2.5 > 150 µg/m³
        """)
    
    with col3:
        st.markdown("**Ensemble Models**")
        st.markdown("""
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - LightGBM
        
        **Tuned:** Learning rate, max depth, n_estimators, subsample
                    
        **Sample Weighting:** Applied to all models for extreme pollution focus
        """)
    
    st.markdown("""
    <div class="success-box">
        <h4>Best Model: Lasso Regression</h4>
        <ul>
            <li><strong>RMSE:</strong> 19.69 µg/m³</li>
            <li><strong>MAE:</strong> 16.31 µg/m³</li>
            <li><strong>R² Score:</strong> 0.837 (explains 83.7% of variance)</li>
            <li><strong>MAPE:</strong> 17.85%</li>
            <li><strong>Best Parameter:</strong> alpha = 0.1</li>
        </ul>
        <p><strong>Why Lasso won:</strong> Extensive feature engineering (100+ features with interactions, lags, rolling stats, extreme pollution detectors) made the relationship effectively linear. Lasso's L1 regularization performed automatic feature selection, identifying PM10×O₃ interaction as strongest predictor. After variance/correlation filtering and proper scaling, linear models outperformed complex ensembles because engineered features already captured non-linear patterns. Sample weighting (2× for extreme pollution) helped model focus on critical smog events.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== VISUALIZATION APPROACH ====================
    st.markdown('<p class="section-header">Visualization Strategy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>10 Visualizations Implemented</h4>
        <p>Our visualization page demonstrates variety, interactivity, and purposeful design:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Exploration (Viz 1-7)**")
        st.markdown("""
        1. **Time Series:** Daily PM2.5 trends with moving average and filters
        2. **Box Plots:** Seasonal distribution (Winter/Spring/Summer/Fall)
        3. **Correlation Heatmap:** 9×9 feature relationships
        4. **Calendar Heatmap:** Monthly-daily pollution grid
        5. **Pie Chart:** AQI category distribution
        6. **Polar Wind Rose:** PM2.5 by wind speed bins
        7. **Bar Chart:** Year-over-year monthly comparison
        """)
    
    with col2:
        st.markdown("**ML Performance (Viz 8-10)**")
        st.markdown("""
        8. **Model Comparison:** 4-panel subplot (RMSE, MAE, R², MAPE)
        9. **Prediction Accuracy:** Actual vs Predicted + error histogram
        10. **Feature Importance:** Top 15 Lasso coefficients
        
        **Interactive Elements:**
        - Year filters
        - Moving average toggle
        - Color-by options (season/year/category)
        """)
    
    st.markdown("""
    <div class="success-box">
        <h4>Design Principles</h4>
        <ul>
            <li><strong>Variety:</strong> 10 different chart types - line, box, heatmap (×2), pie, polar, bar (×3), histogram, subplots</li>
            <li><strong>Interactivity:</strong> Filters, checkboxes, radio buttons for user exploration</li>
            <li><strong>Dashboard:</strong> Executive KPI cards (avg PM2.5, peak, good days, unhealthy days, total)</li>
            <li><strong>Clarity:</strong> Each visualization includes insight box explaining key findings</li>
            <li><strong>Color Coding:</strong> EPA-standard AQI colors, gradient backgrounds, high-contrast text</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== KEY FINDINGS ====================
    st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
            <h4>Environmental Insights (DV)</h4>
            <ul>
                <li>Winter PM2.5 (~120 µg/m³) is 3× higher than summer (~40 µg/m³)</li>
                <li>Temperature shows strong negative correlation (r = -0.65) with PM2.5</li>
                <li>Calm winds (<2 m/s) triple PM2.5 levels vs high wind (>6 m/s)</li>
                <li>Mid-November to early January are annual crisis periods</li>
                <li>Only 15% of days had "Good" air quality over 2.5 years</li>
                <li>2024 showed improvement over 2023; 2025 continuing positive trend</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>ML Performance Insights</h4>
            <ul>
                <li>Lasso achieves 83.7% variance explained (R² = 0.837)</li>
                <li>Average prediction error: 16.3 µg/m³ (MAE)</li>
                <li>PM10×Ozone interaction is strongest predictor (coef: 28.3)</li>
                <li>Feature engineering > complex models (Lasso beat XGBoost)</li>
                <li>Prediction errors are normally distributed (unbiased model)</li>
                <li>Model slightly underestimates during pollution spikes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
