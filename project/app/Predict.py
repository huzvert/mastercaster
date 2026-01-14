import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import joblib
from pathlib import Path
from feature_engineer import engineer_features_for_date, get_feature_names

# Get the correct base path for file loading
BASE_DIR = Path(__file__).parent.parent

# Page configuration
st.set_page_config(
    page_title="PM2.5 Forecasting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    /* Main header styling */
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
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #cccccc;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #d0d0d0;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-color: #0066cc;
    }
    
    .info-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    /* Button styling */
    .stButton>button {
        background: #0066cc;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.65rem 1.8rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: #0052a3;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3);
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input {
        border-radius: 6px;
        border: 1.5px solid #999999;
        padding: 0.5rem;
        transition: border-color 0.2s ease;
        color: #1a1a1a;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #0066cc;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f8f8;
    }
    
    /* AQI badge */
    .aqi-badge {
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .aqi-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .aqi-category {
        font-size: 1.5rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Feature tag */
    .feature-tag {
        display: inline-block;
        background: #2C5F7C;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'random_date_data' not in st.session_state:
    st.session_state.random_date_data = None
if 'use_random_date' not in st.session_state:
    st.session_state.use_random_date = False

@st.cache_resource
def load_data():
    """Load historical data for lag/rolling features."""
    try:
        data_path = BASE_DIR / "data" / "processed" / "merged_data_extended.csv"
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    """Load feature scaler."""
    try:
        scaler_path = BASE_DIR / "models" / "scaler.pkl"
        return joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

@st.cache_resource
def load_model(model_name):
    """Load trained model."""
    try:
        model_path = BASE_DIR / "models" / f"{model_name}_model.pkl"
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

@st.cache_data
def load_model_performance():
    """Load model performance metrics from training results."""
    try:
        results_path = BASE_DIR / "models" / "training_results.json"
        with open(results_path, "r") as f:
            results = json.load(f)
        return results
    except:
        return None

def get_aqi_category(pm25_value):
    """Get AQI category and color based on PM2.5 value (US EPA standard)."""
    if pm25_value <= 12.0:
        return {
            'category': 'Good',
            'color': '#27AE60',
            'health_implications': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'recommendation': 'Great day for outdoor activities!'
        }
    elif pm25_value <= 35.4:
        return {
            'category': 'Moderate',
            'color': '#F39C12',
            'health_implications': 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.',
            'recommendation': 'Generally safe, but sensitive individuals should limit prolonged outdoor exertion.'
        }
    elif pm25_value <= 55.4:
        return {
            'category': 'Unhealthy for Sensitive Groups',
            'color': '#E67E22',
            'health_implications': 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
            'recommendation': 'Children, elderly, and people with respiratory conditions should reduce outdoor activity.'
        }
    elif pm25_value <= 150.4:
        return {
            'category': 'Unhealthy',
            'color': '#E74C3C',
            'health_implications': 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
            'recommendation': 'Everyone should limit prolonged outdoor exertion. Wear masks outdoors.'
        }
    elif pm25_value <= 250.4:
        return {
            'category': 'Very Unhealthy',
            'color': '#8E44AD',
            'health_implications': 'Health alert: The risk of health effects is increased for everyone.',
            'recommendation': 'Avoid outdoor activities. Stay indoors with air purifiers.'
        }
    else:
        return {
            'category': 'Hazardous',
            'color': '#943126',
            'health_implications': 'Health warning of emergency conditions: everyone is more likely to be affected.',
            'recommendation': 'Emergency conditions! Stay indoors. Avoid all outdoor activity.'
        }

def validate_inputs(**kwargs):
    """Validate user inputs for reasonable ranges."""
    errors = []
    
    # Pollution parameters
    if kwargs.get('pm10', 0) > 600:
        errors.append("PM10 value seems unusually high (>600 µg/m³)")
    if kwargs.get('ozone', 0) > 400:
        errors.append("Ozone value seems unusually high (>400 µg/m³)")
    if kwargs.get('carbon_monoxide', 0) > 15000:
        errors.append("CO value seems unusually high (>15000 µg/m³)")
    
    # Weather parameters
    if kwargs.get('tmax', 0) < -20 or kwargs.get('tmax', 0) > 55:
        errors.append("Temperature outside normal range for Lahore (-20 to 55°C)")
    if kwargs.get('wspd', 0) > 30:
        errors.append("Wind speed seems unusually high (>30 m/s)")
    
    return errors

def main():
    # Header
    st.markdown('<p class="main-header">Master Forecaster</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced PM2.5 Air Quality Prediction System for Lahore, Pakistan</p>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    scaler = load_scaler()
    
    if df is None or scaler is None:
        st.error("❌ Failed to load required data files. Please check file paths.")
        st.stop()
    
    # Get date range
    min_date = df['date'].iloc[30].date()
    max_date = df['date'].max().date()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model selection with descriptions
        st.markdown("### Select ML Model")
        model_descriptions = {
            "Lasso Regression": "Best Overall (R²=0.837)",
            "Ridge Regression": "High Performance (R²=0.824)",
            "XGBoost": "Gradient Boosting (R²=0.750)",
            "LightGBM": "Fast & Efficient (R²=0.727)",
            "Gradient Boosting": "Ensemble Method (R²=0.752)",
            "Random Forest": "Tree-based (R²=0.672)",
            "SVR": "Support Vector (R²=-0.304)"
        }
        
        model_options = {
            "Lasso Regression": "lasso",
            "Ridge Regression": "ridge",
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Gradient Boosting": "gradient_boosting",
            "Random Forest": "random_forest",
            "SVR": "svr"
        }
        
        selected_model_name = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            index=0,
            format_func=lambda x: f"{x} - {model_descriptions[x]}",
            help="Select the machine learning model for prediction"
        )
        
        selected_model = model_options[selected_model_name]
        
        st.markdown("---")
        
        # Reference date selection
        st.markdown("### Reference Date")
        st.caption("Historical data context for lag and rolling features")
        
        selected_date = st.date_input(
            "Select reference date",
            value=datetime(2024, 11, 15).date(),
            min_value=min_date,
            max_value=max_date,
            help="This date provides historical context (lag features, rolling averages)"
        )
        
        st.markdown("---")
        
        # Random date generator
        st.markdown("### Random Date Test")
        
        if st.button("Generate Random Date", use_container_width=True, help="Randomly select a historical date and auto-fill values"):
            # Select random date from available range (with at least 30 days prior data)
            available_dates = df[df['date'] >= df['date'].iloc[30]]['date'].tolist()
            random_date = np.random.choice(available_dates)
            random_date = pd.to_datetime(random_date).date()
            
            # Get data for that date
            date_data = df[df['date'] == pd.to_datetime(random_date)].iloc[0]
            
            # Store in session state
            st.session_state.random_date_data = {
                'date': random_date,
                'pm10': float(date_data.get('pm10', 150.0)),
                'ozone': float(date_data.get('ozone', 100.0)),
                'co': float(date_data.get('carbon_monoxide', 3000.0)),
                'so2': float(date_data.get('sulphur_dioxide', 30.0)),
                'no2': float(date_data.get('nitrogen_dioxide', 80.0)),
                'tmax': float(date_data.get('tmax', 25.0)),
                'humidity': float(date_data.get('relative_humidity_2m', 65.0)),
                'prcp': float(date_data.get('prcp', 0.0)),
                'wspd': float(date_data.get('wspd', 2.0)),
                'pres': float(date_data.get('pres', 1013.0)),
                'fire_frp_max': float(date_data.get('fire_frp_max', 0.0)),
                'fire_frp_total': float(date_data.get('fire_frp_total', 0.0)),
                'actual_pm25': float(date_data.get('pm2_5_mean', 0.0))
            }
            st.session_state.use_random_date = True
            st.rerun()
        
        if st.session_state.random_date_data and st.session_state.use_random_date:
            st.success(f"Random date: {st.session_state.random_date_data['date'].strftime('%b %d, %Y')}")
            st.caption(f"Actual PM2.5: {st.session_state.random_date_data['actual_pm25']:.1f} µg/m³")
            if st.button("Clear Random Date", use_container_width=True):
                st.session_state.use_random_date = False
                st.session_state.random_date_data = None
                st.rerun()
    
    # Main content area
    st.markdown('<p class="section-header">Enter Current Conditions</p>', unsafe_allow_html=True)
    
    # Apply random date if active
    if st.session_state.use_random_date and st.session_state.random_date_data:
        # Use random date data
        defaults = st.session_state.random_date_data
        # Override selected_date with random date
        selected_date = st.session_state.random_date_data['date']
    else:
        # Use default values
        defaults = {}
    
    # Input sections with better organization
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### Air Quality Parameters")
        pm10 = st.number_input(
            "PM10 (µg/m³)", 
            min_value=0.0, max_value=600.0, 
            value=defaults.get('pm10', 150.0), 
            step=10.0,
            help="Particulate Matter 10 micrometers or less in diameter"
        )
        ozone = st.number_input(
            "Ozone (µg/m³)", 
            min_value=0.0, max_value=400.0, 
            value=defaults.get('ozone', 100.0), 
            step=10.0,
            help="Ground-level ozone (O₃) concentration"
        )
        carbon_monoxide = st.number_input(
            "Carbon Monoxide (µg/m³)", 
            min_value=0.0, max_value=15000.0, 
            value=defaults.get('co', 3000.0), 
            step=100.0,
            help="CO from vehicle emissions and combustion"
        )
        sulphur_dioxide = st.number_input(
            "Sulphur Dioxide (µg/m³)", 
            min_value=0.0, max_value=200.0, 
            value=defaults.get('so2', 30.0), 
            step=5.0,
            help="SO₂ from industrial processes and fuel combustion"
        )
        nitrogen_dioxide = st.number_input(
            "Nitrogen Dioxide (µg/m³)", 
            min_value=0.0, max_value=300.0, 
            value=defaults.get('no2', 80.0), 
            step=10.0,
            help="NO₂ primarily from vehicle emissions"
        )
    
    with col2:
        st.markdown("#### Weather Conditions")
        tmax = st.number_input(
            "Max Temperature (°C)", 
            min_value=-10.0, max_value=50.0, 
            value=defaults.get('tmax', 25.0), 
            step=1.0,
            help="Maximum temperature for the day"
        )
        relative_humidity = st.number_input(
            "Relative Humidity (%)", 
            min_value=0.0, max_value=100.0, 
            value=defaults.get('humidity', 65.0), 
            step=5.0,
            help="Percentage of moisture in the air"
        )
        prcp = st.number_input(
            "Precipitation (mm)", 
            min_value=0.0, max_value=200.0, 
            value=defaults.get('prcp', 0.0), 
            step=0.5,
            help="Rainfall amount"
        )
        wspd = st.number_input(
            "Wind Speed (m/s)", 
            min_value=0.0, max_value=30.0, 
            value=defaults.get('wspd', 2.0), 
            step=0.5,
            help="Wind speed - higher values help disperse pollution"
        )
        pres = st.number_input(
            "Atmospheric Pressure (hPa)", 
            min_value=900.0, max_value=1100.0, 
            value=defaults.get('pres', 1013.0), 
            step=1.0,
            help="Barometric pressure"
        )
    
    with col3:
        st.markdown("#### Fire Activity")
        fire_frp_max = st.number_input(
            "Fire FRP Max", 
            min_value=0.0, max_value=100.0, 
            value=defaults.get('fire_frp_max', 0.0), 
            step=1.0,
            help="Maximum Fire Radiative Power (crop burning indicator)"
        )
        fire_frp_total = st.number_input(
            "Fire FRP Total", 
            min_value=0.0, max_value=1000.0, 
            value=defaults.get('fire_frp_total', 0.0), 
            step=10.0,
            help="Total Fire Radiative Power from all fires"
        )
        
        st.markdown("#### Quick Stats")
        if st.session_state.use_random_date and st.session_state.random_date_data:
            st.success(f"""
            **Random Test Mode**  
            **Reference Date:** {selected_date.strftime('%b %d, %Y')}  
            **Target Date:** {(pd.to_datetime(selected_date) + timedelta(days=1)).strftime('%b %d, %Y')}  
            **Actual PM2.5 (Today):** {st.session_state.random_date_data['actual_pm25']:.1f} µg/m³  
            **Model:** {selected_model_name}
            """)
        else:
            st.info(f"""
            **Reference Date:** {selected_date.strftime('%b %d, %Y')}  
            **Target Date:** {(pd.to_datetime(selected_date) + timedelta(days=1)).strftime('%b %d, %Y')}  
            **Model:** {selected_model_name}
            """)
    
    # Validate inputs
    validation_errors = validate_inputs(
        pm10=pm10, ozone=ozone, carbon_monoxide=carbon_monoxide,
        tmax=tmax, wspd=wspd
    )
    
    if validation_errors:
        with st.expander("Input Validation Warnings", expanded=True):
            for error in validation_errors:
                st.warning(error)
    
    st.markdown("---")
    
    # Predict button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "Predict Tomorrow's PM2.5", 
            type="primary", 
            use_container_width=True
        )
    
    # Make prediction
    if predict_button:
        try:
            with st.spinner("Engineering features and making prediction..."):
                # Create a temporary dataframe with user inputs
                user_data = df.copy()
                
                # Update the selected date row with user inputs
                date_idx = user_data[user_data['date'] == pd.to_datetime(selected_date)].index
                
                if len(date_idx) == 0:
                    st.error(f"❌ Date {selected_date} not found in dataset!")
                    st.stop()
                
                date_idx = date_idx[0]
                
                # Update with user inputs
                user_data.loc[date_idx, 'pm10'] = pm10
                user_data.loc[date_idx, 'ozone'] = ozone
                user_data.loc[date_idx, 'carbon_monoxide'] = carbon_monoxide
                user_data.loc[date_idx, 'sulphur_dioxide'] = sulphur_dioxide
                user_data.loc[date_idx, 'nitrogen_dioxide'] = nitrogen_dioxide
                user_data.loc[date_idx, 'tmax'] = tmax
                user_data.loc[date_idx, 'relative_humidity_2m'] = relative_humidity
                user_data.loc[date_idx, 'prcp'] = prcp
                user_data.loc[date_idx, 'wspd'] = wspd
                user_data.loc[date_idx, 'pres'] = pres
                user_data.loc[date_idx, 'fire_frp_max'] = fire_frp_max
                user_data.loc[date_idx, 'fire_frp_total'] = fire_frp_total
                
                # Calculate derived weather features
                tmin = user_data.loc[date_idx, 'tmin']
                tavg = (tmax + tmin) / 2
                user_data.loc[date_idx, 'tavg'] = tavg
                
                # Engineer features
                features = engineer_features_for_date(user_data, selected_date)
                
                # Get feature names in correct order
                feature_names = get_feature_names()
                feature_vector = features[feature_names].values.reshape(1, -1)
                
                # Handle NaN values
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Load model and predict
                model = load_model(selected_model)
                if model is None:
                    st.error("Failed to load model")
                    st.stop()
                
                predicted_pm25 = model.predict(feature_vector_scaled)[0]
                
                # Get actual value for next day if available
                next_day = pd.to_datetime(selected_date) + timedelta(days=1)
                actual_pm25 = None
                if next_day in df['date'].values:
                    actual_pm25 = df[df['date'] == next_day]['pm2_5_mean'].values[0]
                
                # Store result
                st.session_state.prediction_result = {
                    'prediction_date': selected_date,
                    'target_date': next_day.date(),
                    'predicted_pm25': float(predicted_pm25),
                    'actual_pm25': float(actual_pm25) if actual_pm25 is not None else None,
                    'model_name': selected_model_name,
                    'user_inputs': {
                        'pm10': pm10, 'ozone': ozone, 'co': carbon_monoxide,
                        'so2': sulphur_dioxide, 'no2': nitrogen_dioxide,
                        'tmax': tmax, 'humidity': relative_humidity,
                        'prcp': prcp, 'wspd': wspd, 'pres': pres,
                        'fire_frp_max': fire_frp_max, 'fire_frp_total': fire_frp_total
                    }
                }
                
                st.success("Prediction completed successfully")
                
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)
        
        # Date info
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <h4 style="color: #1a1a1a;">Reference Date</h4>
                <h2 style="color: #0066cc;">{}</h2>
                <p style="color: #333333;">Today's conditions</p>
            </div>
            """.format(result['prediction_date'].strftime('%b %d, %Y')), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <h4 style="color: #1a1a1a;">Target Date</h4>
                <h2 style="color: #0066cc;">{}</h2>
                <p style="color: #333333;">Prediction for tomorrow</p>
            </div>
            """.format(result['target_date'].strftime('%b %d, %Y')), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <h4 style="color: #1a1a1a;">Model Used</h4>
                <h2 style="color: #0066cc;">{}</h2>
                <p style="color: #333333;">ML Algorithm</p>
            </div>
            """.format(result['model_name']), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main prediction with AQI category
        predicted_pm25 = result['predicted_pm25']
        aqi_info = get_aqi_category(predicted_pm25)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="aqi-badge" style="background-color: {aqi_info['color']};">
                <p class="aqi-value" style="color: white;">{predicted_pm25:.1f}</p>
                <p style="color: white; font-size: 1.2rem; margin: 0;">µg/m³</p>
                <p class="aqi-category" style="color: white;">{aqi_info['category']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Health Implications</h4>
                <p>{aqi_info['health_implications']}</p>
                <br>
                <h4>Recommendation</h4>
                <p><strong>{aqi_info['recommendation']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Actual vs Predicted comparison
        if result['actual_pm25'] is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Prediction Accuracy</p>', unsafe_allow_html=True)
            
            actual_pm25 = result['actual_pm25']
            error = predicted_pm25 - actual_pm25
            error_pct = abs(error / actual_pm25) * 100
            accuracy = 100 - error_pct
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted PM2.5", f"{predicted_pm25:.1f} µg/m³", 
                         help="Model prediction")
            with col2:
                st.metric("Actual PM2.5", f"{actual_pm25:.1f} µg/m³",
                         help="Real observed value")
            with col3:
                st.metric("Absolute Error", f"{abs(error):.1f} µg/m³", 
                         delta=f"{error:+.1f}",
                         delta_color="inverse",
                         help="Difference between predicted and actual")
            with col4:
                st.metric("Accuracy", f"{accuracy:.1f}%",
                         help="Prediction accuracy")
            
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Predicted',
                x=['PM2.5 Level'],
                y=[predicted_pm25],
                marker_color='#667eea',
                text=[f"{predicted_pm25:.1f}"],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=['PM2.5 Level'],
                y=[actual_pm25],
                marker_color='#ff7f0e',
                text=[f"{actual_pm25:.1f}"],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=dict(
                    text="Predicted vs Actual PM2.5 Comparison",
                    font=dict(size=20, color='#2c3e50')
                ),
                yaxis_title="PM2.5 Concentration (µg/m³)",
                barmode='group',
                height=400,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Input summary
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Input Parameters", expanded=False):
            inputs = result['user_inputs']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Air Quality**")
                st.write(f"• PM10: {inputs['pm10']:.1f} µg/m³")
                st.write(f"• Ozone: {inputs['ozone']:.1f} µg/m³")
                st.write(f"• CO: {inputs['co']:.1f} µg/m³")
                st.write(f"• SO₂: {inputs['so2']:.1f} µg/m³")
                st.write(f"• NO₂: {inputs['no2']:.1f} µg/m³")
            
            with col2:
                st.markdown("**Weather**")
                st.write(f"• Max Temp: {inputs['tmax']:.1f}°C")
                st.write(f"• Humidity: {inputs['humidity']:.1f}%")
                st.write(f"• Precipitation: {inputs['prcp']:.1f} mm")
                st.write(f"• Wind Speed: {inputs['wspd']:.1f} m/s")
                st.write(f"• Pressure: {inputs['pres']:.1f} hPa")
            
            with col3:
                st.markdown("**Fire Activity**")
                st.write(f"• FRP Max: {inputs['fire_frp_max']:.1f}")
                st.write(f"• FRP Total: {inputs['fire_frp_total']:.1f}")
        
        # Download results
        st.markdown("<br>", unsafe_allow_html=True)
        result_df = pd.DataFrame([{
            'Reference_Date': result['prediction_date'],
            'Target_Date': result['target_date'],
            'Predicted_PM25': result['predicted_pm25'],
            'Actual_PM25': result['actual_pm25'],
            'AQI_Category': aqi_info['category'],
            'Model': result['model_name'],
            **{k.upper(): v for k, v in result['user_inputs'].items()}
        }])
        
        csv = result_df.to_csv(index=False)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="Download Prediction Results (CSV)",
                data=csv,
                file_name=f"lahore_pm25_prediction_{result['target_date']}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Welcome screen
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h3>Welcome to Master Forecaster</h3>
            <p style="font-size: 1.05rem; color: #34495E;">
                An advanced machine learning system for predicting PM2.5 air quality levels in Lahore. 
                The system leverages historical air quality, weather, and fire data to generate accurate forecasts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>How to Use</h4>
                <ol style="font-size: 0.95rem;">
                    <li><strong>Select Model:</strong> Choose from 7 ML models in sidebar</li>
                    <li><strong>Pick Reference Date:</strong> Select date for historical context</li>
                    <li><strong>Enter Values:</strong> Input current pollution and weather data</li>
                    <li><strong>Get Prediction:</strong> Click predict to see tomorrow's PM2.5</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>System Overview</h4>
                <ul style="font-size: 0.95rem;">
                    <li><strong>93 Features:</strong> Temporal, lag, rolling, interactions</li>
                    <li><strong>Historical Context:</strong> Uses past 30 days of data</li>
                    <li><strong>7 ML Models:</strong> From simple regression to XGBoost</li>
                    <li><strong>High Accuracy:</strong> Best model R² = 0.837</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown('<p class="section-header">Sample Scenarios</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #dc3545; padding: 1.5rem; border-radius: 8px; color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
                <h4 style="color: white; margin-top: 0;">Winter Smog</h4>
                <p><strong>High Pollution Scenario</strong></p>
                <ul style="font-size: 0.9rem; line-height: 1.6;">
                    <li>PM10: 300+ µg/m³</li>
                    <li>Temp: 10-15°C</li>
                    <li>Wind: <2 m/s</li>
                    <li>Humidity: >80%</li>
                </ul>
                <p><em>Expected: Hazardous AQI</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #ff9800; padding: 1.5rem; border-radius: 8px; color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
                <h4 style="color: white; margin-top: 0;">Monsoon</h4>
                <p><strong>Moderate Scenario</strong></p>
                <ul style="font-size: 0.9rem; line-height: 1.6;">
                    <li>PM10: 100-150 µg/m³</li>
                    <li>Temp: 25-30°C</li>
                    <li>Wind: 3-5 m/s</li>
                    <li>Rain: 10-30 mm</li>
                </ul>
                <p><em>Expected: Moderate AQI</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #28a745; padding: 1.5rem; border-radius: 8px; color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
                <h4 style="color: white; margin-top: 0;">Summer Clear</h4>
                <p><strong>Low Pollution Scenario</strong></p>
                <ul style="font-size: 0.9rem; line-height: 1.6;">
                    <li>PM10: 40-80 µg/m³</li>
                    <li>Temp: 32-38°C</li>
                    <li>Wind: >5 m/s</li>
                    <li>Humidity: <50%</li>
                </ul>
                <p><em>Expected: Good/Moderate AQI</em></p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
